# research

A persistent, evidence-driven research environment.

Not an autonomous agent with research tools bolted on: a store of
investigations, evidence, claims and provenance, with a constrained set of
research operations over it. Models supply semantic judgement. Storage,
identity, deduplication, budgets and provenance are ordinary code.

All nine milestones are implemented: the evidence foundation, the academic,
web/news and social slices, the claim/entity/event graph, the planner and
specialised research workers, bounded recursion with explicit stopping
criteria, a report assembled from stored state, and a read-only UI over the
same state. It has been run against live providers, not only fixtures — see
[Running against live sources](#running-against-live-sources).

## Try it

Everything below runs offline against a bundled corpus. No API keys, no
network.

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'

# An autonomous investigation: plan, work, recurse, stop. No network, no keys.
.venv/bin/research --offline investigate "Are small modular reactors competitive for AI data centres?"
.venv/bin/research report investigation:1 --no-summary

.venv/bin/research demo                     # the same corpus, scripted rather than planned
.venv/bin/research show evidence:9          # where did this come from?
.venv/bin/research independence investigation:1
.venv/bin/research claim new investigation:1 "SMR costs have risen above projections"
.venv/bin/research claim link investigation:1 claim:1 evidence:8 --excerpt "The target price for power"
.venv/bin/research claim show claim:1
.venv/bin/research questions investigation:1
.venv/bin/research timeline investigation:1 --publications
.venv/bin/research graph investigation:1
.venv/bin/research activity investigation:1
.venv/bin/research budget investigation:1

# Search what the investigation already holds - no provider call, no budget.
.venv/bin/research find investigation:1 "cost escalation"

# Read your own files in as evidence: PDF, text, Markdown, HTML, JSON.
.venv/bin/research ingest investigation:1 ./papers --type paper --family academic
```

`investigate` plans the work, runs specialised workers over the bundled
corpus, turns their follow-ups into new tasks within the depth and budget
limits, and stops for a recorded reason:

```
plan: Establish what the literature, the recent record and primary sources say…
  [task:1] academic: establish what peer-reviewed work says about …
  [task:2] news: establish what has recently been reported about …
  [task:3] primary_source: reach the underlying records behind reports about …
  [task:4] skeptic: find evidence that contradicts the emerging picture on …

-> task:1 academic: establish what peer-reviewed work says about small modular…
   completed in 6 steps; 6 evidence, 1 claims
   + task:5 skeptic (depth 2): test the claim behind: establish what peer-review…
…
stopped: no_open_tasks - every planned task has run and none proposed further work
```

`demo` runs the same corpus through a fixed script instead of a planner, and
leaves an investigation on disk. A sample of its output:

```
13 documents held; 11 independent sources
ID           TYPE                      PUBLISHED   TITLE                                  INDEPENDENCE
evidence:8   original_news_reporting   2023-11-08  NuScale and utility group terminat…    independent
evidence:9   secondary_news_reporting  2023-11-08  NuScale project ends after subscri…    syndicated_copy of evidence:8
evidence:10  original_news_reporting   2025-02-19  Tech companies sign nuclear power …    independent
evidence:11  secondary_news_reporting  2025-02-20  Tech giants are betting on nuclear…    derived_article of evidence:10
evidence:13  regulatory_document       2025-05-29  NRC issues standard design approva…    independent
```

Thirteen documents, eleven sources. A wire story carried by two outlets is
one source; a rewrite that credits another article is not corroboration of
it. That distinction is the point of the system.

Against live sources, drop `--offline`:

```bash
.venv/bin/research new "are small modular reactors competitive for AI data centres?"
.venv/bin/research search investigation:1 "small modular reactor levelized cost" --family academic
.venv/bin/research search investigation:1 "NuScale project cancellation" --family news
.venv/bin/research citations investigation:1 evidence:1 --direction both --depth 2
```

Academic sources (OpenAlex, Crossref, arXiv, Semantic Scholar) and the GDELT
news index need no credentials. Keyed web providers are used only if a key is
present; without one the system reports which sources it had rather than
failing.

## Running against live sources

Drop `--offline` and the same commands use real providers. Academic sources
(OpenAlex, Crossref, arXiv, Semantic Scholar), the GDELT news index, Bluesky
and Mastodon need no credentials; keyed web providers and YouTube are used
only if a key is present.

```bash
export RESEARCH_CONTACT_EMAIL=you@example.org
research new "are small modular reactors competitive for AI data centres?"
research search investigation:1 "small modular reactor levelized cost" --family academic
research citations investigation:1 evidence:9 --direction backward --depth 1
research search investigation:1 "nuclear reactor economics" --family social

# A whole investigation, driven by the built-in rule-based researcher rather
# than a paid model: a smoke test of a deployment that costs nothing.
research investigate "your question" --model offline
```

Live running is not the same as mocked running, and testing against real
providers changed the code. Some of what it found:

- Crossref rejects a whole request — HTTP 400, no results — if one field in
  `select` is not available on that route. `language` is returned in full
  records but is not selectable, so *every* Crossref search was failing.
- arXiv needs its terms combined deliberately. A quoted phrase matches
  nothing; a bag of words matches on any term and returns whatever is most
  cited for the commonest one — "small modular nuclear reactors economically
  competitive" came back full of neutrino physics; ANDing every term of a
  long query returns nothing at all. ANDing the leading four terms returns
  the reactor-economics literature that was asked for.
- OpenAlex answers a rate-limited call with `Retry-After: 76939` — about 21
  hours. The client was capping that at 30s and retrying twice, so one
  refusing provider cost 61 seconds of every search. A `Retry-After` longer
  than a few seconds now means "unavailable", not "wait".
- Federal sites label the padlock icon in their banner with an SVG
  `<title>Lock</title>`, which the extractor was reading as the document
  title.
- A shared egress IP gets rate-limited by OpenAlex, Semantic Scholar and
  GDELT. Retrying a provider that is refusing you costs the full retry budget
  on every search, so a provider that fails three times running is now
  skipped for the rest of the investigation — recorded, and reported, not
  silent.
- An empty result and an absent provider are different answers. With no
  keyed web provider configured, web search has no provider at all; that is
  now said plainly rather than returned as "nothing found". It also gets its
  own stopping reason: an investigation that ends because its sources were
  unreachable reports `sources_unavailable` — "a gap in coverage, not a
  finding" — rather than claiming diminishing returns on a topic it never
  searched.

The autonomous runs above used `--model offline`, the built-in rule-based
researcher. It exercises the whole loop without a credential, but it is a
decision rule, not a researcher: writing a good query is exactly the semantic
judgement a real model supplies, and the difference is visible in what comes
back.

## Core idea

Research is the domain, so the domain objects are research objects:

| Object | What it is |
| --- | --- |
| `Investigation` | A question and everything learned about it |
| `ResearchTask` | One tractable piece of it, recursively decomposable |
| `EvidenceDocument` | Externally retrieved material, normalised |
| `Claim` | A proposition evidence can support or contradict |
| `Entity` / `Event` | Actors and dated occurrences, evidence-backed |
| `Citation` | A document-to-document edge, resolved or still a frontier |
| `Provenance` | How a document was obtained, attached to every one |

Three things are kept strictly apart:

- **Evidence** is material retrieved from outside the system.
- **Claim** is a structured proposition about the world.
- **Analysis** is model reasoning about evidence and claims.

Analysis never becomes evidence. A model's summary of a paper is stored as
analysis on a claim link, not as a document; the document is what the
provider returned.

Investigation state lives in SQLite, outside any model context. A resumed
run reads tasks, evidence and budget from the store — it never replays a
conversation to work out what is known.

## Architecture

```
Research request
      |
      v
Operations  search_academic / search_news / search_web / fetch_source /
            follow_citations                      (research verbs, not endpoints)
      |
      v
Source registry ── routes intent to providers by capability and credentials
      |
      v
Adapters  openalex  crossref  arxiv  semantic_scholar  gdelt  brave/tavily/serper  fetch
      |
      v
Acquisition  normalise -> deduplicate -> charge budget -> persist with provenance
      |            ^
      |            |  ingest  local files and folders: PDF, text, Markdown, HTML
      v
Store  investigations, tasks, documents, claims, entities, events,
       citations, relationships, search_queries, source_fetches, budget_usage
      |
      v
Retrieval  lexical (BM25) + graph expansion + optional vectors, fused
      |
      v
Graph / CLI  citation traversal, independence report, activity log, inspection
```

Each layer depends only on the one below it, and a test walks the imports to
keep that true. The planning layer thinks
"search academic literature", never "call provider X endpoint Y"; provider
selection happens in the registry, below planning.

See [docs/architecture.md](docs/architecture.md) for the detail, including
how to add a source.

## What the system does with what it finds

**Deduplication and independence.** Four academic providers return one paper;
ten sites carry one wire story. Both collapse, by different rules:

- authoritative identifiers (DOI, arXiv, PubMed, provider id) → same record,
  merged, with each contributing provider credited;
- same canonical URL or content hash → same artifact;
- high body-text overlap across different domains → syndicated copy;
- an article that credits and links another held document → derived, not
  independent.

Copies are kept, not discarded: that a story was syndicated to nine outlets
is itself evidence about its spread. They share an `independence_key`, and
anything that counts sources counts groups.

**Citation traversal.** Backward through references to find the work a
literature rests on; forward through citing papers to find replications and
rebuttals. Every traversal declares depth, per-node and document limits
before it starts, and records which one stopped it. A cited work that has not
been fetched is stored as an unresolved edge — the frontier is visible
without spending anything to look at it.

**Autonomy with limits.** A planner decomposes the question into tasks;
specialised workers — scout, academic, news, primary-source, social, skeptic —
each get the slice of the action vocabulary their role needs, and a worker
that finds a lead worth its own work delegates it. Recursion is bounded three
ways: by depth, by the task budget, and by refusing objectives that repeat
work already planned. Without the third, workers recommend each other in
circles.

Workers act by emitting one JSON action per turn — `search_corpus`,
`search_academic`, `follow_citations`, `find_primary_source`,
`find_counterevidence`, `create_claim`, `link_evidence`, `get_evidence`,
`spawn_research_task`, `complete_research_task` and the rest. That vocabulary is closed: there is no
action that runs a command, reads a file, or reaches an address the
acquisition layer did not sanction. A worker returns identifiers and a
summary; retrieved text stays in the store.

**Claims.** A claim is a proposition; evidence is attached to it with a
stance, and the status follows from what is attached rather than from what a
worker believes:

```
claim:2  Modular factory construction will reverse historical nuclear cost escalation
status          contradicted
assessment      supported by 1 independent peer-reviewed source, 2022; contradicted by
                1 independent peer-reviewed source, 2023 (1 retracted item supporting it;
                the contradicting evidence is more recent than any supporting evidence)

-- what would strengthen this ------------------------------------------------
  - all support comes from a single source; look for independent confirmation
  - no primary record among the supporting evidence; trace the assertion to a
    filing, dataset, transcript or paper
```

There is no confidence score. Instead the evidence is characterised: how many
*independent* sources support it, whether any is a primary record, whether a
retraction applies, whether the contradicting evidence is newer than the
support, and whether the only backing is the announcing party's own material.
`research questions` lists the claims whose evidence is thin and says what
would fix each one.

Two integrity rules are enforced in code. A quotation attached to a claim must
actually appear in the document it is attributed to — a fabricated excerpt is
refused, not flagged. And model reasoning goes in its own field, so a report
can always separate what a source said from what a model concluded.

**Entities and events.** Identity is decided by authoritative identifiers
(ORCID, OpenAlex, ROR, Wikidata, SEC CIK) where they exist. Conflicting
identifiers mean different entities whatever the names say; ambiguous names
resolve to candidates rather than a merge; a match made on a name like
"J. Smith" is flagged for review rather than quietly trusted. Events need at
least one evidence document, and a dated event must state its precision, so a
timeline never implies certainty it does not have.

**Stopping on purpose.** An investigation ends because a rule fired, and the
rule is recorded: budget or runtime exhausted, no open tasks, diminishing
returns (a corpus that has become mostly copies, or a run of searches
returning nothing), or evidence sufficient (every claim evidenced with no
outstanding gaps). `research status` shows what each rule currently sees.

**Reports from the record.** The report is assembled from stored state —
claims with their assessments, evidence with its provenance, the timeline,
the open questions, the stopping reason. A model may write the executive
summary; it cannot introduce a fact, and a summary citing identifiers that do
not exist is discarded rather than published. Every heading the brief asks
for is there, and every finding carries the claim and evidence ids behind it.

**A UI, once the CLI worked.** `research ui` serves a read-only view of the
same stored state: the task tree with its spawned children, claims with their
excerpts and gaps, a claim/evidence graph that folds copies into the document
they copy, entities with name-only matches flagged, the timeline, the source
list, open questions and the full activity log. Only GET is answered, the
route table is fixed, and retrieved content reaches the browser as data and
is rendered as text — the client has no `innerHTML` and the page loads no
remote resources.

**Obsidian.** `research export obsidian <investigation> <vault>` writes the
investigation into a vault as notes, because the mapping is close to exact:
an investigation is already a graph of records joined by stable identifiers,
and a vault is a graph of notes joined by links.

```
Research/investigation-1 Are SMRs competitive…/
  investigation-1 ….md      the report, with claim:3 and evidence:12 as live links
  investigation-1 ….canvas  the claim/evidence graph, laid out deterministically
  Claims/     claim-1 The flagship project was terminated.md
  Evidence/   evidence-6 NuScale and utility group terminate….md
  Entities/   Events/   Tasks/
```

Each note carries its record in YAML frontmatter — source type, publisher,
DOI, content hash, provenance, independence — so Dataview queries and
Obsidian's own search work on it, and the identifier is an alias, so typing
`evidence:12` resolves to the right note. Three properties survive the
translation: a syndicated copy is not a second note in the graph (it is
listed on the note for the document it copies), every evidence note carries
the provider call and content hash that produced it, and retrieved text sits
in a callout marked external while model reasoning is labelled analysis.

A vault is a hostile rendering target — Obsidian renders HTML in an Electron
window, resolves `[[wikilinks]]` into real graph edges, and lets plugins
execute fenced blocks — so retrieved content is escaped for that context:
markup neutralised, links defanged, fences broken, and frontmatter written
through a real YAML serialiser so a title containing `---` cannot end the
block and spill into the note. The export writes only inside its own folder
and will not overwrite a note it did not generate without `--force`.

**Searching what it already holds.** Every search above calls a provider. The
cheapest question an investigation has is the one that does not:
`research find <investigation> <query>`, and the `search_corpus` action, read
the store instead. Three retrievers, fused by reciprocal rank fusion:

- **lexical** — SQLite FTS5 with BM25 and a porter stemmer, title and
  abstract weighted above body text. The index is written in the same
  transaction as the document, so it cannot drift from the corpus.
- **graph** — the investigation's own edges: what a document cites, what
  cites it, what else is attached to the same claim or entity. This is the
  part a keyword index cannot do, and every result says why it surfaced.
- **vectors** — off by default. They wait for a retrieval problem lexical
  search cannot solve; what is in the tree is the seam, not the
  architecture. There is no vector database: vectors live in the same SQLite
  file and are compared against the shortlist the other retrievers produced.

```
$ research find investigation:1 "nuclear construction cost overrun"
evidence:36  [lexical]       Continuous Improvement and Cost Overrun in Construction…
evidence:43  [graph+lexical] Nuclear power plant construction costs, 1970-2020
             why: cited by evidence:9
```

RRF uses only each retriever's *ordering*: BM25 scores, graph weights and
cosine similarities are not comparable, and normalising them against each
other would invent a relationship that is not there. Results respect
independence like everything else — syndicated copies fold into the result
for the document they copy, and are named there.

**Reading what you already have.** `research ingest <investigation> <paths…>`
takes local files and folders in as evidence: PDF, plain text, Markdown, HTML
and JSON. They go through the same pipeline as anything retrieved — the same
deduplication, the same provenance, the same untrusted-content envelope — so
a paper saved on disk and the same paper found through a provider are one
source, not two.

```
$ research ingest investigation:1 ~/reading --type paper --family academic
+ evidence:14  Construction cost overruns in small modular reactor programmes
               /home/me/reading/overruns.pdf
```

PDFs are read without a dependency, and `pip install research[pdf]` adds
pypdf for the font encodings the built-in reader cannot map. Which matters,
because the failure mode is the dangerous one: a CID-encoded PDF read without
its encoding yields characters in roughly the right quantity and entirely the
wrong identity, and that noise would be stored as evidence, indexed and
quoted. So extraction is gated on legibility — text that does not read as
prose is refused with a warning naming the remedy, rather than stored as a
document whose contents are wrong. Where characters are dropped, the count
travels with the document. A file's modification time is not treated as a
publication date.

**Provenance.** Every document records the provider, the endpoint, the search
query or fetch that produced it, the document it was reached from, and when.
Every search and every fetch — including the failures — is a row in the
store. `research show <id>` prints the chain.

**Budgets.** Depth, tasks, searches, documents (globally and per family),
citation depth and documents, provider calls, model calls, tokens, failed
source calls and runtime. Counters are durable, so a resumed investigation
continues spending the allowance it started with rather than a fresh one.
Research should terminate because it decided to, not because a context
window filled up.

## Security model

External content is hostile data. The controls are in code, not in prompts:

- The research loop has no shell. There is no command that evaluates or
  executes anything; if sandboxed data analysis is added later it will be a
  separate capability with explicit inputs and outputs.
- One component makes outbound requests. It allows only `http`/`https`,
  refuses loopback, private, link-local and reserved addresses (re-checking
  after every redirect), caps response bodies before reading them,
  allowlists content types rather than sniffing them, bounds retries and
  paces requests per host.
- Retrieved text reaching a model prompt is wrapped as labelled external
  evidence with envelope and role markers defanged, truncated to a stated
  budget, and tagged with the document id so any assertion made from it can
  be traced back.
- Credentials come from a narrow allowlist of environment variables and are
  handed to the single adapter that needs them. Nothing else reads the host
  environment; a serialised config never contains a secret.
- HTML is parsed for text and metadata only. Scripts are discarded, not
  interpreted; the one exception is `application/ld+json`, read as data.
- Filesystem access is narrow, explicit and outside the research loop. No
  action opens a path: ingestion is something a person runs, naming the
  files, and a worker only ever sees the resulting documents. A run reads
  nothing outside the paths it was given — a symlink leaving them is skipped
  rather than followed — skips hidden files, caps file size, and allowlists
  types by content rather than by name.
- A PDF is read, never run. It can carry JavaScript, embedded files and
  launch actions; the reader takes bytes out of content streams and ignores
  every other structure in the file.

## Configuration

Optional. Without a config file the keyless sources are used and the default
budget applies.

```yaml
# research.yaml
model:
  # 'openai' means any OpenAI-compatible endpoint, including a local server.
  provider: anthropic
  model: claude-sonnet-5
  max_steps_per_task: 8

research:
  budget:
    max_depth: 4
    max_tasks: 30
    max_searches: 60
    max_documents: 250
    max_academic_documents: 100
    max_news_documents: 100
    max_citation_depth: 2
    max_runtime_minutes: 30

acquisition:
  request_timeout_seconds: 20
  max_response_bytes: 5000000

retrieval:
  # Lexical and graph retrieval are always on. Vectors are opt-in, and only
  # worth their cost once lexical retrieval is demonstrably failing.
  embeddings_enabled: false
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1

providers:
  arxiv:
    enabled: true
  brave:
    enabled: true
```

```bash
export RESEARCH_ANTHROPIC_API_KEY=...           # or RESEARCH_OPENAI_API_KEY
export RESEARCH_CONTACT_EMAIL=you@example.org   # polite pool at OpenAlex/Crossref
export RESEARCH_BRAVE_API_KEY=...               # optional web search
export RESEARCH_SEMANTIC_SCHOLAR_API_KEY=...    # optional, raises rate limits
.venv/bin/research --config research.yaml sources
```

## Tests

```bash
.venv/bin/python -m pytest          # 641 tests, no network, ~21s
```

Providers and models alike are exercised through recorded payloads served by
a mock transport, and the autonomous path runs against a rule-based model
that needs no credential. The suite covers normalisation, URL and DOI
handling, deduplication and syndication, claim assessment and excerpt
verification, entity resolution and its refusals, evidence-backed timelines,
citation traversal and its termination, the action vocabulary's role
restrictions, the worker loop and its failure modes, planner validation,
bounded recursion, every stopping rule, report traceability, budget
enforcement, provider outage and partial failure, SSRF and prompt-injection
defences, resuming an investigation, vault export and its escaping, corpus
retrieval and fusion, PDF extraction and its legibility gate, local
ingestion and the paths it refuses, and the CLI end to end.

PDFs are generated in the suite rather than checked in, which keeps them
deterministic and lets a test ask for the awkward cases on purpose:
compressed or not, real text or unmappable two-byte font codes. The
optional pypdf dependency is exercised both ways - the built-in reader is
tested with the import forced to fail, so the suite does not quietly stop
covering it once pypdf is installed.

## Status

| Milestone | State |
| --- | --- |
| 1. Research state and evidence foundation | done |
| 2. Academic vertical slice | done |
| 3. Web/news vertical slice | done |
| 4. Claims and provenance graph | done |
| 5. Planner and specialised workers | done |
| 6. Recursive follow-up | done |
| 7. Synthesis | done |
| 8. Investigation UI | done |
| 9. Public social sources | Bluesky, Mastodon, YouTube; Reddit deliberately omitted |
| Obsidian export | done |
| Corpus retrieval (lexical + graph, vectors optional) | done |
| Local document ingestion, PDF extraction | done |

Reddit is deliberately absent from the social sources: its API requires
registered OAuth credentials and its terms restrict what may be stored and
redistributed, so it is not something to enable by default. The interface is
ready for it where an operator has the standing to use it.

What would come next, in order: a sandboxed data-analysis capability with
explicit inputs and outputs (the one place the non-goals leave room for
execution), transcript evidence for video, OCR for scanned PDFs (the one
case ingestion currently refuses rather than guesses at), and per-provider
adaptive pacing rather than one global rate limit.
