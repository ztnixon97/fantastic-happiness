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
uv sync          # one command: interpreter, dependencies, dev group, venv

# An autonomous investigation: plan, work, recurse, stop. No network, no keys.
uv run research --offline investigate "Are small modular reactors competitive for AI data centres?"
uv run research report investigation:1 --no-summary

uv run research demo                     # the same corpus, scripted rather than planned
uv run research show evidence:9          # where did this come from?
uv run research independence investigation:1
uv run research claim new investigation:1 "SMR costs have risen above projections"
uv run research claim link investigation:1 claim:1 evidence:8 --excerpt "The target price for power"
uv run research claim show claim:1
uv run research questions investigation:1
uv run research timeline investigation:1 --publications
uv run research graph investigation:1
uv run research activity investigation:1
uv run research budget investigation:1

# Search what the investigation already holds - no provider call, no budget.
uv run research find investigation:1 "cost escalation"

# Measure it: retrieval, source independence, and a whole run's record.
uv run research measure

# Ask again later, and be told only what changed.
uv run research investigate --investigation investigation:1
uv run research diff investigation:1

# Read your own files in as evidence: PDF, text, Markdown, HTML, JSON.
uv run research ingest investigation:1 ./papers --type paper --family academic
```

`uv sync` installs from `uv.lock`, so every machine gets the same
resolution. Document conversion and the sentence encoder share a torch
install; `pyproject.toml` pins it to the CPU wheels, which is the difference
between a 1.6GB environment and a 6.2GB one, because nothing here wants a
CUDA runtime. Model weights download on first use, and both features degrade
to the dependency-free readers and a lexical+graph ranking without them.

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
uv run research new "are small modular reactors competitive for AI data centres?"
uv run research search investigation:1 "small modular reactor levelized cost" --family academic
uv run research search investigation:1 "NuScale project cancellation" --family news
uv run research citations investigation:1 evidence:1 --direction both --depth 2
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
how to add a source, and [docs/landscape.md](docs/landscape.md) for what
comparable tools do, what this one does that they do not, and what is
missing.

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

**Asking first.** A run may ask a few clarifying questions before planning —
which alternative a comparison is against, over what horizon — and often asks
none, because a well-posed question needs none. Answers become context for
the planner, never evidence. A piped or scheduled run skips it and records
that it did.

**Autonomy with limits.** Tasks run several at a time, on models chosen per
role: planning and synthesis get the strategic tier, gathering gets the fast
one, skeptics stay on the default because attacking a conclusion is not grunt
work. A planner decomposes the question into tasks;
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
- **vectors** — a local sentence encoder, no credential and no outbound
  request. This is the one that finds a document sharing *no words* with the
  query, so it searches the whole corpus rather than re-ranking what lexical
  found. There is no vector database: vectors are blobs in the same SQLite
  file, and a search tops up the index by a bounded batch, so it maintains
  itself rather than needing an indexing step.

```
$ research find investigation:1 "nuclear construction cost overrun"
evidence:36  [lexical]       Continuous Improvement and Cost Overrun in Construction…
evidence:43  [graph+lexical] Nuclear power plant construction costs, 1970-2020
             why: cited by evidence:9

$ research find investigation:1 "substation headroom for datacentre loads"
evidence:4   [vector] Utility statement          ← not one of those words is
evidence:5   [vector] Construction cost overruns…  in the corpus
$ research find investigation:1 "substation headroom for datacentre loads" --no-embeddings
nothing held matches 'substation headroom for datacentre loads'
```

RRF uses only each retriever's *ordering*: BM25 scores, graph weights and
cosine similarities are not comparable, and normalising them against each
other would invent a relationship that is not there. Results respect
independence like everything else — syndicated copies fold into the result
for the document they copy, and are named there.

A model that will not load costs the ranking its third opinion and nothing
else: the search runs on lexical and graph results and reports that it was
degraded. `embedding_model_path` points at weights already on disk, for a
machine that should not fetch any.

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

**Layout, tables, Office formats and OCR.**
[Docling](https://github.com/docling-project/docling) does the reading, and
the built-in readers sit behind it as the fallback. It does layout analysis
and table structure, it opens Word, PowerPoint, Excel and EPUB, and it reads
scans — which is the difference between a filing this system can use and one
it can only refuse. `--no-docling` and `ingest.docling_enabled: false` fall
back to the readers that need nothing installed.

The escalation is the design. A text layer is read in milliseconds; OCR takes
tens of seconds, so it is not spent on documents that do not need it:

```
                    ┌ docling (layout + tables) ┐
PDF ────────────────┤                           ├── legible? ── store
                    └ pypdf → built-in reader ──┘      │
                                                       │ no text layer
                                  docling + OCR ───────┘
```

The legibility gate above is what decides a document is a scan: an earlier
pass producing nothing readable is precisely the signal that there is nothing
to read without OCR. `--ocr always` puts the OCR pass first for a corpus
known to be scanned; `--ocr off` never runs it. Images skip straight to OCR,
because an image has no text layer to try. Every document records which
reader produced it, and whether its title was read, inferred from layout, or
taken from the file name.

```
$ research ingest investigation:1 ~/filings --docling --ocr auto --type regulatory
+ evidence:1   NOTICE OF CONSTRUCTION COST REVISION
               /home/me/filings/notice.pdf  [docling+ocr]
```

Two consequences are worth knowing rather than discovering. Model weights
download on first use — point `docling_artifacts_path` at a directory you
have pre-populated and it converts without reaching out at all. And
conversion means model inference, with OCR native image decoders too, over
external documents; that is a larger attack surface than a regular
expression over a content stream, and turning it off is a setting.

**Measured, not argued.** `research measure` scores three things offline and
deterministically: retrieval, as twelve hand-judged queries with a
per-retriever ablation; source independence, as twelve hand-judged document
pairs; and a whole run, by the record it left rather than by a judge grading
its prose.

```
$ uv run research measure retrieval
RETRIEVERS     RECALL_AT_5  RECALL_AT_10  MRR  NDCG_AT_10
lexical        0.9306       1.0           1.0  0.9413
lexical+graph  0.9722       1.0           1.0  0.9498
```

It paid for itself on the first run. Graph expansion was making every
ranking *worse* — mean reciprocal rank 1.00 without it, 0.79 with it — and
the cause was not where the first two guesses put it. FTS5 returns every
document containing any term, BM25 separated real matches from incidental
ones by 28×, and reciprocal rank fusion throws those scores away and keeps
only positions: a document matching on one stray word sat at "rank 6", close
enough under RRF that any second signal lifted it over a direct match. A
retriever should return its matches, not its corpus in order. The cutoff that
fixes it was chosen by sweeping it against the query set, not by argument.
`docs/architecture.md` has the detail.

Independence scores precision 1.00, recall 0.60: it never merges two real
sources, and it misses derived articles. That is a number to improve rather
than a claim to make, which is the point of having it.

**Ask again later.** Every run ends with a snapshot of what the claims
amounted to, so the next one can say what is different rather than handing
you a fresh report to re-read. No tool in the survey can do this; most of
them keep nothing between runs, so they would have to build a store first.

```
$ uv run research diff investigation:1
Changes that bear on the conclusions
  claim:1  Peer-reviewed work says SMR costs beat gas
      evidence retracted since the last run: evidence:1
      status supported -> contradicted
      counterevidence 0 -> 1 independent sources
```

The diff is about claims, not documents — "forty-one new documents" is
activity, and the line above is an answer. Changes are sorted by whether
they ought to change your mind: a retraction, a status flip or new
counterevidence lead; a fifth supporting source when you had four does not.
Retraction is the case that justifies the feature, because it happens after
a run is over and nothing about that run will ever notice it.

**Provenance.** Every document records the provider, the endpoint, the search
query or fetch that produced it, the document it was reached from, and when.
Every search and every fetch — including the failures — is a row in the
store. `research show <id>` prints the chain.

**Budgets.** Depth, tasks, searches, documents (globally and per family),
citation depth and documents, provider calls, model calls, tokens, money,
failed source calls and runtime. Cost is counted only for models the config
has priced — there is no built-in price table, because a stale price reported
as this run's cost would be a fabricated figure. Counters are durable, so a resumed investigation
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
- Document conversion runs models over external input, and that is the
  largest attack surface here: layout and table models parse every
  document, and OCR adds native image decoders. It is a real trade for
  being able to read a scanned filing, it is the default, and
  `ingest.docling_enabled: false` is the way back to readers that are not.
  Weights download on first use unless `docling_artifacts_path` points at a
  directory already populated.
- The embedder runs locally by default, so held evidence is not sent
  anywhere to be ranked. Naming a hosted provider instead does send document
  text to it; that is a configuration change, and the credential comes from
  the same allowlist as everything else.

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
  # All three retrievers are on. The default embedder runs in-process, so
  # vectors cost no credential and no outbound request.
  embeddings_enabled: true
  embedding_provider: local          # or a name in the credential allowlist
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  # embedding_model_path: /opt/models/minilm   # weights already on disk

model:
  # Tiers, so the planner need not share the gatherers' model. Omit them and
  # every role uses `model` above.
  fast: gpt-4.1-mini
  strategic:
    model: claude-opus-5
    max_tokens: 8192
  max_concurrent_tasks: 4
  # No built-in price table: a stale price reported as this run's cost would
  # be a fabricated figure. (input, output) per million tokens.
  prices:
    claude-sonnet-5: [3.0, 15.0]

# MCP servers are addresses, never commands: this system will not start one.
mcp_servers:
  - name: house_index
    url: https://mcp.internal.example/mcp
    family: corporate
    credential: house_index   # a name in the allowlist, never a literal key

ingest:
  # Docling reads documents; the built-in readers are the fallback.
  docling_enabled: true
  ocr: auto            # off | auto (only when there is no text layer) | always
  ocr_languages: [en]
  max_pages: 300
  # Pre-downloaded model weights; set it to convert without a network.
  # docling_artifacts_path: /opt/docling-models

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
uv run research --config research.yaml sources
```

## Tests

```bash
uv run pytest             # 810 tests, no network, no models, ~26s
uv run pytest -m models   # 5 more that load the real stack, ~14s
```

CI runs both, in separate jobs: `uv sync --frozen` so a stale lockfile fails
rather than silently resolving something else, then ruff and the fast suite;
and a second job with the model cache warmed that runs the `models` marker.

The default suite runs no models at all — they would need a network, a cache
directory and a lot of CPU, and the suite would stop being deterministic. The
cost of that is the one thing it cannot see: a broken install. A torch and
torchvision wheel mismatch, found while moving to uv, left the embedder
raising on every load while all 674 tests passed. The `models` marker is the
answer — a handful of tests that load Docling and the encoder for real, run
after changing dependencies.

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
ingestion and the paths it refuses, the docling chain and its escalation to
OCR, and the CLI end to end.

PDFs are generated in the suite rather than checked in, which keeps them
deterministic and lets a test ask for the awkward cases on purpose:
compressed or not, real text or unmappable two-byte font codes. The
optional pypdf dependency is exercised both ways - the built-in reader is
tested with the import forced to fail, so the suite does not quietly stop
covering it once pypdf is installed. Docling is never actually run: it
downloads model weights and takes tens of seconds per document, so what the
suite tests is the wiring - when it is asked, what is done with what it
returns, and what happens when it is absent or fails. The same goes for the
sentence encoder. Both are made unavailable for the whole suite, which
exercises exactly the path a machine without them takes, and the suite passes
identically with them installed and without them.

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
| Docling: layout, tables, Office formats, OCR | done, default |
| Vector retrieval, local encoder | done, default |
| Per-role models, cost in money, parallel tasks | done |
| Passage-level evidence, clarifying questions | done |
| Evaluation harness (`research measure`) | done |
| MCP servers as sources | done, HTTP only |
| Re-run and diff | done |

Reddit is deliberately absent from the social sources: its API requires
registered OAuth credentials and its terms restrict what may be stored and
redistributed, so it is not something to enable by default. The interface is
ready for it where an operator has the standing to use it.

What would come next, in order: a sandboxed data-analysis capability with
explicit inputs and outputs (the one place the non-goals leave room for
execution), transcript evidence for video, and per-provider adaptive pacing
rather than one global rate limit.
