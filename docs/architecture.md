# Architecture

## Layering

```
research/
  models/         domain objects; no I/O, no provider knowledge
  normalize/      deterministic: URLs, DOIs, text, fingerprints, HTML, document assembly
  budgets.py      durable resource counters; everything charges against them
  llm/            provider-agnostic model access
  storage/        SQLite schema, repositories, one store facade
  sources/        provider adapters behind one interface + the registry
  graph/          read-only views: citations, claims, entities, timelines, independence
  acquisition/    deduplicate -> charge budget -> persist; untrusted-content handling;
                  local ingestion
  retrieval/      searching held evidence: lexical, graph, optional vectors, fusion
  operations/     research verbs composed from the layers above
  agents/         the action vocabulary, the worker loop, the planner
  orchestration/  the scheduler and the stopping rules
  synthesis/      the report, assembled from stored state
  cli/            inspection and execution surface
  ui/             read-only projection of stored state for a browser
  export/         projections into other tools (Obsidian vaults)
```

Dependencies point downward only. `models` imports nothing from the package
except itself; `normalize` imports `models`; `sources` may use `normalize`
and `models` but never `storage`; `acquisition` and `operations` are where
sources and storage meet. Nothing below `cli` knows a CLI exists.

These rules are enforced by `tests/unit/test_architecture.py`, which walks
every module's imports rather than trusting this page to stay true. Document
assembly (`normalize/document.py`) sits in the deterministic layer precisely
because source adapters need it and must not depend on the layer that
persists what they return.

The rule that keeps this honest: **a source adapter cannot persist anything.**
It translates a provider response into `SearchHit`s and `EvidenceDocument`s
with an empty `id`, and hands them back. Identity, deduplication, budget and
provenance belong to the acquisition layer, which is the only writer of the
document table.

## Why the model does not hold the state

An investigation is a graph of documents, claims, entities, events and
citations that outgrows a context window quickly, and that must survive a
process restart. So the store is authoritative and the model is a consumer of
it:

- Child tasks persist evidence and return a small structured result
  (`TaskResult`: summary, claim ids, evidence ids, open questions,
  recommended follow-ups). They never hand retrieved text up to a parent.
- Anything from outside that reaches a prompt goes through
  `acquisition.untrusted.as_external_evidence`, which labels it, defangs
  envelope and role markers, truncates it, and stamps it with the document id
  so the resulting assertion can be traced back.
- Resuming means opening the same SQLite file. There is no conversation to
  replay.

## Data model

Tables, and what each exists to answer:

| Table | Question it answers |
| --- | --- |
| `investigations` | what was asked, under what budget, and why it stopped |
| `research_tasks` | how the question was decomposed, and what each part returned |
| `documents` | what material is held, normalised |
| `document_identities` | is this material already held? (indexed exact keys) |
| `document_fingerprints` | is this material *nearly* the same as something held? |
| `claims`, `claim_evidence`, `claim_links` | what is asserted, and what bears on it |
| `entities`, `entity_identifiers`, `entity_aliases` | who is involved, resolved deterministically where possible |
| `events`, `event_entities`, `event_evidence` | what happened, when, on what evidence |
| `citations` | which documents cite which — including edges not yet resolved |
| `relationships` | every other graph edge, each with its own provenance |
| `search_queries`, `source_fetches` | what was asked of providers, and what came back, failures included |
| `budget_usage` | what has been spent, durably |
| `documents_fts` | which held documents use these words (FTS5, schema v2) |
| `document_embeddings` | vectors, when they are enabled at all (schema v2) |

Identifiers are readable and sequential (`evidence:412`, `claim:17`),
allocated by the store. A person reading a report can type one back into the
CLI. `research.ids.parse_id` rejects anything malformed, so an identifier
arriving from model output cannot be mistaken for a valid reference.

Two implementation notes worth knowing:

- SimHash values are stored as hex text. SQLite's `INTEGER` is signed 64-bit
  and the fingerprint is unsigned.
- Citation edges store `''` rather than `NULL` for absent DOIs and external
  ids, because SQLite treats `NULL`s as distinct in a `UNIQUE` constraint,
  which would defeat edge deduplication.
- The schema carries a version. Opening an older database migrates it and
  backfills what the new tables need - schema v2 adds the full-text index
  and backfills it from `documents`, so retrieval works on investigations
  that were gathered before it existed.

## Evidence and provenance

Every `EvidenceDocument` carries a `Provenance`: provider, retrieval method,
endpoint, the query or fetch id that produced it, the document it was reached
from, and the time. The chain a report needs is therefore always available:

```
statement -> claim -> claim_evidence -> document -> provenance -> query/fetch -> provider
```

`source_type` places a document relative to the underlying record — a filing
is nearer than an article about the filing — via `PRIMARY_SOURCE_DISTANCE`.
It is an ordering heuristic for deciding what to chase, not a credibility
score: a bad filing outranks a good article on that axis, which is exactly
why the axis is not called quality.

News results start as `secondary_news_reporting`. "Original reporting" is a
finding, not a default, and belongs to the news researcher once it has been
established.

## Deduplication and independence

Two questions, deliberately separate: *have I got this artifact?* and *is
this a new source?*

1. **Exact identity**, in order of authority. DOI, arXiv id, PubMed id and
   provider id mean the same record — the held document is enriched from the
   new copy (never overwritten) and the second provider is credited in
   `also_provided_by`. Canonical URL and content hash mean the same artifact;
   if the hash matches across different domains it is a syndicated copy
   rather than a duplicate.
2. **Near duplication.** Candidates come from SimHash bands and matching
   title keys, then are scored by the maximum of Jaccard and containment in
   both directions. Containment matters because a copy that adds a house
   sidebar, or drops the closing paragraphs, is still a copy. Above the
   threshold: same domain → near duplicate, different domain → syndicated
   copy.
3. **Shared headline plus shared wire credit.** Thin documents (abstracts,
   snippets) cannot be judged on body overlap; an identical headline with the
   same wire service credit is the stronger signal there.
4. **Attribution links.** An article that links a held document on another
   domain is written from it, however thoroughly it was paraphrased. Text
   comparison loses to a competent rewrite; an attribution link does not.

Everything that is not independent inherits the original's
`independence_key`, and `independent_documents()` groups by it. The copies
are kept — that a wire story reached nine outlets is a fact about its spread
— but anything that counts sources counts groups.

Where the evidence is ambiguous the answer is "not independent", because the
failure that matters is counting one source several times.

## The autonomous loop

```
question
   |
   v
Planner ── validates and stores 3-6 tasks (unknown roles and empty objectives dropped)
   |
   v
Scheduler ── takes the highest-priority pending task, checks the stopping rules
   |
   v
Worker (one role) ── loop, at most N steps:
   |    model emits one JSON action -> runtime validates it -> operation runs
   |    -> compact observation (identifiers, counts, one-line briefs)
   |
   +-> spawn_research_task  ── a child task, depth + 1
   +-> complete_research_task ── summary, claim ids, evidence ids, follow-ups
   |
   v
Scheduler ── turns follow-ups into tasks, re-checks the stopping rules, repeats
   |
   v
Synthesizer ── writes the report from stored state; introduces no facts
```

The worker loop is a JSON action protocol rather than any vendor's
function-calling format. That is deliberate: the same agent code runs against
a model with no tool-calling support, including a local one, and the parser
tolerates the fences and preamble models wrap JSON in.

### What a worker can and cannot do

`agents/actions.py` holds the closed vocabulary: sixteen research-native
actions, each declaring its parameters and the roles allowed to use it. A
news worker cannot walk a citation graph; a skeptic cannot plan. There is no
action that executes anything, and `tests/unit/test_agents.py` asserts it.

`agents/runtime.py` is the boundary. Every action passes through it, and it
validates identifier syntax before use, refuses actions outside the role,
executes through the ordinary operations the CLI also calls, and returns a
*compact* observation — identifiers, counts, one-line briefs. Search results
never arrive as text, so a page cannot address the model simply by being
found. A worker reads a document only by asking for it, and gets it back
wrapped as labelled external evidence.

### Bounding recursion

Three limits, because any one alone fails:

* **Depth** — `spawn_research_task` refuses beyond `max_depth`.
* **Task budget** — every task, planned or spawned, charges `Resource.TASKS`.
* **Duplicate objectives** — the planner refuses an objective that matches one
  already planned. Without this, a skeptic recommends a skeptic and the run
  fills its budget with the same question asked five ways.

### Stopping

`orchestration/stopping.py` evaluates rules in order of authority: budget
exhausted, runtime exhausted, no open tasks, diminishing returns, evidence
sufficient. Two independent diminishing-returns signals are checked
separately, because they occur apart: a corpus that has become mostly copies,
and a run of searches that returned nothing. Whichever fires is written to
`investigations.stop_reason` with its detail, so the question "why did it stop
there?" has an answer months later.

## Synthesis

`synthesis/report.py` assembles the report from the record: claims with their
assessments, one entry per independent source with its copies named, the
timeline, the open questions, the stopping reason. `synthesis/synthesizer.py`
asks a model for the executive summary only — and checks it, rejecting a
summary that cites claim or evidence identifiers the investigation does not
hold. A report whose prose cannot be traced is worse than a report with a
dull summary, so the dull summary is the fallback.

The synthesizer is given the record, not the corpus, and has no search
actions. A synthesis step that quietly gathers more evidence produces a
report nobody can trace.

## Claims: status without a score

`graph/claims.py` derives a claim's status from what is attached to it, by
rules stated in code and checkable by a reader:

| Situation | Status |
| --- | --- |
| nothing linked | `unverified` |
| standing support and standing contradiction | `mixed` |
| standing contradiction only | `contradicted` |
| every linked item retracted | `insufficient_evidence` |
| support is one non-independent cluster of secondary or self-reported material | `insufficient_evidence` |
| otherwise, support only | `supported` |

"Standing" means not retracted: a withdrawn paper is neither support nor
rebuttal. Alongside the status comes a sentence — "supported by 1 independent
primary source, 2025", "supported by 1 independent source (2 documents, 1 of
them copies or rewrites)" — and a list of gaps that names what would improve
the assessment. That list is what a follow-up planner reads: not "what is
unknown" in the abstract, but which propositions are thin and in what way.

`possibly_superseded` is set when the newest contradicting document postdates
every supporting one. It is the cheapest available check for a finding that
has been overtaken.

### Two integrity rules

`operations/claims.py` enforces both in code, because both are cases where a
model's output could otherwise become indistinguishable from retrieved
material:

* An excerpt must appear in the document it is attributed to. Comparison
  folds quotation marks, dashes and whitespace — those differences do not make
  a quotation inauthentic — but wording differences do, and a mismatch is a
  refusal, not a warning.
* Reasoning goes in `analysis`, never in `excerpt`. The CLI renders the two
  under separate headings for the same reason.

## Entity resolution

Deterministic identifiers decide identity. `EntityRegistry.resolve` tries, in
order: an authoritative identifier match (ORCID, OpenAlex, ROR, Wikidata, SEC
CIK, DOI, arXiv, PubMed); then a name or alias match among candidates that do
not *conflict* on an authoritative identifier; then creation.

Three refusals matter more than the matches:

* Two different ORCIDs are two people, whatever the names say.
* Several plausible candidates and nothing authoritative to choose between
  them resolves to `AMBIGUOUS` with the candidates attached — the caller
  decides, nothing is merged.
* A match made on a weak name ("J. Smith") is recorded on the entity as
  `review_needed`, so conflation is visible rather than silent.

`register_document` registers only what a provider's structured fields say:
authors and the publishing venue. An entity found by *reading* the text is a
model judgement, and belongs to a research worker rather than to deterministic
code.

## Timelines

An event is an assertion, so it carries the same burden as a claim: at least
one evidence document, and an explicit `DatePrecision` whenever it is dated.
`operations/timeline.py` refuses an undocumented event, a dated event with
unknown precision, an undated event claiming precision, and an event that ends
before it starts.

Timelines mix two entry kinds. `event` entries are asserted and evidence-backed.
`publication` entries are derived from the corpus — one per independence group,
dated by the earliest copy — so a wire story carried by nine outlets is one
point on the chronology, not nine. `first_appearance` answers "when did this
story actually break?" by following the independence group rather than the
document in hand.

## Source interface

```python
class ResearchSource(Protocol):
    name: str
    async def search(self, query: ResearchQuery) -> list[SearchHit]: ...
    async def fetch(self, hit: SearchHit) -> EvidenceDocument: ...
    def capabilities(self) -> SourceCapabilities: ...
```

`SourceCapabilities` is how routing stays honest: it declares the families
served, whether search results already carry usable content, whether the
provider exposes references or citing works, and whether it needs a key. The
registry uses it to skip providers that cannot serve a request and to prefer
keyless ones, so an investigation degrades to fewer sources instead of
failing when a key is absent.

Academic hits are promoted to documents without a second request — the search
response already contains the record. Web and news hits are pointers: their
snippets are written by the search engine, not the publisher, so a result
whose body cannot be fetched is recorded as unretrieved rather than stored as
evidence.

### Adding a source

1. Subclass `SourceAdapter` (or `AcademicAdapter` for literature providers).
2. Implement `search`, `capabilities`, and `fetch` if the provider returns
   content; add `get_references` / `get_citing_papers` if it has a citation
   graph, and `lookup_doi` if it can resolve one.
3. Translate into `SearchHit` / `EvidenceDocument` only. No storage, no
   logging, no budget: `operations` does that for every provider uniformly.
4. Register it in `sources/registry.py` (add its credential env var to
   `config.PROVIDER_KEY_ENV` if it needs one).
5. Record a payload under `tests/fixtures/` and test the translation.

`sources/static.py` and `sources/offline.py` implement the same interface over
a fixed corpus, which is how the demo and the tests run without a network
while still exercising the real fetcher, extractor and deduplication rules.

## Budgets and stopping

`BudgetLedger` enforces limits against durable counters. Limits are checked
before a counter moves, so a stored count is always a count of work actually
permitted, and a document refused by a per-family limit does not consume the
global one.

`exhausted_resources()` and `near_exhaustion()` are the signals a planner
will use alongside the other stopping criteria — claims adequately evidenced,
secondary claims traced to primaries, counterevidence no longer changing the
picture, searches returning only material already held, citation traversal
reaching diminishing relevance. `investigations.stop_reason` records which
one ended the run.

## What live running changed

Mocked providers answer the shape you recorded. Real ones answer what they
feel like, and testing against them changed the code in ways no fixture would
have prompted:

* **Crossref's `select` is route-specific.** Asking for one field that is not
  selectable on `/works` fails the entire request with HTTP 400. `language`
  is present in full records but is not selectable there, so every Crossref
  search was returning nothing. `SELECT_FIELDS` is now a named constant with
  that fact written next to it.
* **arXiv needs its terms combined deliberately.** A quoted phrase matches
  nothing. A bag of words matches on *any* term and ranks by citation, so
  "small modular nuclear reactors economically competitive" returns neutrino
  physics. ANDing every term of a long query returns nothing. The adapter
  ANDs the leading four terms - the ones that carry the subject - and exposes
  a `phrase` filter for callers that want an exact phrase.
* **`Retry-After` can mean tomorrow.** OpenAlex answers a rate-limited call
  asking to be retried in about 21 hours. The client capped the wait at 30s
  and retried twice, so one refusing provider cost 61 seconds of every
  search. A `Retry-After` beyond a few seconds is now treated as
  unavailability rather than as a delay to honour.
* **Real pages have `<title>` inside `<svg>`.** Federal sites label their
  banner padlock with `<title>Lock</title>`, which the extractor was reading
  as the document title. A `<title>` inside a skipped element is now ignored.
* **A refusing provider is expensive.** A rate-limited provider costs the
  full retry budget on every search. Three consecutive failures now retire a
  provider for the rest of the investigation, counted from the query log so
  the decision survives a restart and is visible in `research activity`.
* **"Nothing found" and "nothing looked" are different answers.** With no
  keyed web provider configured, web search has no provider at all. That is
  now recorded as `no_provider` and reported to the worker as a failure with
  a suggestion, rather than as an empty result it would read as evidence of
  absence.
* **A failure to look is not a finding.** An investigation whose searches
  could not run was stopping with "diminishing returns", which tells a reader
  the topic was exhausted. Unreachable providers now stop the run as
  `sources_unavailable`, and the diminishing-returns rule counts only
  searches that actually ran.
* **Truncating an observation breaks it.** Observations were being cut to a
  character budget, which turns JSON into something a worker cannot parse -
  and the worker then behaves as though the search found nothing. Lists are
  now shortened structurally, with a count of what was withheld.

The first three were single-line bugs that every mocked test passed.

## The UI

`research ui` serves a read-only projection of stored state. It is outside
the research core in the strongest sense: nothing below it imports it, and
removing the package changes nothing about how research runs.

The safety properties are structural rather than advisory. Only GET is
answered; the route table is fixed and identifiers are validated against
`prefix:number` before they reach the store; every handler is a read model.
On the client side, retrieved content is inserted with `textContent` only -
there is no `innerHTML`, `insertAdjacentHTML` or `eval` in the file, and a
test asserts their absence - and the page is served under a content policy
that forbids remote script, inline script and inline style. Driving the page
in a real browser found two bugs that reading it did not: a legend built with
inline `style` attributes its own CSP rejects, and a CSS specificity mistake
that left the swatches grey.

The graph view folds copies into the document they copy, for the same reason
the counting does: drawing a syndicated copy as its own node is the visual
form of counting it as its own source.

## Retrieval

The brief's rule was that vector search waits for a demonstrated retrieval
problem. Live running produced one, and it was not the problem embeddings
solve. Every search in the system meant calling a provider, so an
investigation could not answer the cheapest question it has - *do I already
have something about this?* - and a held document could be read back only by
an identifier someone already knew. Workers re-queried providers for material
sitting in SQLite, and paid budget for it.

`research/retrieval/` answers that question with three retrievers and a
fusion step.

**Lexical** is FTS5 with BM25 and a porter stemmer, weighted
`(title 8, abstract 4, body 1)`: a paper whose *title* is about the subject
is more on point than one mentioning it in passing, and BM25 alone does not
know that. The index is maintained by `DocumentRepository`, in the same
transaction as the row, through `storage/fts.py` - a separate module for a
layering reason worth stating, since storage must not import retrieval, and
an earlier version that did produced a circular import. The index can also be
rebuilt from the documents table, which is what makes it safe to treat as a
cache rather than as state.

Queries are prose, not a query language. `prepare_query` strips FTS5 syntax
characters and quotes each term, so a stray quotation mark in a research
question cannot become a syntax error - or a MATCH clause.

**Graph expansion** is the part a keyword index cannot do. From the top
lexical hits it walks the investigation's own edges - what a document cites,
what cites it, what else is attached to the same claim or the same entity,
what copies it - and every hit carries its reason, so a result says *cited by
evidence:9* rather than arriving with an unexplained score. In live running
this is what promoted a paper that never used the query's words into the top
three.

**Vectors** are off by default and are a third opinion when on. There is no
vector database: vectors are float32 blobs in the same SQLite file, and
similarity is computed only over candidates the other retrievers already
surfaced, which keeps the work proportional to the shortlist rather than to
the corpus. A corpus bounded by a document budget does not need an index
server, and adding one would mean a second store that can disagree with the
first.

**Fusion** is reciprocal rank fusion, weighted per retriever. BM25 scores,
graph connection weights and cosine similarities are not comparable, and
normalising them against each other invents a relationship that is not
there; RRF uses only each retriever's ordering, which is the part all three
agree is meaningful. `k = 60` is the value from the original paper and is
left alone until there is evidence for changing it.

Results are collapsed by `independence_key` before they are returned, and the
original is preferred over a copy as the representative - the same rule the
report and the graph already apply, for the same reason. The copies are named
on the result that stands for them, so nothing is hidden.

The whole thing is reachable as `research find`, and as the `search_corpus`
action. Putting it in the vocabulary is the point: a worker that can look
inside first stops paying a provider for what the investigation already
owns.

## Ingestion

`research/acquisition/ingest.py` reads local files and folders in as
evidence. It is in the acquisition layer, not the action vocabulary, and that
placement is the design:

**No action opens a path.** Ingestion is something a person runs, naming the
files; a worker only ever sees the resulting documents, wrapped in the same
untrusted-content envelope as anything fetched. `tests/unit/test_security.py`
asserts this structurally - no action parameter names a path, and the action
runtime's source contains no filesystem call.

**A run reads only what it was given.** The named paths are the roots. A
symlink pointing out of a root is skipped rather than followed, hidden files
and directories are skipped, the walk is depth-bounded, files above a size
limit are refused rather than loaded, and file types are an allowlist decided
by content first and by extension second.

**Ingested material is not privileged.** It goes through the same pipeline as
anything retrieved: deduplicated against held documents, charged to the
document budget, stored with provenance. A paper saved on disk and the same
paper found through a provider are one source. The absolute path is the
document's external id, so re-ingesting a folder merges rather than
duplicating.

**A file's modification time is not a publication date.** It is kept as
metadata. A document dated by its mtime would sit in a timeline and be
weighed for recency as if that date meant something.

### Reading PDFs

Most primary records are published as PDFs, so `normalize/pdf.py` extracts
text from them - and this is where the interesting failure lives.

There are three, and which one runs is decided by what the cheaper ones
produce.

`pypdf` is used when installed (`pip install research[pdf]`), because it
handles the font encodings real documents use. Without it, a built-in reader
decodes Flate-compressed content streams and the text-showing operators,
which covers PDFs produced from text; it does *not* resolve embedded CMaps,
so a document using a CID font comes out as noise. pypdf is also stricter
about file structure than the format is in practice, so a file it refuses
falls back to the simpler reader rather than being given up on.

That noise is the dangerous case, because it looks like text to everything
downstream: it would be stored as evidence, indexed, and quoted. So
extraction is gated on legibility - character classes, space density and the
proportion of word-shaped tokens - and text that does not read as prose is
discarded with a warning naming the remedy, rather than passed on as a
document whose contents are wrong. Where individual characters cannot be
mapped (a subsetted font putting the ligature in "firmly" at `\x02` is the
common case) they are dropped and the count travels with the document:
guessing at the missing letters would put words in a document that does not
contain them, and excerpts are checked against stored text.

Verified against real files rather than only generated ones: a 29-page arXiv
paper and a 28-page SEC form both come out legible through the built-in
reader, the latter with its unmappable characters counted.

### Docling, and what it is allowed to change

`normalize/docling_reader.py` is the seam for Docling, which does layout
analysis and table structure, opens the Office formats, and reads scans with
OCR. It is off unless an operator turns it on, and its absence changes
nothing: every caller asks for it through `converter_for(settings)`, which
returns `None` when it is switched off or not installed, and `None` is how
the rest of the code says "use the readers that need no dependency".

The chain in `extract_pdf` is ordered by cost, and each step's output
decides whether the next one runs:

```
docling (layout, tables)  ─┐
                           ├─ legible?  ── yes ──▶ store
pypdf → built-in reader   ─┘      │
                                  └── no text layer ──▶ docling + OCR
```

The escalation to OCR reuses a gate that already existed. The legibility
check was built to stop noise from a CID-encoded PDF being stored as
evidence; "nothing readable came out" is also exactly the signal that a
document has no text layer, which is what a scan is. So `ocr: auto` spends
tens of seconds per document only where milliseconds have already failed,
rather than on every PDF. `ocr: always` reorders the chain for a corpus known
to be scanned; `ocr: off` leaves it out. An image skips the text-layer
attempt entirely - there is nothing there to try - and is refused rather
than stored empty when OCR is off.

OCR output faces the same gate as everything else. Degraded text is still
text; noise is still noise, whichever reader produced it.

Three properties are deliberate:

*It cannot quietly become required.* The extras are separate
(`research[pdf]`, `research[docling]`), the config flag defaults to off, and
a configuration that asks for Docling on a machine without it says so and
falls back rather than failing the ingest.

*It cannot quietly become the security model.* Enabling it means model
inference - and, with OCR, native image decoders - parsing external
documents in-process. That is a real change in attack surface, stated in the
module's own docstring and in the README's security section, and it is why
this is a decision an operator makes rather than a default. `artifacts_path`
exists so a machine that should not reach out does not have to.

*It cannot quietly invent metadata.* Docling infers a title from layout
rather than reading one off a field, so a document records where its title
came from: `docling layout (title)`, `first heading`, or `file name`. The
same document also records which reader produced its text, so a report can
be traced to the pass that read it.

Tests never run it. It downloads model weights on first use and takes tens of
seconds per document, so the suite exercises the wiring through a stub - when
it is asked, with what, what is done with what it returns, and what happens
when it is absent or fails - and passes identically with it installed and
without it. Verified live against a scanned PDF with no text layer, which
came back with every figure in it correct, and against a .docx, which the
built-in readers cannot open at all.

Fetching PDFs over HTTP is the same extraction behind the content-type
allowlist, which now admits `application/pdf`. A PDF is read, never run: it
can carry JavaScript, embedded files and launch actions, and the reader takes
bytes out of content streams and ignores every other structure in the file.

## Export

`research/export/` is a presentation layer beside `research/ui/`: nothing
below it imports it, and removing it changes nothing about how research runs.

The Obsidian exporter is the interesting one because a vault is a graph
already. Records become notes, relationships become wikilinks, and the
structured record goes into YAML frontmatter where Dataview and search can
reach it. Identifiers are registered as note aliases, so `evidence:12` in a
report resolves to the note rather than reading as text.

Two constraints shape the implementation:

*A vault is a rendering target, not a text dump.* Obsidian renders raw HTML
inside Electron, resolves `[[...]]` into real edges, and lets Dataview and
Templater execute fenced blocks. So `escape_external` neutralises markup,
wikilinks, embeds, fences and comment markers before retrieved content lands
in a note, and frontmatter goes through a YAML serialiser rather than string
formatting - a title containing a line reading `---` would otherwise end the
block and spill its remainder into the body as content.

*The vault belongs to the user.* The exporter writes only inside its own
folder, resolves every path to check that, and refuses to overwrite a file
that does not carry its marker. A note someone has edited is reported and
left alone unless `--force` says otherwise. The canvas carries the marker in
its JSON for the same reason - and if Obsidian drops that key when the reader
edits the canvas, the file stops being ours, which is the right outcome.

## Deliberate omissions

- **Vector search is off, not absent.** It stayed out until lexical retrieval
  had something it could not do; see *Retrieval* above for what changed and
  what was built instead. Enabling it is a configuration flag and a
  credential, and the default remains off.
- **No credibility scores.** Claim status is an enum plus prose: "supported
  by one preprint and contradicted by two later studies" says something a
  number does not.
- **Synchronous storage.** The store is local and fast relative to network
  acquisition; an async driver would buy nothing today. The repository
  interface is the seam if that changes.
- **No UI.** Per the plan, not until the CLI pipeline is good. (Since built:
  `research ui`, read-only.)
- **OCR is off, not absent.** A scanned PDF is refused with a reason
  unless Docling is enabled, because reading one means model inference over
  hostile input. When it is enabled, OCR output is held to the same
  legibility gate as everything else.
