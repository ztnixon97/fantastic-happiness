# Architecture

## Layering

```
research/
  models/         domain objects; no I/O, no provider knowledge
  normalize/      deterministic: URLs, DOIs, text, fingerprints, HTML, document assembly
  storage/        SQLite schema, repositories, one store facade
  sources/        provider adapters behind one interface + the registry
  acquisition/    deduplicate -> charge budget -> persist; untrusted-content handling
  graph/          read-only views over stored citations
  orchestration/  budgets (planner and stopping rules land here)
  operations/     research verbs composed from the layers above
  cli/            inspection and execution surface
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

## Deliberate omissions

- **No vector search.** Exact identity, fingerprints and provider relevance
  have not yet failed. Embeddings arrive when there is a demonstrated
  retrieval problem that needs them, not before.
- **No credibility scores.** Claim status is an enum plus prose: "supported
  by one preprint and contradicted by two later studies" says something a
  number does not.
- **Synchronous storage.** The store is local and fast relative to network
  acquisition; an async driver would buy nothing today. The repository
  interface is the seam if that changes.
- **No UI.** Per the plan, not until the CLI pipeline is good.
