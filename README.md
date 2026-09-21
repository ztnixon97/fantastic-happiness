# research

A persistent, evidence-driven research environment.

Not an autonomous agent with research tools bolted on: a store of
investigations, evidence, claims and provenance, with a constrained set of
research operations over it. Models supply semantic judgement. Storage,
identity, deduplication, budgets and provenance are ordinary code.

Milestones 1–4 are implemented: the evidence foundation, the academic and
web/news vertical slices, and the claim/entity/event graph on top of them,
plus a CLI that can gather, normalise, persist and inspect a mixed corpus and
turn it into an argument. The planner, recursive follow-up and synthesis are
designed for but not yet built — see [Status](#status).

## Try it

Everything below runs offline against a bundled corpus. No API keys, no
network.

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'

.venv/bin/research demo                     # a full investigation, offline
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
```

`demo` searches academic, news and web sources, follows a citation graph in
both directions, deduplicates, and leaves an investigation on disk. A sample
of its output:

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
      |
      v
Store  investigations, tasks, documents, claims, entities, events,
       citations, relationships, search_queries, source_fetches, budget_usage
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

## Configuration

Optional. Without a config file the keyless sources are used and the default
budget applies.

```yaml
# research.yaml
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

providers:
  arxiv:
    enabled: true
  brave:
    enabled: true
```

```bash
export RESEARCH_CONTACT_EMAIL=you@example.org   # polite pool at OpenAlex/Crossref
export RESEARCH_BRAVE_API_KEY=...               # optional web search
export RESEARCH_SEMANTIC_SCHOLAR_API_KEY=...    # optional, raises rate limits
.venv/bin/research --config research.yaml sources
```

## Tests

```bash
.venv/bin/python -m pytest          # 335 tests, no network, ~10s
```

Providers are exercised through recorded payloads served by a mock
transport. The suite covers normalisation, URL and DOI handling,
deduplication and syndication, claim assessment and excerpt verification,
entity resolution and its refusals, evidence-backed timelines, citation
traversal and its termination, budget enforcement, provider outage and
partial failure, SSRF and prompt-injection defences, resuming an
investigation, and the CLI end to end.

## Status

| Milestone | State |
| --- | --- |
| 1. Research state and evidence foundation | done |
| 2. Academic vertical slice | done |
| 3. Web/news vertical slice | done |
| 4. Claims and provenance graph | done |
| 5. Planner and specialised workers | roles, operations and task model defined; not wired to a model yet |
| 6. Recursive follow-up | budgets, stopping signals and structured follow-ups in place; scheduler next |
| 7. Synthesis | not started |
| 8. Investigation UI | deliberately not started |
| 9. Public social sources | not started; the source interface is ready for them |

The next step is Milestone 5: a provider-agnostic model interface and the
specialised research roles — planner, scout, academic, news, primary-source
and skeptic — driving the operations that already exist.
