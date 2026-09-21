# What comparable tools do, and what this one is missing

Surveyed 2026-09-21 by reading the repositories and docs directly: gpt-researcher,
STORM/Co-STORM, PaperQA2, langchain-ai/open_deep_research, langchain-ai/local-deep-researcher,
dzhng/deep-research, HuggingFace open-deep-research, jina-ai/node-DeepResearch,
zilliztech/deep-searcher, SciPhi-AI/R2R, infiniflow/ragflow, SurfSense.

A few things below are marked *unconfirmed* — the doc site 404'd, or a claim came from
a search summary rather than the source. Those are flagged in place rather than smoothed
over, and should be checked before anything is built on them.

## Where this system is already ahead

Worth stating first, because it decides what is worth copying and what is not.

**Nothing in the survey verifies a quotation.** Not one tool checks that a quoted span
actually appears in the source it is attributed to. The closest anyone comes:

- R2R extracts citation *spans* (`CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9]{7,8})\]")`,
  `extract_citation_spans`, `CitationTracker`) — it confirms a citation id is real and
  locates it in the output, not that the sentence is supported.
- PaperQA2 records which context ids literally appear in the raw answer
  (`used_contexts` in `src/paperqa/types.py`) — a substring check.
- gpt-researcher runs a hallucination judge (`HaluEvalDocumentSummaryNonFactual`)
  *in `evals/`*, post hoc, not in the pipeline.

`excerpt_appears_in` refuses the write. That is a stronger guarantee than anything
surveyed, and it is enforced in code rather than in a prompt.

**Everyone stops on a budget, not on sufficiency.** gpt-researcher (depth 2 × breadth 3),
local-deep-researcher (3 loops), open_deep_research (6 supervisor iterations × 10 tool
calls), deep-searcher (`max_iter`), HF open-deep-research (12/20 steps), PaperQA2
(`agent.timeout: 500.0`). The only sufficiency test found anywhere is
node-DeepResearch's gate — "Is answer definitive? → Has references?" — before it will
terminate. Our `StopReason` set (`evidence_sufficient`, `diminishing_returns`,
`sources_unavailable`, budget, no-open-tasks) is the more developed version of the idea,
and it is recorded rather than implicit.

**Nobody models source independence.** No tool in the survey detects syndication, wire
credit, or derived articles. Ten outlets carrying one Reuters story are ten sources
everywhere else. This is the single most distinctive thing here and nobody is competing
on it.

**Persistent, inspectable state.** PaperQA2 (tantivy indexes, `sync_with_paper_directory`),
R2R, RAGFlow, deep-searcher and SurfSense persist. The entire web-agent tier —
gpt-researcher, open_deep_research, local-deep-researcher, dzhng, HF, jina — has **no
persistence between runs at all**; every run re-fetches. Our claim graph, provenance
chain and activity log have no equivalent in the survey.

PaperQA2 is the only tool with comparable evidence discipline, and it is worth reading
closely: scholarly metadata from Semantic Scholar, Crossref, Unpaywall and OpenAlex
including **citation counts with a retraction check**, and a `contracrow` preset for
contradiction detection.

## Gaps, ranked by what they would actually buy

### 1. No evaluation harness at all — and so no way to know if a change helped

Five of the surveyed tools ship one, and it is the difference between engineering and
decoration:

| Tool | Benchmark | Published result |
| --- | --- | --- |
| open_deep_research | Deep Research Bench (RACE, LLM-judge) | 0.4344, #6, **$87.83 / 207M tokens** |
| HF open-deep-research | GAIA validation | 55% pass@1 (vs 67% for OpenAI's) |
| PaperQA2 | LitQA2, held-out splits committed | claims superhuman on QA |
| gpt-researcher | SimpleQA + a hallucination judge | in-repo runner |
| deep-searcher | Recall@K on 2WikiMultiHopQA | **recall vs iteration and token-cost vs iteration curves** |

674 unit tests say the code does what it was written to do. Not one of them says the
research is any good. Every design decision in this repo — RRF weights, the stopping
rules, `EMBED_PER_SEARCH`, the graph expansion weights — is currently justified by
argument alone.

deep-searcher's curves are the model to copy: they publish marginal recall against
iteration count and token cost against iteration count, which is *direct evidence for
choosing a loop budget*. We have budgets and stopping rules set by judgement. This
would replace judgement with measurement.

Benchmarks across tools are completely non-overlapping — no two report a comparable
number — so the choice is ours. LitQA2 fits the academic path; 2WikiMultiHopQA Recall@K
is cheap and mostly offline; a claim-verification set would test what is actually
distinctive here.

### 2. Evidence is read as "the first 4000 characters", not as the relevant passage

`_do_get_evidence` returns `as_external_evidence(document, limit=characters)` — the head
of the document, whatever the question was. A 29-page PDF returns its opening regardless.

Everyone with a retrieval layer does better:

- **PaperQA2's RCS** is the most interesting design in the survey: chunk, embed, take
  `evidence_k: 10`, then "create scored summary of each chunk **in the context of the
  current query**", then "use LLM to re-score and select most relevant summaries", down
  to `answer_max_sources: 5`. The retrieval unit is an LLM-written, query-conditioned
  summary of a chunk rather than the chunk.
- gpt-researcher filters scraped text with an `EmbeddingsFilter` at `SIMILARITY_THRESHOLD:
  0.42` before it reaches the writer.
- deep-searcher reranks chunks with a YES/NO LLM prompt.
- R2R and RAGFlow do hybrid retrieval with fusion at the chunk level.

We already have the machinery — FTS5, vectors, RRF — but it operates on whole documents
and only for `search_corpus`. Passage-level retrieval *inside* a document, so
`get_evidence` returns what bears on the claim, is the highest-value retrieval change
available and reuses what is built.

Worth noting: **no tool in the survey defaults to a named cross-encoder** (bge-reranker,
Cohere Rerank). The field reranks with LLM calls or with fusion. Our RRF is in line with
the better half.

### 3. The scheduler is strictly sequential

`research/orchestration/scheduler.py` runs one task at a time — one `await worker.run(task)`
in a loop. Five surveyed tools fan out with an explicit concurrency knob:
`DEEP_RESEARCH_CONCURRENCY: 4` and `MAX_SCRAPER_WORKERS: 15` (gpt-researcher),
`max_concurrent_research_units: 5` (open_deep_research), `max_concurrent_requests: 4`
(PaperQA2), tier-matched concurrency (dzhng).

Our tasks are independent by construction and the budget ledger is already durable and
centralised, so this is mostly a scheduler change. It is the difference between a
five-minute run and a twenty-minute one.

### 4. One model does every job

`ModelSettings` has a single `provider`/`model`. The four most mature tools all split:

- gpt-researcher: `FAST_LLM` / `SMART_LLM` / `STRATEGIC_LLM`, separate token limits.
- STORM: `conv_simulator_lm`, `question_asker_lm`, `outline_gen_lm`, `article_gen_lm`,
  `article_polish_lm` (Co-STORM adds six more).
- open_deep_research: summarization `gpt-4.1-mini`, research `gpt-4.1`, **compression**
  `gpt-4.1`, final report `gpt-4.1`.
- PaperQA2: `llm`, `summary_llm`, `agent_llm`, `enrichment_llm`.

We have roles already — planner, scout, academic, skeptic, synthesizer — so this is
config plus a lookup, and it is where the cost savings are.

### 5. No clarification before a run starts

`research investigate` goes straight from question to plan. Three tools ask first:
dzhng's "Smart Follow-up", open_deep_research's `allow_clarification: True`, and its
legacy plan-approval flow. Co-STORM goes further and is the only tool in the survey with
genuine **mid-research** steering — `step(user_utterance=...)` into a live discourse with
a Moderator agent that "generates thought-provoking questions inspired by information
discovered by the retriever but not directly used in previous turns".

A research question is usually underspecified. Asking two questions up front is cheap
and changes what gets planned.

### 6. Cost is counted in tokens, not money

`Resource.TOKENS` and `Resource.MODEL_CALLS` exist; there is no currency. gpt-researcher
tracks per-run cost including embeddings, PaperQA2 gets it via LiteLLM and ships
`tier1_limits`…`tier5_limits` presets matching OpenAI rate tiers, and open_deep_research
publishes cost per benchmark run. Pricing is a table and a multiply; the counters exist.

### 7. MCP

gpt-researcher (`MCP_STRATEGY` of `fast`/`deep`/`disabled`, hybrid `RETRIEVER="tavily,mcp"`,
and a GPTR MCP server), open_deep_research (`mcp_config`, "full MCP compatibility"),
RAGFlow and SurfSense all support it. Not found in PaperQA2, STORM, local-deep-researcher,
dzhng, deep-searcher or node-DeepResearch; unconfirmed for R2R.

Our `ResearchSource` protocol is the right shape for it — an MCP server would be another
adapter behind `search`/`fetch`/`capabilities`. Worth doing when a specific server is
wanted, not before.

### 8. Export beyond Markdown

gpt-researcher does PDF and Word. SurfSense — which pivoted in 2026 to a local-first
desktop app explicitly against NotebookLM — does pptx, docx, xlsx, self-contained HTML,
typeset PDF, flashcards, quizzes, mind maps and an offline podcast via Kokoro-82M.
Everyone else emits Markdown. We have Markdown and an Obsidian vault, which is better
than most. Low priority.

## What nobody has

**Scheduled or recurring research with change detection.** No repo in the survey
documents a scheduler, a watch mode, or re-running a question and diffing against the
prior answer. For most of them this is structural: the entire web-agent tier keeps no
state between runs, so there is nothing to diff against.

We keep the investigation, the claims, the evidence and the provenance. "Re-run this
question and tell me what changed — which claims gained support, which were contradicted,
what is newly retracted" is a feature the architecture already supports and the field has
not built.

**A result cache keyed on query or URL.** Nobody has one. PaperQA2's document index is the
closest analogue and it caches documents, not searches.

## What I would do first

1. **An evaluation harness**, because everything else is unmeasurable without it, and
   because the questions it would answer — how many loops, how much budget, do vectors
   earn their place — are ones we currently answer by argument.
2. **Passage-level evidence retrieval**, because reading the first 4000 characters of a
   filing is the weakest link in an otherwise careful evidence chain, and the retrieval
   machinery is already there.
3. **Parallel task execution**, because it is mostly mechanical and the runs are slow.
4. **Per-role models**, because it is cheap in both senses.

Then, as a differentiator rather than catching up: **re-run and diff**, which nothing in
this survey can do.
