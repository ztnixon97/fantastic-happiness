"""Does retrieval find the right documents?

Runs a hand-judged query set against the bundled corpus and scores the
ranking, once per combination of retrievers. The ablation is the point: it
is what turns "vectors seem to help" into a number, and it is the thing
nothing in this system could answer before.

The corpus is fifteen documents, which is small. The numbers are therefore
about *this* corpus and are useful for comparing configurations against
each other, not for comparing this system against a published leaderboard.
Saying so is better than implying otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from research.evaluation.corpus import EvaluationCorpus, build_corpus, load_dataset
from research.evaluation.metrics import RankingScore, average
from research.retrieval.embeddings import EmbeddingIndex
from research.retrieval.search import CorpusSearch
from research.storage.store import ResearchStore

#: The combinations worth reporting. Graph expansion needs lexical seeds, so
#: there is no graph-only row: it would measure nothing.
ABLATIONS: dict[str, tuple[str, ...]] = {
    "lexical": ("lexical",),
    "lexical+graph": ("lexical", "graph"),
    "lexical+vector": ("lexical", "vector"),
    "vector": ("vector",),
    "all": ("lexical", "graph", "vector"),
}


@dataclass(slots=True)
class RetrievalResult:
    retrievers: str
    scores: list[RankingScore] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, float]:
        return average(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrievers": self.retrievers,
            **{name: round(value, 4) for name, value in self.summary.items()},
            "queries": len(self.scores),
        }


async def evaluate_retrieval(
    *,
    dataset: str = "retrieval",
    embeddings: EmbeddingIndex | None = None,
    ablations: Sequence[str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Score the query set under each combination of retrievers."""
    data = load_dataset(dataset)
    queries = data.get("queries") or []
    wanted = list(ablations or ABLATIONS)
    if embeddings is None:
        # Reporting a vector row without an embedder would print zeroes that
        # look like a finding rather than an absence.
        wanted = [name for name in wanted if "vector" not in ABLATIONS[name]]

    with ResearchStore.in_memory() as store:
        corpus = build_corpus(store)
        index = _rebind(embeddings, store) if embeddings is not None else None
        search = CorpusSearch(store, corpus.investigation_id, embeddings=index)

        results = []
        for name in wanted:
            results.append(
                await _score(search, corpus, queries, ABLATIONS[name], name, limit=limit)
            )
        per_query = _per_query(results, corpus)

    return {
        "dataset": data.get("name", dataset),
        "documents": 15,
        "queries": len(queries),
        "ablations": [result.to_dict() for result in results],
        "per_query": per_query,
        "note": data.get("note", ""),
    }


async def _score(
    search: CorpusSearch,
    corpus: EvaluationCorpus,
    queries: Sequence[dict[str, Any]],
    retrievers: tuple[str, ...],
    name: str,
    *,
    limit: int,
) -> RetrievalResult:
    result = RetrievalResult(retrievers=name)
    for entry in queries:
        relevant = corpus.resolve(entry.get("relevant") or [])
        hits = await search.search(
            entry["query"],
            limit=limit,
            retrievers=retrievers,
            # Copies are folded away in normal use, which would make a query
            # whose answer is three copies of one story look like a miss.
            # An evaluation of *retrieval* wants what retrieval found.
            collapse_copies=False,
        )
        ranked = [hit.document.id for hit in hits]
        result.scores.append(RankingScore.of(entry["query"], ranked, relevant))
    return result


def _per_query(
    results: Sequence[RetrievalResult], corpus: EvaluationCorpus
) -> list[dict[str, Any]]:
    """The full configuration, query by query, so a bad number can be chased."""
    full = next((result for result in results if result.retrievers == "all"), None)
    full = full or (results[-1] if results else None)
    if full is None:
        return []
    return [
        {
            "query": score.query,
            "recall_at_10": round(score.recall_at_10, 3),
            "mrr": round(score.mrr, 3),
            "missed": corpus.label(list(score.missed)),
        }
        for score in full.scores
    ]


def _rebind(embeddings: EmbeddingIndex, store: ResearchStore) -> EmbeddingIndex:
    """Point an index at the evaluation's own store rather than the caller's."""
    return EmbeddingIndex(store.db, embeddings.client)
