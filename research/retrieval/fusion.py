"""Combining rankings from retrievers that do not share a scale.

BM25 scores, graph connection weights and cosine similarities are not
comparable, and normalising them against each other invents a relationship
that is not there. Reciprocal rank fusion sidesteps the problem by using only
each retriever's *ordering*, which is the part every one of them agrees is
meaningful.

It has no tuned parameters beyond ``k``, which controls how much weight the
top of each list carries. 60 is the value from the original paper and is left
alone until there is evidence for changing it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

RRF_K = 60


@dataclass(slots=True)
class FusedHit:
    document_id: str
    score: float = 0.0
    #: Rank this document held in each retriever that returned it, 1-based.
    ranks: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def retrievers(self) -> list[str]:
        return sorted(self.ranks)


def reciprocal_rank_fusion(
    rankings: dict[str, Sequence[str]],
    *,
    weights: dict[str, float] | None = None,
    k: int = RRF_K,
    reasons: dict[str, Iterable[str]] | None = None,
) -> list[FusedHit]:
    """Fuse ordered lists of document ids into one ranking.

    ``rankings`` maps a retriever name to its results, best first.
    """
    weights = weights or {}
    fused: dict[str, FusedHit] = defaultdict(lambda: FusedHit(document_id=""))

    for retriever, ordered in rankings.items():
        weight = weights.get(retriever, 1.0)
        for position, document_id in enumerate(ordered, start=1):
            hit = fused[document_id]
            hit.document_id = document_id
            hit.score += weight / (k + position)
            hit.ranks[retriever] = min(position, hit.ranks.get(retriever, position))

    if reasons:
        for document_id, explanations in reasons.items():
            if document_id in fused:
                for reason in explanations:
                    if reason not in fused[document_id].reasons:
                        fused[document_id].reasons.append(reason)

    return sorted(fused.values(), key=lambda hit: (-hit.score, hit.document_id))
