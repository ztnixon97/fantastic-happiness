"""Ranking metrics, and nothing else.

Deliberately the textbook definitions with no tuning in them: a harness
whose metrics have opinions cannot be used to settle an argument about the
thing it measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Share of the relevant documents that appear in the top k."""
    if not relevant:
        return 0.0
    found = set(ranked[:k]) & set(relevant)
    return len(found) / len(set(relevant))


def precision_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0 or not ranked:
        return 0.0
    top = ranked[:k]
    return len(set(top) & set(relevant)) / len(top)


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    """1/rank of the first relevant result, or 0 if none is ranked."""
    wanted = set(relevant)
    for position, document_id in enumerate(ranked, start=1):
        if document_id in wanted:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalised discounted cumulative gain, binary relevance.

    Rewards putting relevant results near the top rather than merely
    somewhere in the list, which is what recall alone cannot see.
    """
    wanted = set(relevant)
    if not wanted:
        return 0.0
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, document_id in enumerate(ranked[:k], start=1)
        if document_id in wanted
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(wanted), k) + 1))
    return gain / ideal if ideal else 0.0


@dataclass(frozen=True, slots=True)
class RankingScore:
    """One ranking, scored. Averaged across a query set by :func:`average`."""

    query: str
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_10: float
    retrieved: int
    relevant: int
    missed: tuple[str, ...] = ()

    @classmethod
    def of(cls, query: str, ranked: Sequence[str], relevant: Sequence[str]) -> "RankingScore":
        return cls(
            query=query,
            recall_at_5=recall_at_k(ranked, relevant, 5),
            recall_at_10=recall_at_k(ranked, relevant, 10),
            precision_at_5=precision_at_k(ranked, relevant, 5),
            mrr=reciprocal_rank(ranked, relevant),
            ndcg_at_10=ndcg_at_k(ranked, relevant, 10),
            retrieved=len(ranked),
            relevant=len(set(relevant)),
            missed=tuple(sorted(set(relevant) - set(ranked[:10]))),
        )


def average(scores: Sequence[RankingScore]) -> dict[str, float]:
    """Mean of each metric across a query set."""
    if not scores:
        return {}
    fields = ("recall_at_5", "recall_at_10", "precision_at_5", "mrr", "ndcg_at_10")
    return {
        field: sum(getattr(score, field) for score in scores) / len(scores)
        for field in fields
    }


@dataclass(frozen=True, slots=True)
class Classification:
    """A confusion matrix, for questions with a right answer."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def accuracy(self) -> float:
        total = (
            self.true_positive + self.false_positive
            + self.false_negative + self.true_negative
        )
        return (self.true_positive + self.true_negative) / total if total else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
        }
