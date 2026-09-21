"""Is the system right about which sources are independent?

This is the capability nothing else in the surveyed field has, so it is the
one most worth measuring. Getting it wrong is not a ranking nuisance: ten
outlets carrying one wire story counted as ten confirmations is a claim
reported as corroborated when it is not.

The positive class is "not independent", because that is the judgement with
consequences. A false positive merges two real sources into one and
understates the evidence; a false negative is the failure that inflates it.
Both are counted separately for that reason.
"""

from __future__ import annotations

from typing import Any

from research.evaluation.corpus import build_corpus, load_dataset
from research.evaluation.metrics import Classification
from research.storage.store import ResearchStore


async def evaluate_independence(*, dataset: str = "independence") -> dict[str, Any]:
    data = load_dataset(dataset)
    pairs = data.get("pairs") or []

    with ResearchStore.in_memory() as store:
        corpus = build_corpus(store)
        true_positive = false_positive = false_negative = true_negative = 0
        mistakes: list[dict[str, Any]] = []

        for pair in pairs:
            left_id = corpus.ids.get(pair["left"])
            right_id = corpus.ids.get(pair["right"])
            if not left_id or not right_id:
                continue
            expected_independent = bool(pair.get("independent", True))
            observed_independent = _independent(store, left_id, right_id)

            if not expected_independent and not observed_independent:
                true_positive += 1
            elif expected_independent and not observed_independent:
                false_positive += 1
                mistakes.append(_mistake(pair, "merged two independent sources"))
            elif not expected_independent and observed_independent:
                false_negative += 1
                mistakes.append(_mistake(pair, "counted one source twice"))
            else:
                true_negative += 1

    matrix = Classification(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )
    return {
        "dataset": data.get("name", dataset),
        "pairs": len(pairs),
        "positive_class": "not independent",
        **matrix.to_dict(),
        "mistakes": mistakes,
        "note": data.get("note", ""),
    }


def _independent(store: ResearchStore, left: str, right: str) -> bool:
    """Whether the store treats these two documents as separate sources.

    Read the way the rest of the system reads it: two documents are one
    source when they share an independence key, whichever direction the
    copy relation happens to point.
    """
    documents = {document.id: document for document in store.documents.get_many([left, right])}
    first, second = documents.get(left), documents.get(right)
    if first is None or second is None:  # pragma: no cover - ids come from the map
        return True
    keys = {
        first.independence_key or first.id,
        second.independence_key or second.id,
    }
    if len(keys) == 1:
        return False
    return not (
        first.duplicate_of == second.id
        or second.duplicate_of == first.id
        or first.derived_from == second.id
        or second.derived_from == first.id
    )


def _mistake(pair: dict[str, Any], what: str) -> dict[str, Any]:
    return {
        "pair": f"{pair['left']} / {pair['right']}",
        "error": what,
        **({"note": pair["note"]} if pair.get("note") else {}),
    }
