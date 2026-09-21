"""Persisting retrieved material as evidence.

One path in: a normalised document arrives, is compared against what is
already held, is charged against the budget, and is stored with its
provenance and its independence group. Nothing else writes to the document
table.
"""

from __future__ import annotations

from dataclasses import dataclass

from research.acquisition.deduplicate import DuplicateDetector, DuplicateVerdict
from research.normalize.document import enrich
from research.errors import BudgetExceeded
from research.models.common import DuplicateRelation
from research.models.evidence import EvidenceDocument
from research.orchestration.budgets import BudgetLedger
from research.storage.store import ResearchStore

#: Relations where the incoming copy is the *same artifact* already held: the
#: held document is enriched and no second row is written.
_MERGE_RELATIONS = frozenset(
    {DuplicateRelation.SAME_RECORD, DuplicateRelation.EXACT_DUPLICATE}
)


@dataclass(slots=True)
class AcquisitionResult:
    """What happened to one candidate document."""

    document: EvidenceDocument
    verdict: DuplicateVerdict
    created: bool
    enriched: bool = False
    #: Set when the document was refused because a budget was exhausted.
    refused: str | None = None

    @property
    def is_new_evidence(self) -> bool:
        """True only for material that counts as an independent source."""
        return self.created and self.verdict.independent


class EvidenceAcquirer:
    """Normalised document in, persisted evidence out."""

    def __init__(
        self,
        store: ResearchStore,
        *,
        ledger: BudgetLedger | None = None,
        detector: DuplicateDetector | None = None,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.detector = detector or DuplicateDetector(store.documents)

    def persist(
        self,
        document: EvidenceDocument,
        *,
        investigation_id: str | None,
    ) -> AcquisitionResult:
        document.investigation_id = investigation_id
        verdict = self.detector.check(document, investigation_id)

        if verdict.relation in _MERGE_RELATIONS and verdict.matched_document_id:
            held = self.store.documents.get(verdict.matched_document_id)
            changed = enrich(held, document)
            if changed:
                self.store.documents.update(held)
            if document.provenance and document.provenance.fetch_id:
                # The call still happened and still produced this document;
                # the log must not show a retrieval that led nowhere.
                self.store.fetches.attach_document(
                    document.provenance.fetch_id, held.id
                )
            return AcquisitionResult(
                document=held, verdict=verdict, created=False, enriched=changed
            )

        if self.ledger is not None and not self.ledger.can_spend_document(
            document.source_family
        ):
            return AcquisitionResult(
                document=document,
                verdict=verdict,
                created=False,
                refused="document budget exhausted",
            )

        document.id = self.store.documents.new_id()
        independence_key = document.id
        if verdict.matched_document_id and not verdict.independent:
            matched = self.store.documents.get(verdict.matched_document_id)
            independence_key = matched.independence_key or matched.id
            if verdict.is_duplicate:
                document.duplicate_of = matched.id
            else:
                document.derived_from = matched.id
            document.duplicate_relation = verdict.relation
            document.metadata["duplicate_reason"] = verdict.explanation
            if verdict.similarity is not None:
                document.metadata["duplicate_similarity"] = verdict.similarity
        document.independence_key = independence_key

        if self.ledger is not None:
            try:
                self.ledger.spend_document(document.source_family)
            except BudgetExceeded as exc:
                return AcquisitionResult(
                    document=document, verdict=verdict, created=False, refused=str(exc)
                )

        self.store.documents.add(document)

        if document.provenance and document.provenance.fetch_id:
            self.store.fetches.attach_document(document.provenance.fetch_id, document.id)
        if investigation_id and (document.doi or document.external_id):
            self.store.citations.resolve_document(
                investigation_id,
                document_id=document.id,
                doi=document.doi,
                external_id=document.external_id,
            )
        return AcquisitionResult(document=document, verdict=verdict, created=True)

    def persist_all(
        self,
        documents: list[EvidenceDocument],
        *,
        investigation_id: str | None,
    ) -> list[AcquisitionResult]:
        """Persist in order. Order matters: the first copy seen becomes the
        one later copies are judged against."""
        return [
            self.persist(document, investigation_id=investigation_id)
            for document in documents
        ]


def summarise(results: list[AcquisitionResult]) -> dict[str, int]:
    """Counts for a task result or CLI line."""
    summary = {
        "candidates": len(results),
        "new_evidence": 0,
        "duplicates": 0,
        "derived": 0,
        "merged": 0,
        "refused": 0,
    }
    for result in results:
        if result.refused:
            summary["refused"] += 1
        elif not result.created:
            summary["merged"] += 1
        elif result.verdict.is_duplicate:
            summary["duplicates"] += 1
        elif result.verdict.is_derived:
            summary["derived"] += 1
        else:
            summary["new_evidence"] += 1
    return summary
