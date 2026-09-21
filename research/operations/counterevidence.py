"""Looking for the evidence that would change the answer.

Searching for a proposition finds material that agrees with it; that is how
search works, and it is why a system that only searches confirms whatever it
started with. This operation searches *against* a proposition instead, along
the lines where refutation actually lives: failed replications, later
corrections, retractions, denials, cancellations, methodological criticism.

The query shaping is deterministic. Judging what the results mean is not, and
stays with the skeptic role.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from research.models.claim import EvidenceStance
from research.models.common import SourceFamily
from research.operations.citation_chase import CitationChase
from research.operations.search import SearchOperation
from research.storage.store import ResearchStore

#: Query shapes for scholarly refutation.
ACADEMIC_ANGLES = (
    "{proposition} failed replication",
    "{proposition} criticism methodology",
    "{proposition} contradictory evidence",
    "{proposition} retraction correction",
)

#: Query shapes for the news and official record.
NEWS_ANGLES = (
    "{proposition} denied disputed",
    "{proposition} cancelled delayed abandoned",
    "{proposition} correction retracted report",
)


@dataclass(slots=True)
class CounterevidenceResult:
    proposition: str
    claim_id: str | None = None
    queries: list[str] = field(default_factory=list)
    new_document_ids: list[str] = field(default_factory=list)
    all_document_ids: list[str] = field(default_factory=list)
    citing_papers: list[str] = field(default_factory=list)
    retracted_found: list[str] = field(default_factory=list)
    stopped_by: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "proposition": self.proposition,
            "claim_id": self.claim_id,
            "queries_run": len(self.queries),
            "new_evidence": list(self.new_document_ids),
            "forward_citations_checked": list(self.citing_papers),
            "retractions_found": list(self.retracted_found),
            "stopped_by": self.stopped_by,
        }


class CounterevidenceSearch:
    def __init__(
        self,
        store: ResearchStore,
        search: SearchOperation,
        *,
        investigation_id: str,
        citation_chase: CitationChase | None = None,
        task_id: str | None = None,
    ) -> None:
        self.store = store
        self.search = search
        self.investigation_id = investigation_id
        self.citation_chase = citation_chase
        self.task_id = task_id

    async def run(
        self,
        proposition: str,
        *,
        claim_id: str | None = None,
        families: Sequence[SourceFamily] = (SourceFamily.ACADEMIC, SourceFamily.NEWS),
        angles_per_family: int = 2,
        limit: int = 6,
    ) -> CounterevidenceResult:
        result = CounterevidenceResult(proposition=proposition, claim_id=claim_id)
        subject = proposition.strip().rstrip(".")

        for family in families:
            angles = ACADEMIC_ANGLES if family is SourceFamily.ACADEMIC else NEWS_ANGLES
            for template in angles[:angles_per_family]:
                query = template.format(proposition=subject)
                outcome = await self.search.search(
                    query,
                    family=family,
                    limit=limit,
                    objective=f"counterevidence against: {subject}",
                )
                result.queries.append(query)
                result.new_document_ids.extend(outcome.new_document_ids)
                result.all_document_ids.extend(outcome.document_ids)
                if outcome.stopped_by:
                    result.stopped_by = outcome.stopped_by
                    return self._finish(result)

        if claim_id and self.citation_chase is not None:
            await self._check_forward_citations(claim_id, result)
        return self._finish(result)

    async def _check_forward_citations(
        self, claim_id: str, result: CounterevidenceResult
    ) -> None:
        """Follow the supporting papers forward.

        Replications, corrections and rebuttals cite the work they are about,
        so the citing side of a supporting paper is where they are found.
        """
        supporting = [
            link.document_id
            for link in self.store.claims.evidence_links(claim_id)
            if link.stance is EvidenceStance.SUPPORTS
        ]
        for document_id in supporting[:3]:
            document = self.store.documents.get(document_id)
            if document.source_family is not SourceFamily.ACADEMIC:
                continue
            chase = await self.citation_chase.chase(
                document_id, direction="forward", max_depth=1, max_documents=8
            )
            result.citing_papers.extend(chase.added_document_ids)
            result.new_document_ids.extend(chase.added_document_ids)
            if chase.stopped_by and "budget" in chase.stopped_by:
                result.stopped_by = chase.stopped_by
                return

    def _finish(self, result: CounterevidenceResult) -> CounterevidenceResult:
        """Surface retractions found along the way; they are never incidental."""
        for document in self.store.documents.get_many(sorted(set(result.new_document_ids))):
            title = (document.title or "").lower()
            if document.metadata.get("is_retracted") or "retract" in title:
                result.retracted_found.append(document.id)
        return result
