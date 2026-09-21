"""Claim operations.

The verbs a research worker uses to turn retrieved material into an argument:
state a proposition, attach the evidence that bears on it, and let the status
follow from what is attached rather than from what the worker believes.

Two integrity rules are enforced here rather than requested in a prompt:

* A quotation attached to a claim must actually appear in the document it is
  attributed to. A fabricated excerpt is refused.
* Analysis is stored in its own field, never in the excerpt field, so a
  report can always separate what a source said from what a model concluded.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

from research.errors import IntegrityError
from research.graph.claims import ClaimAssessment, ClaimGraph
from research.models.claim import Claim, ClaimEvidenceLink, ClaimStatus, EvidenceStance
from research.models.evidence import EvidenceDocument
from research.normalize.text import normalize_whitespace, truncate
from research.storage.store import ResearchStore

#: Claim-to-claim relations. ``REFINES`` is the useful one in practice: a
#: narrower, better-evidenced restatement of a claim that was too broad.
CLAIM_RELATIONS = ("SUPPORTS", "CONTRADICTS", "REFINES", "RELATED_TO")

_QUOTES = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}


def _comparable(text: str | None) -> str:
    """Fold a string to the form used for verbatim comparison.

    Publishers and models differ on quotation marks, dashes and spacing; none
    of those differences make a quotation inauthentic. Wording differences do.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    for character, replacement in _QUOTES.items():
        folded = folded.replace(character, replacement)
    folded = re.sub(r"\s+", " ", folded)
    return folded.casefold().strip()


def excerpt_appears_in(document: EvidenceDocument, excerpt: str) -> bool:
    """Whether ``excerpt`` is really present in the document's own text."""
    needle = _comparable(excerpt)
    if not needle:
        return False
    haystacks = (document.text, document.abstract, document.title)
    return any(needle in _comparable(hay) for hay in haystacks if hay)


@dataclass(slots=True)
class LinkResult:
    """Outcome of attaching a document to a claim."""

    link: ClaimEvidenceLink
    claim: Claim
    #: Set when the linked document is a copy of another held document; the
    #: link stands, but it adds no independent weight.
    not_independent_of: str | None = None
    excerpt_verified: bool = False


class ClaimOperations:
    """Create claims, attach evidence, and keep status honest."""

    def __init__(
        self,
        store: ResearchStore,
        *,
        investigation_id: str,
        task_id: str | None = None,
    ) -> None:
        self.store = store
        self.investigation_id = investigation_id
        self.task_id = task_id
        self.graph = ClaimGraph(store, investigation_id)

    # -- creation -------------------------------------------------------
    def create_claim(
        self,
        text: str,
        *,
        entity_ids: Sequence[str] = (),
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Claim:
        """State a proposition. It starts unverified, whatever anyone believes."""
        cleaned = normalize_whitespace(text)
        if not cleaned:
            raise IntegrityError("a claim needs text")
        claim = self.store.claims.create(
            self.investigation_id,
            cleaned,
            notes=notes,
            created_by_task_id=self.task_id,
            metadata=dict(metadata or {}),
        )
        for entity_id in entity_ids:
            self.link_entity(claim.id, entity_id)
        return claim

    def link_entity(self, claim_id: str, entity_id: str) -> None:
        """Record that a claim is about an entity, as a graph edge."""
        self.store.entities.get(entity_id)  # raises if unknown
        self.store.relationships.add(
            self.investigation_id,
            subject_type="claim",
            subject_id=claim_id,
            predicate="ABOUT",
            object_type="entity",
            object_id=entity_id,
            created_by_task_id=self.task_id,
        )

    def relate_claims(self, claim_id: str, related_claim_id: str, relation: str) -> None:
        relation = relation.upper()
        if relation not in CLAIM_RELATIONS:
            raise IntegrityError(
                f"unknown claim relation {relation!r}; expected one of {CLAIM_RELATIONS}"
            )
        if claim_id == related_claim_id:
            raise IntegrityError("a claim cannot be related to itself")
        self.store.claims.get(claim_id, hydrate=False)
        self.store.claims.get(related_claim_id, hydrate=False)
        self.store.claims.link_claims(claim_id, related_claim_id, relation)

    # -- evidence -------------------------------------------------------
    def link_evidence(
        self,
        claim_id: str,
        document_id: str,
        stance: EvidenceStance,
        *,
        excerpt: str | None = None,
        analysis: str | None = None,
        refresh_status: bool = True,
    ) -> LinkResult:
        """Attach a document to a claim, with an optional verbatim excerpt.

        The excerpt is checked against the document. This is the one place
        where a model's output could otherwise become indistinguishable from
        retrieved material, so it is checked in code.
        """
        claim = self.store.claims.get(claim_id, hydrate=False)
        if claim.investigation_id != self.investigation_id:
            raise IntegrityError(
                f"{claim_id} belongs to {claim.investigation_id}, not {self.investigation_id}"
            )
        document = self.store.documents.get(document_id)
        if document.investigation_id != self.investigation_id:
            raise IntegrityError(
                f"{document_id} belongs to {document.investigation_id}, "
                f"not {self.investigation_id}"
            )

        verified = False
        if excerpt is not None:
            if not excerpt_appears_in(document, excerpt):
                raise IntegrityError(
                    f"the excerpt does not appear in {document_id}: "
                    f"{truncate(normalize_whitespace(excerpt), 120)!r}. "
                    "Quote the document or leave the excerpt empty and put the "
                    "reasoning in 'analysis'."
                )
            verified = True

        link = ClaimEvidenceLink(
            claim_id=claim_id,
            document_id=document_id,
            stance=stance,
            excerpt=normalize_whitespace(excerpt) if excerpt else None,
            analysis=normalize_whitespace(analysis) if analysis else None,
            created_by_task_id=self.task_id,
        )
        self.store.claims.link_evidence(link)
        self.store.relationships.add(
            self.investigation_id,
            subject_type="document",
            subject_id=document_id,
            predicate=str(stance).upper(),
            object_type="claim",
            object_id=claim_id,
            evidence_document_id=document_id,
            created_by_task_id=self.task_id,
        )
        if refresh_status:
            self.refresh_status(claim_id)
        return LinkResult(
            link=link,
            claim=self.store.claims.get(claim_id),
            not_independent_of=document.duplicate_of or document.derived_from,
            excerpt_verified=verified,
        )

    # -- reading --------------------------------------------------------
    def get_claim(self, claim_id: str) -> ClaimAssessment:
        return self.graph.assess(claim_id)

    def list_claims(self, *, status: ClaimStatus | None = None, limit: int = 200) -> list[Claim]:
        return self.store.claims.list(self.investigation_id, status=status, limit=limit)

    def refresh_status(self, claim_id: str) -> ClaimAssessment:
        """Recompute status and notes from what is attached, and persist them."""
        assessment = self.graph.assess(claim_id)
        self.store.claims.set_status(
            claim_id, assessment.status, notes=assessment.explanation
        )
        return assessment

    def refresh_all(self) -> list[ClaimAssessment]:
        return [
            self.refresh_status(claim.id)
            for claim in self.store.claims.list(self.investigation_id, hydrate=False)
        ]

    def open_questions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Claims whose evidence is thin, with the reason and what would fix it.

        This is what a follow-up planner reads: not 'what is unknown' in the
        abstract, but which specific propositions are under-evidenced and in
        what way.
        """
        questions: list[dict[str, Any]] = []
        for assessment in self.graph.assess_all(limit=limit):
            if assessment.status in (ClaimStatus.SUPPORTED,) and not assessment.gaps:
                continue
            if not assessment.gaps:
                continue
            questions.append(
                {
                    "claim_id": assessment.claim_id,
                    "text": assessment.text,
                    "status": str(assessment.status),
                    "explanation": assessment.explanation,
                    "gaps": list(assessment.gaps),
                    "possibly_superseded": assessment.possibly_superseded,
                }
            )
        return questions

