"""Claims: structured propositions about the world.

A Claim is not evidence and not analysis. It is a proposition that evidence
can support or contradict. Keeping the three separate is what makes a report
traceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from research.models.common import StrEnum, utcnow


class ClaimStatus(StrEnum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidenceStance(StrEnum):
    """How a document relates to a claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    QUALIFIES = "qualifies"
    MENTIONS = "mentions"


@dataclass(slots=True)
class ClaimEvidenceLink:
    """A single evidence-to-claim relationship, with its own provenance."""

    claim_id: str
    document_id: str
    stance: EvidenceStance
    #: Verbatim span from the document that motivated the link. Quoted
    #: external text, never model prose.
    excerpt: str | None = None
    #: Model reasoning about *why* this document bears on the claim. Analysis,
    #: explicitly separated from the evidence itself.
    analysis: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    created_by_task_id: str | None = None


@dataclass(slots=True)
class Claim:
    """A proposition under investigation."""

    id: str
    investigation_id: str
    text: str
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    entity_ids: list[str] = field(default_factory=list)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    related_claim_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    created_by_task_id: str | None = None
    #: Prose explanation of the evidentiary situation, e.g. "supported by one
    #: preprint and contradicted by two later studies". Deliberately used in
    #: place of an invented numeric confidence score.
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
