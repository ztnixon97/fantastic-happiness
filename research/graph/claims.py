"""Assessing what the held evidence actually says about a claim.

This is where a numeric confidence score would normally go. There is not one,
because there is no methodology behind such a number that would survive
inspection. Instead the evidence is characterised: how many *independent*
sources support it, whether any of them is a primary record, whether the
contradicting evidence is newer than the supporting evidence, and whether the
only support comes from the party making the announcement.

The output is a status plus a sentence a reader can check - "supported by two
independent primary sources", "supported only by secondary reporting (three
copies of one wire story)", "announced by the party involved and not
independently verified".

Everything here is derived from stored relationships. Nothing is asked of a
model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from research.graph.independence import independent_documents
from research.models.claim import Claim, ClaimEvidenceLink, ClaimStatus, EvidenceStance
from research.models.common import SourceType
from research.models.evidence import EvidenceDocument
from research.storage.store import ResearchStore

#: Source types that are the record itself rather than an account of it.
PRIMARY_SOURCE_TYPES = frozenset(
    {
        SourceType.CORPORATE_FILING,
        SourceType.REGULATORY_DOCUMENT,
        SourceType.GOVERNMENT_DOCUMENT,
        SourceType.TRANSCRIPT,
    }
)

#: Material published by an interested party about itself. It is evidence that
#: something was *said*, which is not the same as evidence that it is so.
SELF_REPORTED_TYPES = frozenset(
    {
        SourceType.PRESS_RELEASE,
        SourceType.OFFICIAL_STATEMENT,
        SourceType.SOCIAL_POST,
    }
)

#: Accounts at a distance from the record.
SECONDARY_TYPES = frozenset(
    {
        SourceType.SECONDARY_NEWS_REPORTING,
        SourceType.WEB_PAGE,
        SourceType.FORUM_POST,
        SourceType.OTHER,
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    """What a set of documents amounts to, once copies are collapsed."""

    document_ids: list[str] = field(default_factory=list)
    independent_groups: list[list[str]] = field(default_factory=list)
    primary_source_ids: list[str] = field(default_factory=list)
    preprint_ids: list[str] = field(default_factory=list)
    self_reported_ids: list[str] = field(default_factory=list)
    retracted_ids: list[str] = field(default_factory=list)
    source_types: dict[str, int] = field(default_factory=dict)
    wire_services: list[str] = field(default_factory=list)
    earliest: datetime | None = None
    latest: datetime | None = None

    @property
    def count(self) -> int:
        """Documents held. Not the number of sources."""
        return len(self.document_ids)

    @property
    def independent_count(self) -> int:
        """Independent sources. This is the number a report may cite."""
        return len(self.independent_groups)

    @property
    def has_primary_source(self) -> bool:
        return bool(self.primary_source_ids)

    @property
    def only_self_reported(self) -> bool:
        return bool(self.document_ids) and len(self.self_reported_ids) == len(self.document_ids)

    @property
    def only_secondary(self) -> bool:
        secondary = sum(
            count
            for source_type, count in self.source_types.items()
            if SourceType.coerce(source_type, SourceType.OTHER) in SECONDARY_TYPES
        )
        return bool(self.document_ids) and secondary == len(self.document_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "documents": list(self.document_ids),
            "independent_sources": self.independent_count,
            "independent_groups": [list(group) for group in self.independent_groups],
            "primary_sources": list(self.primary_source_ids),
            "preprints": list(self.preprint_ids),
            "self_reported": list(self.self_reported_ids),
            "retracted": list(self.retracted_ids),
            "source_types": dict(self.source_types),
            "wire_services": list(self.wire_services),
            "earliest": self.earliest.isoformat() if self.earliest else None,
            "latest": self.latest.isoformat() if self.latest else None,
        }


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    """A claim's evidentiary situation, stated in full."""

    claim_id: str
    text: str
    status: ClaimStatus
    explanation: str
    support: EvidenceSummary
    contradiction: EvidenceSummary
    qualification: EvidenceSummary
    #: Concrete things that would improve the assessment, in priority order.
    gaps: list[str] = field(default_factory=list)
    #: True when the newest contradicting evidence postdates every supporting
    #: document: the sign of a finding that may have been superseded.
    possibly_superseded: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "status": str(self.status),
            "explanation": self.explanation,
            "support": self.support.to_dict(),
            "contradiction": self.contradiction.to_dict(),
            "qualification": self.qualification.to_dict(),
            "gaps": list(self.gaps),
            "possibly_superseded": self.possibly_superseded,
        }


class ClaimGraph:
    """Reads claims together with the evidence attached to them."""

    def __init__(self, store: ResearchStore, investigation_id: str) -> None:
        self.store = store
        self.investigation_id = investigation_id

    # -- assembly -------------------------------------------------------
    def summarise(self, documents: Iterable[EvidenceDocument]) -> EvidenceSummary:
        documents = list(documents)
        if not documents:
            return EvidenceSummary()

        groups = independent_documents(self.store.documents, [d.id for d in documents])
        dates = [document.published_at for document in documents if document.published_at]
        source_types: dict[str, int] = {}
        for document in documents:
            key = str(document.source_type)
            source_types[key] = source_types.get(key, 0) + 1

        wire_services = sorted(
            {
                str(document.metadata["wire_service"])
                for document in documents
                if document.metadata.get("wire_service")
            }
        )
        return EvidenceSummary(
            document_ids=[document.id for document in documents],
            independent_groups=groups,
            primary_source_ids=[
                document.id
                for document in documents
                if document.source_type in PRIMARY_SOURCE_TYPES
            ],
            preprint_ids=[
                document.id
                for document in documents
                if document.source_type is SourceType.ACADEMIC_PREPRINT
            ],
            self_reported_ids=[
                document.id
                for document in documents
                if document.source_type in SELF_REPORTED_TYPES
            ],
            retracted_ids=[
                document.id
                for document in documents
                if document.metadata.get("is_retracted")
            ],
            source_types=source_types,
            wire_services=wire_services,
            earliest=min(dates) if dates else None,
            latest=max(dates) if dates else None,
        )

    def assess(self, claim_id: str) -> ClaimAssessment:
        claim = self.store.claims.get(claim_id)
        links = self.store.claims.evidence_links(claim_id)
        by_stance: dict[EvidenceStance, list[str]] = {}
        for link in links:
            by_stance.setdefault(link.stance, []).append(link.document_id)

        support = self.summarise(
            self.store.documents.get_many(by_stance.get(EvidenceStance.SUPPORTS, []))
        )
        contradiction = self.summarise(
            self.store.documents.get_many(by_stance.get(EvidenceStance.CONTRADICTS, []))
        )
        qualification = self.summarise(
            self.store.documents.get_many(by_stance.get(EvidenceStance.QUALIFIES, []))
        )

        status = derive_status(support, contradiction)
        superseded = _is_possibly_superseded(support, contradiction)
        return ClaimAssessment(
            claim_id=claim.id,
            text=claim.text,
            status=status,
            explanation=explain(support, contradiction, qualification, superseded=superseded),
            support=support,
            contradiction=contradiction,
            qualification=qualification,
            gaps=identify_gaps(support, contradiction, links),
            possibly_superseded=superseded,
        )

    def assess_all(self, *, limit: int = 200) -> list[ClaimAssessment]:
        return [
            self.assess(claim.id)
            for claim in self.store.claims.list(self.investigation_id, limit=limit, hydrate=False)
        ]

    def claims_for_document(self, document_id: str) -> list[tuple[Claim, ClaimEvidenceLink]]:
        return [
            (self.store.claims.get(link.claim_id, hydrate=False), link)
            for link in self.store.claims.claims_for_document(document_id)
        ]


# ---------------------------------------------------------------- deriving
def derive_status(
    support: EvidenceSummary, contradiction: EvidenceSummary
) -> ClaimStatus:
    """Status from the evidence, by rules a reader can check.

    Retracted work is discounted on both sides: a withdrawn paper is not
    support, and it is not a rebuttal either. 'Insufficient' is a real answer
    and is used - a single cluster of secondary reporting, or a company's own
    announcement about itself, is not the same as support.
    """
    if not support.count and not contradiction.count:
        return ClaimStatus.UNVERIFIED

    standing_support = support.count - len(support.retracted_ids)
    standing_contradiction = contradiction.count - len(contradiction.retracted_ids)

    if standing_support and standing_contradiction:
        return ClaimStatus.MIXED
    if standing_contradiction:
        return ClaimStatus.CONTRADICTED
    if not standing_support:
        # Everything linked has been withdrawn.
        return ClaimStatus.INSUFFICIENT_EVIDENCE

    thin = support.independent_count <= 1 and not support.has_primary_source
    if thin and (support.only_secondary or support.only_self_reported):
        return ClaimStatus.INSUFFICIENT_EVIDENCE
    return ClaimStatus.SUPPORTED


def explain(
    support: EvidenceSummary,
    contradiction: EvidenceSummary,
    qualification: EvidenceSummary | None = None,
    *,
    superseded: bool = False,
) -> str:
    """Prose describing the evidence, of the kind a footnote can carry."""
    parts: list[str] = []
    if support.count:
        parts.append(f"supported by {_describe(support)}")
    if contradiction.count:
        verb = "contradicted by" if support.count else "contradicted by"
        parts.append(f"{verb} {_describe(contradiction)}")
    if qualification and qualification.count:
        parts.append(f"qualified by {_describe(qualification)}")
    if not parts:
        return "no evidence has been linked to this claim yet"

    sentence = "; ".join(parts)
    notes: list[str] = []
    if support.only_self_reported:
        notes.append(
            "the support is the announcing party's own material, so this is evidence "
            "that the claim was made, not that it holds"
        )
    if support.wire_services and support.independent_count < support.count:
        notes.append(
            f"copies of one {support.wire_services[0]} story are counted once"
        )
    if support.preprint_ids and not support.has_primary_source:
        notes.append("the supporting work includes preprints, which are not peer reviewed")
    if support.retracted_ids:
        notes.append(f"{_items(len(support.retracted_ids))} supporting it")
    if contradiction.retracted_ids:
        notes.append(f"{_items(len(contradiction.retracted_ids))} contradicting it")
    if superseded:
        notes.append("the contradicting evidence is more recent than any supporting evidence")
    if notes:
        sentence = f"{sentence} ({'; '.join(notes)})"
    return sentence


def _items(count: int) -> str:
    """'1 retracted item' / '2 retracted items', for note text."""
    return f"{count} retracted item" if count == 1 else f"{count} retracted items"


def _describe(summary: EvidenceSummary) -> str:
    """Describe a body of evidence in terms of independent sources."""
    sources = summary.independent_count
    noun = "source" if sources == 1 else "sources"
    descriptor = _quality_word(summary)
    text = f"{sources} independent {descriptor} {noun}".replace("  ", " ")
    if summary.count > sources:
        copies = summary.count - sources
        text += f" ({summary.count} documents, {copies} of them copies or rewrites)"
    if summary.earliest and summary.latest:
        if summary.earliest.year == summary.latest.year:
            text += f", {summary.earliest.year}"
        else:
            text += f", {summary.earliest.year}-{summary.latest.year}"
    return text


def _quality_word(summary: EvidenceSummary) -> str:
    if summary.has_primary_source:
        return "primary"
    if summary.only_self_reported:
        return "self-reported"
    if summary.only_secondary:
        return "secondary"
    peer_reviewed = summary.source_types.get(str(SourceType.ACADEMIC_PEER_REVIEWED), 0)
    if peer_reviewed and peer_reviewed == summary.count:
        return "peer-reviewed"
    if summary.preprint_ids:
        return "preprint"
    return ""


def _is_possibly_superseded(
    support: EvidenceSummary, contradiction: EvidenceSummary
) -> bool:
    if not (support.latest and contradiction.latest):
        return False
    return contradiction.latest > support.latest


def identify_gaps(
    support: EvidenceSummary,
    contradiction: EvidenceSummary,
    links: list[ClaimEvidenceLink],
) -> list[str]:
    """Concrete, actionable shortcomings, in the order worth addressing them."""
    gaps: list[str] = []
    if not links:
        gaps.append("no evidence is linked to this claim")
        return gaps
    if support.count and support.independent_count <= 1:
        gaps.append(
            "all support comes from a single source; look for independent confirmation"
        )
    if support.count and not support.has_primary_source:
        gaps.append(
            "no primary record among the supporting evidence; trace the assertion to a "
            "filing, dataset, transcript or paper"
        )
    if support.count and not contradiction.count:
        gaps.append("no contradicting evidence has been linked; run a counterevidence search")
    if support.only_self_reported:
        gaps.append("only the announcing party's own material supports this")
    if support.preprint_ids and not any(
        source_type == str(SourceType.ACADEMIC_PEER_REVIEWED)
        for source_type in support.source_types
    ):
        gaps.append("supporting literature is preprint-only; look for peer-reviewed work")
    if not any(link.excerpt for link in links):
        gaps.append("no linked evidence carries a verbatim excerpt to check")
    return gaps
