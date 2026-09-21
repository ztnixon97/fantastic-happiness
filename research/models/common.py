"""Shared vocabulary for the research domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    """Timezone-aware 'now'. Centralised so tests can monkeypatch one place."""
    return datetime.now(timezone.utc)


class StrEnum(str, Enum):
    """Enum whose members compare and serialise as their string value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)

    @classmethod
    def coerce(cls, value: Any, default: "StrEnum | None" = None):
        """Best-effort parse; unknown values fall back to ``default``.

        External providers invent categories constantly; an unknown label must
        never crash acquisition.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                pass
        return default


class SourceFamily(StrEnum):
    """Coarse grouping the planner reasons about.

    The planner thinks 'search academic literature', never 'call provider X
    endpoint Y'. Provider selection happens below the planning layer.
    """

    ACADEMIC = "academic"
    NEWS = "news"
    WEB = "web"
    SOCIAL = "social"
    GOVERNMENT = "government"
    CORPORATE = "corporate"
    VIDEO = "video"


class SourceType(StrEnum):
    """What a specific document *is*.

    This is descriptive metadata used for reasoning about independence and
    primary-source distance. It is deliberately **not** a truth score.
    """

    ACADEMIC_PEER_REVIEWED = "academic_peer_reviewed"
    ACADEMIC_PREPRINT = "academic_preprint"
    GOVERNMENT_DOCUMENT = "government_document"
    REGULATORY_DOCUMENT = "regulatory_document"
    OFFICIAL_STATEMENT = "official_statement"
    CORPORATE_FILING = "corporate_filing"
    PRESS_RELEASE = "press_release"
    ORIGINAL_NEWS_REPORTING = "original_news_reporting"
    SECONDARY_NEWS_REPORTING = "secondary_news_reporting"
    SOCIAL_POST = "social_post"
    FORUM_POST = "forum_post"
    VIDEO = "video"
    TRANSCRIPT = "transcript"
    WEB_PAGE = "web_page"
    OTHER = "other"


#: Rough distance from the underlying evidence. Used by the primary-source
#: researcher to decide whether an assertion is worth chasing further. Lower
#: is closer to the original record. It is an ordering heuristic, not a
#: credibility ranking: a bad filing outranks a good news article here.
PRIMARY_SOURCE_DISTANCE: dict[SourceType, int] = {
    SourceType.CORPORATE_FILING: 0,
    SourceType.REGULATORY_DOCUMENT: 0,
    SourceType.GOVERNMENT_DOCUMENT: 0,
    SourceType.OFFICIAL_STATEMENT: 0,
    SourceType.TRANSCRIPT: 0,
    SourceType.ACADEMIC_PEER_REVIEWED: 1,
    SourceType.ACADEMIC_PREPRINT: 1,
    SourceType.PRESS_RELEASE: 1,
    SourceType.SOCIAL_POST: 1,
    SourceType.VIDEO: 1,
    SourceType.ORIGINAL_NEWS_REPORTING: 2,
    SourceType.FORUM_POST: 3,
    SourceType.SECONDARY_NEWS_REPORTING: 3,
    SourceType.WEB_PAGE: 3,
    SourceType.OTHER: 4,
}


class RetrievalMethod(StrEnum):
    """How a document entered the investigation."""

    SEARCH = "search"
    DIRECT_FETCH = "direct_fetch"
    REFERENCE_EXPANSION = "reference_expansion"  # backward through citations
    CITATION_EXPANSION = "citation_expansion"  # forward through citations
    PRIMARY_SOURCE_CHASE = "primary_source_chase"
    COUNTEREVIDENCE_SEARCH = "counterevidence_search"
    SEED = "seed"  # supplied by the user


class DuplicateRelation(StrEnum):
    """Why one document is considered non-independent of another."""

    EXACT_DUPLICATE = "exact_duplicate"  # same bytes/text, same story
    NEAR_DUPLICATE = "near_duplicate"  # trivially reformatted copy
    SYNDICATED_COPY = "syndicated_copy"  # wire copy republished elsewhere
    DERIVED_ARTICLE = "derived_article"  # written primarily from another item
    SAME_RECORD = "same_record"  # same paper from two academic providers


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a piece of material came from and how it was obtained.

    Every EvidenceDocument carries one. Provenance is what makes a final
    report inspectable: a reader can follow a statement to a claim, to
    evidence, to the provider call that produced it.
    """

    provider: str
    retrieval_method: RetrievalMethod = RetrievalMethod.SEARCH
    retrieved_at: datetime = field(default_factory=utcnow)
    #: The endpoint/URL actually called (never a secret-bearing URL).
    provider_endpoint: str | None = None
    #: `search_queries.id` that produced this document, when applicable.
    query_id: str | None = None
    #: `source_fetches.id` for the HTTP call that produced this document.
    fetch_id: str | None = None
    #: The URL requested, before redirects.
    requested_url: str | None = None
    #: Final URL after redirects, when different.
    final_url: str | None = None
    #: The document this one was reached from (citation chase, primary-source
    #: chase, link following).
    parent_document_id: str | None = None
    #: The research task that was running at acquisition time.
    task_id: str | None = None
    notes: str | None = None

    def describe(self) -> str:
        """Short human-readable trace line, for CLI and report footnotes."""
        bits = [f"{self.provider} via {self.retrieval_method}"]
        if self.requested_url:
            bits.append(self.requested_url)
        if self.parent_document_id:
            bits.append(f"reached from {self.parent_document_id}")
        bits.append(self.retrieved_at.isoformat())
        return " | ".join(bits)
