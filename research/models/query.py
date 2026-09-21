"""Provider-independent query and hit representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from research.models.common import SourceFamily, SourceType, utcnow


@dataclass(slots=True)
class ResearchQuery:
    """What the research layer wants, expressed without provider specifics."""

    text: str
    families: list[SourceFamily] = field(default_factory=list)
    limit: int = 10
    published_after: datetime | None = None
    published_before: datetime | None = None
    language: str | None = None
    #: Optional hints a provider may or may not honour, e.g.
    #: ``{"open_access": True}`` or ``{"site": "nrc.gov"}``.
    filters: dict[str, Any] = field(default_factory=dict)
    #: Free-text statement of *why* this search is being run. Persisted with
    #: the query so an investigation log explains itself.
    objective: str | None = None
    investigation_id: str | None = None
    task_id: str | None = None

    def cache_key(self) -> str:
        parts = [
            self.text.strip().lower(),
            ",".join(sorted(str(f) for f in self.families)),
            str(self.limit),
            self.published_after.isoformat() if self.published_after else "",
            self.published_before.isoformat() if self.published_before else "",
            self.language or "",
            repr(sorted(self.filters.items())),
        ]
        return "|".join(parts)


@dataclass(slots=True)
class SearchHit:
    """A provider result before acquisition.

    A hit is a *pointer*: enough to decide whether to spend a fetch on it.
    Some providers (academic APIs) return enough metadata that the hit can be
    promoted to an EvidenceDocument without a further request.
    """

    provider: str
    external_id: str | None = None
    url: str | None = None
    title: str | None = None
    snippet: str | None = None
    authors: list[str] = field(default_factory=list)
    published_at: datetime | None = None
    source_type: SourceType = SourceType.WEB_PAGE
    source_family: SourceFamily = SourceFamily.WEB
    publisher: str | None = None
    doi: str | None = None
    language: str | None = None
    #: Provider-native relevance score, if any. Not comparable across providers.
    score: float | None = None
    #: Provider payload, retained so normalisation stays auditable.
    raw: dict[str, Any] = field(default_factory=dict)
    retrieved_at: datetime = field(default_factory=utcnow)

    @property
    def has_inline_document(self) -> bool:
        """True when the hit already carries enough content to be evidence."""
        return bool(self.raw.get("_inline_document"))
