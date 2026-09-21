"""The normalised representation of externally retrieved material.

Everything the system retrieves - a paper, a filing, an article, a page, a
transcript - becomes an :class:`EvidenceDocument`. Nothing a model writes
ever becomes one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator

from research.models.common import (
    PRIMARY_SOURCE_DISTANCE,
    DuplicateRelation,
    Provenance,
    SourceFamily,
    SourceType,
    utcnow,
)

#: Identity schemes used for deduplication, in descending order of authority.
#: A match on an earlier scheme is a stronger statement than a later one.
IDENTITY_SCHEMES = (
    "doi",
    "arxiv",
    "pmid",
    "isbn",
    "provider_id",
    "canonical_url",
    "content_hash",
    "title_key",
)


@dataclass(slots=True)
class EvidenceDocument:
    """A single retrieved artifact, normalised across providers.

    The text is *evidence*, not instruction. Callers that render ``text`` or
    ``abstract`` into a model prompt must label it as untrusted external
    material (see :mod:`research.acquisition.untrusted`).
    """

    id: str
    source_type: SourceType
    provider: str
    investigation_id: str | None = None
    source_family: SourceFamily = SourceFamily.WEB
    external_id: str | None = None
    canonical_url: str | None = None
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    published_at: datetime | None = None
    fetched_at: datetime = field(default_factory=utcnow)
    text: str | None = None
    abstract: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    provenance: Provenance | None = None

    #: Set when this document is not independent of another one.
    duplicate_of: str | None = None
    duplicate_relation: DuplicateRelation | None = None
    #: Set when this document was produced *from* another document, e.g. a
    #: transcript derived from a video, or an article written from a filing.
    derived_from: str | None = None

    # --- identity / independence helpers, populated by normalisation --------
    doi: str | None = None
    #: Host of the canonical URL, used for syndication reasoning.
    canonical_host: str | None = None
    #: Aggressively normalised title, used as a weak identity key.
    title_key: str | None = None
    #: 64-bit similarity fingerprint of the body text (near-duplicate search).
    simhash: int | None = None
    #: Cluster root: documents sharing this key are *one* piece of evidence.
    independence_key: str | None = None
    #: Publisher/outlet/journal, when known.
    publisher: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if self.independence_key is None:
            self.independence_key = self.id

    # ------------------------------------------------------------------
    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of is not None

    @property
    def primary_source_distance(self) -> int:
        """How far this document sits from an underlying record (lower = closer)."""
        return PRIMARY_SOURCE_DISTANCE.get(self.source_type, 4)

    @property
    def best_text(self) -> str:
        """Full text when available, otherwise the abstract, otherwise title."""
        return self.text or self.abstract or self.title or ""

    def identity_keys(self) -> Iterator[tuple[str, str]]:
        """Yield ``(scheme, value)`` pairs usable for exact deduplication.

        Ordered by authority so the first match wins.
        """
        if self.doi:
            yield "doi", self.doi
        arxiv_id = self.metadata.get("arxiv_id")
        if arxiv_id:
            yield "arxiv", str(arxiv_id)
        pmid = self.metadata.get("pmid")
        if pmid:
            yield "pmid", str(pmid)
        if self.external_id:
            yield "provider_id", f"{self.provider}:{self.external_id}"
        if self.canonical_url:
            yield "canonical_url", self.canonical_url
        if self.content_hash:
            yield "content_hash", self.content_hash
        if self.title_key:
            yield "title_key", self.title_key

    def citation_label(self) -> str:
        """Compact human label, e.g. ``Smith et al. (2024), Nature``."""
        who = ""
        if self.authors:
            who = self.authors[0]
            if len(self.authors) > 2:
                who += " et al."
            elif len(self.authors) == 2:
                who += f" and {self.authors[1]}"
        when = f"({self.published_at.year})" if self.published_at else "(n.d.)"
        where = self.publisher or self.canonical_host or self.provider
        return " ".join(part for part in (who, when, where) if part).strip()
