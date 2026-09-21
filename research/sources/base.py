"""The provider-independent source interface.

The research layer asks for *kinds of material* - academic literature,
current news, a specific document, the works a paper cites. Which provider
answers is decided below this line, in :mod:`research.sources.registry`.

A source returns two things and nothing else:

``search``
    :class:`~research.models.query.SearchHit` pointers, cheap enough to
    triage before spending a fetch.
``fetch``
    A normalised :class:`~research.models.evidence.EvidenceDocument` whose
    ``id`` is empty: identity is assigned by the acquisition layer when the
    document is persisted, because only the store knows whether the material
    is new.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from research.models.common import SourceFamily, SourceType
from research.models.evidence import EvidenceDocument
from research.models.query import ResearchQuery, SearchHit


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """What a provider can actually do.

    The planner uses this to route intent without knowing provider details,
    and the registry uses it to skip providers that cannot serve a request
    (no key configured, no date filtering, no citation graph).
    """

    name: str
    families: tuple[SourceFamily, ...]
    default_source_type: SourceType = SourceType.WEB_PAGE
    supports_search: bool = True
    supports_fetch: bool = True
    #: Search results already carry body text or a usable abstract.
    returns_inline_documents: bool = False
    supports_full_text: bool = False
    supports_date_filter: bool = False
    supports_references: bool = False
    supports_citing_papers: bool = False
    requires_api_key: bool = False
    max_results_per_query: int = 50
    notes: str | None = None
    #: Free-form labels used by the registry, e.g. ``{"keyless", "preprint"}``.
    tags: frozenset[str] = field(default_factory=frozenset)

    def serves(self, family: SourceFamily) -> bool:
        return family in self.families


@runtime_checkable
class ResearchSource(Protocol):
    """Minimal contract every source adapter satisfies."""

    name: str

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        ...

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        ...

    def capabilities(self) -> SourceCapabilities:
        ...


@runtime_checkable
class CitationSource(Protocol):
    """A source that exposes a citation graph.

    Traversal limits are the caller's responsibility (see
    :mod:`research.graph.citations`); an adapter simply answers what one hop
    contains.
    """

    name: str

    async def get_references(self, hit: SearchHit, *, limit: int = 50) -> list[SearchHit]:
        """Works cited *by* this one (backward in time)."""
        ...

    async def get_citing_papers(self, hit: SearchHit, *, limit: int = 50) -> list[SearchHit]:
        """Works that cite this one (forward in time)."""
        ...


class SourceAdapter:
    """Small base class with the bookkeeping every adapter repeats."""

    name: str = "unnamed"
    family: SourceFamily = SourceFamily.WEB
    source_type: SourceType = SourceType.WEB_PAGE

    def capabilities(self) -> SourceCapabilities:  # pragma: no cover - overridden
        return SourceCapabilities(
            name=self.name,
            families=(self.family,),
            default_source_type=self.source_type,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"
