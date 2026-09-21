"""Provider routing.

The layer that turns 'search academic literature' into 'these three adapters,
in this order'. Everything above this line reasons about source families and
capabilities; everything below knows about endpoints.

Selection rules, in order:

1. the provider must be enabled in configuration,
2. it must serve the requested family,
3. it must have whatever credentials it declares it needs,
4. keyless providers come first, so an investigation degrades to a smaller
   set of sources rather than failing when a key is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from research.config import ResearchConfig
from research.normalize.docling_reader import converter_for
from research.models.common import SourceFamily
from research.sources.arxiv import ArxivSource
from research.sources.base import CitationSource, ResearchSource, SourceCapabilities
from research.sources.crossref import CrossrefSource
from research.sources.fetcher import DirectFetchSource
from research.sources.http import SafeHttpClient
from research.sources.news import GdeltNewsSource, NewsSearchSource
from research.sources.openalex import OpenAlexSource
from research.sources.semantic_scholar import SemanticScholarSource
from research.sources.social import BlueskySource, MastodonSource, YouTubeSource
from research.sources.web import (
    BraveSearchSource,
    SerperSearchSource,
    TavilySearchSource,
    WebSearchAdapter,
)


@dataclass(slots=True)
class SourceRegistry:
    """The set of sources available to an investigation."""

    sources: dict[str, ResearchSource] = field(default_factory=dict)
    fetcher: DirectFetchSource | None = None

    def register(self, source: ResearchSource) -> None:
        self.sources[source.name] = source

    def get(self, name: str) -> ResearchSource | None:
        return self.sources.get(name)

    def all(self) -> list[ResearchSource]:
        return list(self.sources.values())

    def capabilities(self) -> list[SourceCapabilities]:
        return [source.capabilities() for source in self.sources.values()]

    def for_family(self, family: SourceFamily) -> list[ResearchSource]:
        """Sources that can search this family, keyless ones first."""
        matches = [
            source
            for source in self.sources.values()
            if source.capabilities().supports_search
            and source.capabilities().serves(family)
        ]
        return sorted(matches, key=_preference_key)

    def citation_sources(self) -> list[ResearchSource]:
        """Sources that can walk the citation graph in at least one direction."""
        return sorted(
            (
                source
                for source in self.sources.values()
                if isinstance(source, CitationSource)
                and (
                    source.capabilities().supports_references
                    or source.capabilities().supports_citing_papers
                )
            ),
            key=_preference_key,
        )

    def doi_resolvers(self) -> list[ResearchSource]:
        """Sources that can turn a bare DOI into a record."""
        return sorted(
            (source for source in self.sources.values() if hasattr(source, "lookup_doi")),
            key=_preference_key,
        )

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for source in self.sources.values():
            capability = source.capabilities()
            rows.append(
                {
                    "name": capability.name,
                    "families": [str(family) for family in capability.families],
                    "search": capability.supports_search,
                    "fetch": capability.supports_fetch,
                    "references": capability.supports_references,
                    "citing": capability.supports_citing_papers,
                    "needs_key": capability.requires_api_key,
                    "notes": capability.notes,
                }
            )
        return sorted(rows, key=lambda row: row["name"])


def _preference_key(source: ResearchSource) -> tuple[int, str]:
    capability = source.capabilities()
    return (1 if capability.requires_api_key else 0, capability.name)


#: Keyed web providers, in the order they are tried when several are configured.
_WEB_PROVIDERS: tuple[tuple[str, type[WebSearchAdapter]], ...] = (
    ("brave", BraveSearchSource),
    ("tavily", TavilySearchSource),
    ("serper", SerperSearchSource),
)


def build_registry(
    config: ResearchConfig,
    client: SafeHttpClient,
    *,
    extra_sources: Iterable[ResearchSource] = (),
) -> SourceRegistry:
    """Assemble the sources this configuration permits.

    Credentials are read from the config's allowlist and handed to the one
    adapter that needs them; no adapter reaches into the environment itself.
    """
    registry = SourceRegistry()

    academic: list[type] = [OpenAlexSource, CrossrefSource, ArxivSource, SemanticScholarSource]
    for source_class in academic:
        name = source_class.name
        if not config.is_enabled(name):
            continue
        settings = config.provider(name)
        registry.register(
            source_class(
                client,
                base_url=settings.base_url,
                contact_email=config.contact_email,
                api_key=config.secret(name),
            )
        )

    if config.is_enabled("gdelt"):
        gdelt = GdeltNewsSource(client, base_url=config.provider("gdelt").base_url)
        registry.register(gdelt)

    for name, source_class in _WEB_PROVIDERS:
        if not config.is_enabled(name):
            continue
        api_key = config.secret(name)
        if not api_key:
            # Declared but unconfigured providers are simply absent; a research
            # run reports which sources it had rather than failing late.
            continue
        web_source = source_class(
            client, api_key=api_key, base_url=config.provider(name).base_url
        )
        registry.register(web_source)
        registry.register(NewsSearchSource(web_source))

    # Public social sources. Keyless ones are always available; YouTube is
    # skipped without a key, like any other keyed provider.
    if config.is_enabled("bluesky"):
        registry.register(BlueskySource(client, base_url=config.provider("bluesky").base_url))
    if config.is_enabled("mastodon"):
        registry.register(
            MastodonSource(
                client,
                base_url=config.provider("mastodon").base_url,
                api_key=config.secret("mastodon"),
            )
        )
    if config.is_enabled("youtube") and config.secret("youtube"):
        registry.register(
            YouTubeSource(
                client,
                api_key=config.secret("youtube"),
                base_url=config.provider("youtube").base_url,
            )
        )

    if config.is_enabled("fetch"):
        fetcher = DirectFetchSource(
            client,
            max_text_characters=config.acquisition.max_text_characters,
            docling=converter_for(config.ingest),
            ocr=config.ingest.ocr,
        )
        registry.register(fetcher)
        registry.fetcher = fetcher

    for source in extra_sources:
        registry.register(source)
        if isinstance(source, DirectFetchSource) and registry.fetcher is None:
            registry.fetcher = source

    return registry
