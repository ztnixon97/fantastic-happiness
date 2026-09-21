"""Source adapters behind one provider-independent interface."""

from research.sources.arxiv import ArxivSource
from research.sources.base import (
    CitationSource,
    ResearchSource,
    SourceAdapter,
    SourceCapabilities,
)
from research.sources.crossref import CrossrefSource
from research.sources.fetcher import DirectFetchSource
from research.sources.http import HttpResponse, SafeHttpClient
from research.sources.news import GdeltNewsSource, NewsSearchSource
from research.sources.offline import build_offline_registry, load_corpus
from research.sources.openalex import OpenAlexSource
from research.sources.registry import SourceRegistry, build_registry
from research.sources.semantic_scholar import SemanticScholarSource
from research.sources.social import BlueskySource, MastodonSource, YouTubeSource
from research.sources.static import StaticAcademicSource, StaticWebSource
from research.sources.web import (
    BraveSearchSource,
    SerperSearchSource,
    TavilySearchSource,
)

__all__ = [
    "ArxivSource",
    "BlueskySource",
    "BraveSearchSource",
    "CitationSource",
    "CrossrefSource",
    "DirectFetchSource",
    "GdeltNewsSource",
    "MastodonSource",
    "HttpResponse",
    "NewsSearchSource",
    "OpenAlexSource",
    "build_offline_registry",
    "load_corpus",
    "ResearchSource",
    "SafeHttpClient",
    "SemanticScholarSource",
    "SerperSearchSource",
    "SourceAdapter",
    "SourceCapabilities",
    "SourceRegistry",
    "StaticAcademicSource",
    "StaticWebSource",
    "TavilySearchSource",
    "YouTubeSource",
    "build_registry",
]
