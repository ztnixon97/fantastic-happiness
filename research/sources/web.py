"""General web search.

Web search providers differ only in their wire format, so they share one
adapter and contribute a parser. The research layer asks for 'a web search';
which provider answers is the registry's decision, made from configuration
and declared capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from research.errors import SourceNotConfigured, SourceUnavailable
from research.models.common import RetrievalMethod, SourceFamily, SourceType
from research.models.evidence import EvidenceDocument
from research.models.query import ResearchQuery, SearchHit
from research.normalize.html import parse_date
from research.normalize.text import normalize_whitespace
from research.normalize.urls import canonicalize_url, registrable_domain
from research.sources.base import SourceAdapter, SourceCapabilities
from research.sources.classify import classify_url
from research.sources.http import SafeHttpClient


class WebSearchAdapter(SourceAdapter, ABC):
    """Common behaviour for keyed web-search providers."""

    family = SourceFamily.WEB
    source_type = SourceType.WEB_PAGE
    base_url: str = ""
    requires_api_key: bool = True

    def __init__(
        self,
        client: SafeHttpClient,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.client = client
        self._api_key = api_key
        self.base_url = (base_url or self.base_url).rstrip("/")

    @property
    def configured(self) -> bool:
        return bool(self._api_key) or not self.requires_api_key

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(SourceFamily.WEB, SourceFamily.NEWS),
            default_source_type=SourceType.WEB_PAGE,
            supports_fetch=False,
            returns_inline_documents=False,
            supports_date_filter=True,
            requires_api_key=self.requires_api_key,
            max_results_per_query=20,
            notes="Discovery only; bodies come from the direct fetcher.",
        )

    def _require_key(self) -> str:
        if not self._api_key:
            raise SourceNotConfigured(
                f"{self.name} needs an API key; set its entry in the credential "
                "allowlist to enable it",
                provider=self.name,
            )
        return self._api_key

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        """Web search results are pointers.

        Their snippets are written by the search engine, not the publisher,
        so they are never promoted to evidence on their own: the body is
        retrieved by the direct fetcher.
        """
        raise SourceUnavailable(
            f"{self.name} returns result pointers only; fetch the URL instead",
            provider=self.name,
            retryable=False,
        )

    def _hit(
        self,
        *,
        url: str | None,
        title: str | None,
        snippet: str | None,
        published_at: Any = None,
        publisher: str | None = None,
        is_news: bool = False,
        raw: dict[str, Any] | None = None,
    ) -> SearchHit | None:
        canonical = canonicalize_url(url)
        if not canonical:
            return None
        payload = dict(raw or {})
        payload.setdefault("_endpoint", self.base_url)
        payload.setdefault("_retrieval_method", str(RetrievalMethod.SEARCH))
        payload.setdefault("search_snippet_is_provider_generated", True)
        return SearchHit(
            provider=self.name,
            url=canonical,
            title=normalize_whitespace(title) or None if title else None,
            snippet=normalize_whitespace(snippet) or None if snippet else None,
            published_at=(
                published_at if hasattr(published_at, "year") else parse_date(published_at)
            ),
            source_type=classify_url(canonical, title=title, is_news_result=is_news),
            source_family=SourceFamily.NEWS if is_news else SourceFamily.WEB,
            publisher=publisher or registrable_domain(canonical),
            raw=payload,
        )

    @abstractmethod
    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        ...


class BraveSearchSource(WebSearchAdapter):
    name = "brave"
    base_url = "https://api.search.brave.com/res/v1"

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        headers = {"X-Subscription-Token": self._require_key(), "Accept": "application/json"}
        params: dict[str, Any] = {"q": query.text, "count": min(query.limit, 20)}
        if query.language:
            params["search_lang"] = query.language
        is_news = SourceFamily.NEWS in query.families
        endpoint = f"{self.base_url}/{'news' if is_news else 'web'}/search"
        if is_news and query.filters.get("freshness"):
            params["freshness"] = query.filters["freshness"]

        payload = await self.client.get_json(
            endpoint, provider=self.name, params=params, headers=headers
        )
        container = payload.get("news" if is_news else "web") or {}
        results = container.get("results") or payload.get("results") or []
        hits = []
        for result in results:
            if not isinstance(result, dict):
                continue
            hit = self._hit(
                url=result.get("url"),
                title=result.get("title"),
                snippet=result.get("description"),
                published_at=result.get("page_age") or result.get("age"),
                publisher=(result.get("meta_url") or {}).get("netloc"),
                is_news=is_news,
                raw={"_endpoint": endpoint, "provider_result": True},
            )
            if hit:
                hits.append(hit)
        return hits


class TavilySearchSource(WebSearchAdapter):
    name = "tavily"
    base_url = "https://api.tavily.com"

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        is_news = SourceFamily.NEWS in query.families
        body: dict[str, Any] = {
            "api_key": self._require_key(),
            "query": query.text,
            "max_results": min(query.limit, 20),
            "search_depth": query.filters.get("depth", "basic"),
            "topic": "news" if is_news else "general",
        }
        if query.filters.get("include_domains"):
            body["include_domains"] = list(query.filters["include_domains"])
        endpoint = f"{self.base_url}/search"
        response = await self.client.request(
            "POST", endpoint, provider=self.name, json_body=body, accept="application/json"
        )
        payload = response.json()
        hits = []
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            hit = self._hit(
                url=result.get("url"),
                title=result.get("title"),
                snippet=result.get("content"),
                published_at=result.get("published_date"),
                is_news=is_news,
                raw={"_endpoint": endpoint, "score": result.get("score")},
            )
            if hit:
                hits.append(hit)
        return hits


class SerperSearchSource(WebSearchAdapter):
    name = "serper"
    base_url = "https://google.serper.dev"

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        is_news = SourceFamily.NEWS in query.families
        endpoint = f"{self.base_url}/{'news' if is_news else 'search'}"
        response = await self.client.request(
            "POST",
            endpoint,
            provider=self.name,
            headers={"X-API-KEY": self._require_key()},
            json_body={"q": query.text, "num": min(query.limit, 20)},
            accept="application/json",
        )
        payload = response.json()
        results = payload.get("news" if is_news else "organic") or []
        hits = []
        for result in results:
            if not isinstance(result, dict):
                continue
            hit = self._hit(
                url=result.get("link"),
                title=result.get("title"),
                snippet=result.get("snippet"),
                published_at=result.get("date"),
                publisher=result.get("source"),
                is_news=is_news,
                raw={"_endpoint": endpoint, "position": result.get("position")},
            )
            if hit:
                hits.append(hit)
        return hits
