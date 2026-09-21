"""News-oriented search.

'Search current news' is a research intent, not a provider. It is served
here either by GDELT - which is keyless and therefore always available - or
by any configured web-search provider re-shaped for news.

News results carry an important caveat that academic results do not: a
result is not evidence that its outlet did the reporting. Everything here
labels results as secondary until the news researcher establishes otherwise,
and deduplication decides what counts as one source.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from research.errors import SourceUnavailable
from research.models.common import RetrievalMethod, SourceFamily, SourceType
from research.models.evidence import EvidenceDocument
from research.models.query import ResearchQuery, SearchHit
from research.normalize.urls import canonicalize_url, registrable_domain
from research.sources.base import SourceAdapter, SourceCapabilities
from research.sources.classify import classify_url
from research.sources.http import SafeHttpClient


class GdeltNewsSource(SourceAdapter):
    """GDELT's article index: broad, multilingual, no credentials required."""

    name = "gdelt"
    family = SourceFamily.NEWS
    source_type = SourceType.SECONDARY_NEWS_REPORTING
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, client: SafeHttpClient, *, base_url: str | None = None) -> None:
        self.client = client
        self.base_url = (base_url or self.base_url).rstrip("/")

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(SourceFamily.NEWS,),
            default_source_type=SourceType.SECONDARY_NEWS_REPORTING,
            supports_fetch=False,
            returns_inline_documents=False,
            supports_date_filter=True,
            requires_api_key=False,
            max_results_per_query=75,
            tags=frozenset({"keyless", "news-index"}),
            notes="Global news index. Returns pointers; bodies come from the fetcher.",
        )

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        params: dict[str, Any] = {
            "query": self._build_query(query),
            "mode": "ArtList",
            "format": "json",
            "maxrecords": min(max(query.limit, 1), 75),
            "sort": "hybridrel",
        }
        if query.published_after or query.published_before:
            params["startdatetime"] = _stamp(query.published_after, "20000101000000")
            params["enddatetime"] = _stamp(query.published_before, _now_stamp())
        else:
            params["timespan"] = query.filters.get("timespan", "3m")

        payload = await self.client.get_json(
            self.base_url, provider=self.name, params=params
        )
        if not isinstance(payload, dict):
            raise SourceUnavailable("unexpected GDELT payload", provider=self.name)

        hits: list[SearchHit] = []
        for article in payload.get("articles", []):
            if not isinstance(article, dict):
                continue
            url = canonicalize_url(article.get("url"))
            if not url:
                continue
            hits.append(
                SearchHit(
                    provider=self.name,
                    url=url,
                    title=article.get("title"),
                    published_at=_parse_seendate(article.get("seendate")),
                    source_type=classify_url(url, title=article.get("title"), is_news_result=True),
                    source_family=SourceFamily.NEWS,
                    publisher=article.get("domain") or registrable_domain(url),
                    language=(article.get("language") or "").lower() or None,
                    raw={
                        "domain": article.get("domain"),
                        "source_country": article.get("sourcecountry"),
                        # GDELT reports when it *saw* the article, which can
                        # postdate publication; the fetcher usually recovers
                        # the publisher's own date and overrides this.
                        "seendate_is_index_time": True,
                        "_endpoint": self.base_url,
                        "_retrieval_method": str(RetrievalMethod.SEARCH),
                    },
                )
            )
        return hits

    def _build_query(self, query: ResearchQuery) -> str:
        parts = [query.text]
        if query.language:
            parts.append(f"sourcelang:{query.language}")
        domains = query.filters.get("include_domains") or []
        if len(domains) == 1:
            parts.append(f"domain:{domains[0]}")
        return " ".join(parts)

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        raise SourceUnavailable(
            "GDELT returns article pointers only; fetch the URL instead",
            provider=self.name,
            retryable=False,
        )


class NewsSearchSource(SourceAdapter):
    """Re-shapes any web-search provider into a news search.

    The planner asks for news; this decides what that means for the provider
    underneath - the news endpoint where one exists, a freshness window where
    it does not.
    """

    family = SourceFamily.NEWS
    source_type = SourceType.SECONDARY_NEWS_REPORTING

    def __init__(self, delegate: SourceAdapter, *, default_timespan: str = "1y") -> None:
        self.delegate = delegate
        self.default_timespan = default_timespan
        self.name = f"news:{delegate.name}"

    def capabilities(self) -> SourceCapabilities:
        underlying = self.delegate.capabilities()
        return SourceCapabilities(
            name=self.name,
            families=(SourceFamily.NEWS,),
            default_source_type=SourceType.SECONDARY_NEWS_REPORTING,
            supports_fetch=False,
            returns_inline_documents=underlying.returns_inline_documents,
            supports_date_filter=underlying.supports_date_filter,
            requires_api_key=underlying.requires_api_key,
            max_results_per_query=underlying.max_results_per_query,
            tags=underlying.tags | {"news-shaped"},
            notes=f"News-shaped queries over {underlying.name}.",
        )

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        news_query = ResearchQuery(
            text=query.text,
            families=[SourceFamily.NEWS],
            limit=query.limit,
            published_after=query.published_after,
            published_before=query.published_before,
            language=query.language,
            filters={"freshness": self.default_timespan, **query.filters},
            objective=query.objective,
            investigation_id=query.investigation_id,
            task_id=query.task_id,
        )
        hits = await self.delegate.search(news_query)
        for hit in hits:
            hit.source_family = SourceFamily.NEWS
            if hit.source_type is SourceType.WEB_PAGE:
                hit.source_type = SourceType.SECONDARY_NEWS_REPORTING
        return hits

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        return await self.delegate.fetch(hit)


def _parse_seendate(value: Any) -> datetime | None:
    if not isinstance(value, str) or len(value) < 15:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _stamp(value: Any, default: str) -> str:
    return value.strftime("%Y%m%d%H%M%S") if value else default


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
