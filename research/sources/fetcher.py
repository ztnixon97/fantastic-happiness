"""Direct URL fetching.

The one adapter that retrieves an arbitrary address. Everything it returns
is external, untrusted content: it is parsed for text and metadata, never
executed, and the safety rules live in :class:`SafeHttpClient` rather than in
any instruction given to a model.
"""

from __future__ import annotations

import json
from typing import Any

from research.normalize.document import document_from_page
from research.models.common import (
    Provenance,
    RetrievalMethod,
    SourceFamily,
    SourceType,
)
from research.models.evidence import EvidenceDocument
from research.models.query import ResearchQuery, SearchHit
from research.normalize.html import ExtractedPage, extract_page
from research.normalize.text import clean_text
from research.normalize.urls import canonicalize_url
from research.sources.base import SourceAdapter, SourceCapabilities
from research.sources.classify import classify_url
from research.sources.http import HttpResponse, SafeHttpClient


class DirectFetchSource(SourceAdapter):
    """Fetches a specific URL and normalises it into evidence."""

    name = "fetch"
    family = SourceFamily.WEB
    source_type = SourceType.WEB_PAGE

    def __init__(
        self,
        client: SafeHttpClient,
        *,
        max_text_characters: int = 400_000,
    ) -> None:
        self.client = client
        self.max_text_characters = max_text_characters

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(
                SourceFamily.WEB,
                SourceFamily.NEWS,
                SourceFamily.GOVERNMENT,
                SourceFamily.CORPORATE,
            ),
            default_source_type=SourceType.WEB_PAGE,
            supports_search=False,
            supports_fetch=True,
            supports_full_text=True,
            requires_api_key=False,
            notes="Retrieves one known URL. Has no discovery of its own.",
        )

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        """A URL is the only 'query' this source understands."""
        url = canonicalize_url(query.text.strip())
        if not url or " " in query.text.strip():
            return []
        return [
            SearchHit(
                provider=self.name,
                url=url,
                title=None,
                source_type=classify_url(url),
                source_family=SourceFamily.WEB,
            )
        ]

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        if not hit.url:
            raise ValueError("DirectFetchSource.fetch requires a hit with a URL")
        return await self.fetch_url(
            hit.url,
            source_family=hit.source_family,
            source_type=hit.source_type,
            published_at_hint=hit.published_at,
            publisher_hint=hit.publisher,
        )

    async def fetch_url(
        self,
        url: str,
        *,
        source_family: SourceFamily = SourceFamily.WEB,
        source_type: SourceType | None = None,
        parent_document_id: str | None = None,
        retrieval_method: RetrievalMethod = RetrievalMethod.DIRECT_FETCH,
        fetch_id: str | None = None,
        task_id: str | None = None,
        published_at_hint: Any = None,
        publisher_hint: str | None = None,
    ) -> EvidenceDocument:
        response = await self.client.request("GET", url, provider=self.name)
        page = self._parse(response)

        provenance = Provenance(
            provider=self.name,
            retrieval_method=retrieval_method,
            provider_endpoint=response.final_url,
            requested_url=url,
            final_url=response.final_url if response.final_url != url else None,
            parent_document_id=parent_document_id,
            fetch_id=fetch_id,
            task_id=task_id,
        )
        document = document_from_page(
            page,
            url=response.final_url,
            provider=self.name,
            provenance=provenance,
            source_type=source_type
            or classify_url(
                response.final_url,
                title=page.title,
                is_news_result=source_family is SourceFamily.NEWS,
            ),
            source_family=source_family,
            max_text_characters=self.max_text_characters,
        )
        if document.published_at is None and published_at_hint is not None:
            document.published_at = published_at_hint
            document.metadata["published_at_source"] = "search result metadata"
        if document.publisher is None and publisher_hint:
            document.publisher = publisher_hint
        document.metadata["http_status"] = response.status_code
        document.metadata["content_type"] = response.content_type
        if response.truncated:
            # Recorded, not hidden: a truncated document must not be quoted as
            # if it were complete.
            document.metadata["truncated"] = True
        return document

    def _parse(self, response: HttpResponse) -> ExtractedPage:
        content_type = (response.content_type or "").split(";", 1)[0].strip().lower()
        if content_type in ("text/html", "application/xhtml+xml"):
            return extract_page(response.text, max_characters=self.max_text_characters)
        if content_type == "application/json":
            return self._page_from_json(response)
        return ExtractedPage(text=clean_text(response.text)[: self.max_text_characters])

    def _page_from_json(self, response: HttpResponse) -> ExtractedPage:
        """Render a JSON body as readable text.

        Government and corporate endpoints often answer with JSON. It is kept
        as data - pretty-printed and truncated - never interpreted.
        """
        try:
            payload = json.loads(response.text)
        except ValueError:
            return ExtractedPage(text=clean_text(response.text)[: self.max_text_characters])
        pretty = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        title = None
        if isinstance(payload, dict):
            for key in ("title", "name", "headline"):
                if isinstance(payload.get(key), str):
                    title = payload[key]
                    break
        return ExtractedPage(title=title, text=pretty[: self.max_text_characters])
