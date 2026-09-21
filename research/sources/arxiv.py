"""arXiv adapter.

arXiv is where much of the relevant technical literature appears first. Its
records are explicitly preprints: the adapter labels them as such, and never
as peer-reviewed, even when a published DOI is attached.
"""

from __future__ import annotations

from typing import Any
from xml.etree import ElementTree

from research.errors import SourceUnavailable
from research.models.common import RetrievalMethod, SourceFamily, SourceType
from research.models.query import ResearchQuery, SearchHit
from research.normalize.doi import normalize_arxiv_id, normalize_doi
from research.normalize.text import normalize_whitespace
from research.sources.academic import AcademicAdapter, parse_iso_date
from research.sources.base import SourceCapabilities

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_MAX_RESULTS = 50


class ArxivSource(AcademicAdapter):
    name = "arxiv"
    source_type = SourceType.ACADEMIC_PREPRINT
    base_url = "https://export.arxiv.org/api"

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(SourceFamily.ACADEMIC,),
            default_source_type=SourceType.ACADEMIC_PREPRINT,
            returns_inline_documents=True,
            supports_date_filter=True,
            supports_references=False,
            supports_citing_papers=False,
            requires_api_key=False,
            max_results_per_query=_MAX_RESULTS,
            tags=frozenset({"keyless", "preprint"}),
            notes="Preprints. No citation graph; pair with OpenAlex for that.",
        )

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        # arXiv treats a quoted string as an exact phrase, which almost never
        # matches a research question, and ANDing every term over-restricts a
        # long one. The unquoted term list is what actually works; a caller
        # that really wants a phrase asks for one.
        phrase = bool(query.filters.get("phrase"))
        search_query = f'all:"{query.text}"' if phrase else f"all:{query.text}"
        if query.published_after or query.published_before:
            start = _stamp(query.published_after, "190001010000")
            end = _stamp(query.published_before, "299901010000")
            search_query = f"({search_query}) AND submittedDate:[{start} TO {end}]"

        endpoint = f"{self.base_url}/query"
        response = await self.client.request(
            "GET",
            endpoint,
            provider=self.name,
            params={
                "search_query": search_query,
                "start": 0,
                "max_results": min(query.limit, _MAX_RESULTS),
                "sortBy": "relevance",
            },
            accept="application/atom+xml",
        )
        return self._parse_feed(response.text, endpoint=endpoint)

    def _parse_feed(self, xml_text: str, *, endpoint: str) -> list[SearchHit]:
        try:
            # ElementTree does not resolve external entities, so parsing an
            # untrusted feed cannot reach the filesystem or the network.
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as exc:
            raise SourceUnavailable(
                f"malformed arXiv feed: {exc}", provider=self.name
            ) from exc
        return [
            hit
            for hit in (
                self._to_hit(entry, endpoint=endpoint)
                for entry in root.findall(f"{_ATOM}entry")
            )
            if hit is not None
        ]

    def _to_hit(self, entry: ElementTree.Element, *, endpoint: str) -> SearchHit | None:
        raw_id = _text(entry.find(f"{_ATOM}id"))
        arxiv_id = normalize_arxiv_id(raw_id)
        if not arxiv_id:
            return None
        doi = normalize_doi(_text(entry.find(f"{_ARXIV}doi")))
        categories = [
            element.get("term")
            for element in entry.findall(f"{_ATOM}category")
            if element.get("term")
        ]
        pdf_url = None
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
        comment = _text(entry.find(f"{_ARXIV}comment"))

        return SearchHit(
            provider=self.name,
            external_id=arxiv_id,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            title=normalize_whitespace(_text(entry.find(f"{_ATOM}title")) or "") or None,
            snippet=normalize_whitespace(_text(entry.find(f"{_ATOM}summary")) or "") or None,
            authors=[
                normalize_whitespace(_text(author.find(f"{_ATOM}name")) or "")
                for author in entry.findall(f"{_ATOM}author")
            ][:50],
            published_at=parse_iso_date(_text(entry.find(f"{_ATOM}published"))),
            source_type=SourceType.ACADEMIC_PREPRINT,
            source_family=SourceFamily.ACADEMIC,
            publisher="arXiv",
            doi=doi,
            raw={
                "arxiv_id": arxiv_id,
                "categories": categories,
                "primary_category": categories[0] if categories else None,
                "pdf_url": pdf_url,
                "updated_at": _text(entry.find(f"{_ATOM}updated")),
                "comment": comment,
                "journal_ref": _text(entry.find(f"{_ARXIV}journal_ref")),
                "_endpoint": endpoint,
                "_retrieval_method": str(RetrievalMethod.SEARCH),
            },
        )

    async def lookup_id(self, arxiv_id: str) -> SearchHit | None:
        normalized = normalize_arxiv_id(arxiv_id)
        if not normalized:
            return None
        endpoint = f"{self.base_url}/query"
        response = await self.client.request(
            "GET",
            endpoint,
            provider=self.name,
            params={"id_list": normalized, "max_results": 1},
            accept="application/atom+xml",
        )
        hits = self._parse_feed(response.text, endpoint=endpoint)
        return hits[0] if hits else None


def _text(element: ElementTree.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    return element.text.strip() or None


def _stamp(value: Any, default: str) -> str:
    return value.strftime("%Y%m%d%H%M") if value else default
