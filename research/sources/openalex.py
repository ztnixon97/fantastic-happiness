"""OpenAlex adapter.

OpenAlex is the backbone of the academic slice: keyless, broad coverage, and
it returns both directions of the citation graph, which is what makes
bounded traversal possible without a second provider.
"""

from __future__ import annotations

from typing import Any

from research.errors import SourceUnavailable
from research.models.common import RetrievalMethod, SourceFamily, SourceType
from research.models.query import ResearchQuery, SearchHit
from research.normalize.doi import normalize_arxiv_id, normalize_doi
from research.sources.academic import (
    AcademicAdapter,
    author_names,
    classify_work,
    inverted_index_to_text,
    parse_iso_date,
)
from research.sources.base import SourceCapabilities

_MAX_PER_PAGE = 50


def _short_id(value: Any) -> str | None:
    """``https://openalex.org/W2741809807`` -> ``W2741809807``."""
    if not value or not isinstance(value, str):
        return None
    return value.rstrip("/").rsplit("/", 1)[-1] or None


class OpenAlexSource(AcademicAdapter):
    name = "openalex"
    base_url = "https://api.openalex.org"

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(SourceFamily.ACADEMIC,),
            default_source_type=SourceType.ACADEMIC_PEER_REVIEWED,
            returns_inline_documents=True,
            supports_date_filter=True,
            supports_references=True,
            supports_citing_papers=True,
            requires_api_key=False,
            max_results_per_query=_MAX_PER_PAGE,
            tags=frozenset({"keyless", "citation-graph"}),
            notes="Open catalogue of scholarly works; both citation directions.",
        )

    # -- search ---------------------------------------------------------
    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        filters: list[str] = []
        if query.published_after:
            filters.append(f"from_publication_date:{query.published_after.date()}")
        if query.published_before:
            filters.append(f"to_publication_date:{query.published_before.date()}")
        if query.filters.get("open_access"):
            filters.append("is_oa:true")
        if query.filters.get("work_type"):
            filters.append(f"type:{query.filters['work_type']}")

        params: dict[str, Any] = {
            "search": query.text,
            "per-page": min(query.limit, _MAX_PER_PAGE),
        }
        if filters:
            params["filter"] = ",".join(filters)
        if self.contact_email:
            # OpenAlex asks for a contact address and gives the polite pool in
            # return: better rate limits, fewer failures mid-investigation.
            params["mailto"] = self.contact_email

        payload = await self.client.get_json(
            f"{self.base_url}/works", provider=self.name, params=params
        )
        return self._hits(payload, endpoint=f"{self.base_url}/works")

    def _hits(
        self,
        payload: Any,
        *,
        endpoint: str,
        method: RetrievalMethod = RetrievalMethod.SEARCH,
    ) -> list[SearchHit]:
        if not isinstance(payload, dict):
            raise SourceUnavailable("unexpected OpenAlex payload", provider=self.name)
        return [
            self._to_hit(work, endpoint=endpoint, method=method)
            for work in payload.get("results", [])
            if isinstance(work, dict)
        ]

    def _to_hit(
        self, work: dict[str, Any], *, endpoint: str, method: RetrievalMethod
    ) -> SearchHit:
        openalex_id = _short_id(work.get("id"))
        location = work.get("primary_location") or {}
        venue = ((location.get("source") or {}).get("display_name")) or None
        doi = normalize_doi(work.get("doi"))
        ids = work.get("ids") or {}
        abstract = inverted_index_to_text(work.get("abstract_inverted_index"))
        landing = location.get("landing_page_url") or work.get("doi")

        raw: dict[str, Any] = {
            "openalex_id": openalex_id,
            "venue": venue,
            "work_type": work.get("type"),
            "cited_by_count": work.get("cited_by_count"),
            "referenced_works": [
                _short_id(ref) for ref in (work.get("referenced_works") or [])
            ][:200],
            "is_retracted": work.get("is_retracted"),
            "is_open_access": (work.get("open_access") or {}).get("is_oa"),
            "_endpoint": endpoint,
            "_retrieval_method": str(method),
        }
        if ids.get("pmid"):
            raw["pmid"] = _short_id(ids["pmid"])
        arxiv_id = normalize_arxiv_id(landing) or normalize_arxiv_id(doi)
        if arxiv_id:
            raw["arxiv_id"] = arxiv_id
        if work.get("is_retracted"):
            # Retraction is the single most decision-relevant fact about a
            # paper; it travels with the hit so the skeptic never has to
            # re-derive it.
            raw["retraction_notice"] = "OpenAlex reports this work as retracted"

        return SearchHit(
            provider=self.name,
            external_id=openalex_id,
            url=landing or (f"https://doi.org/{doi}" if doi else None),
            title=work.get("display_name") or work.get("title"),
            snippet=abstract,
            authors=author_names(work.get("authorships") or []),
            published_at=parse_iso_date(work.get("publication_date"))
            or (
                parse_iso_date(str(work["publication_year"]))
                if work.get("publication_year")
                else None
            ),
            source_type=classify_work(
                work_type=work.get("type"), venue=venue, doi=doi, url=landing
            ),
            source_family=SourceFamily.ACADEMIC,
            publisher=venue,
            doi=doi,
            language=work.get("language"),
            score=work.get("relevance_score"),
            raw=raw,
        )

    # -- citation graph --------------------------------------------------
    async def get_references(self, hit: SearchHit, *, limit: int = 50) -> list[SearchHit]:
        """Works cited by this one, resolved in a single batched request."""
        referenced = [ref for ref in (hit.raw.get("referenced_works") or []) if ref]
        if not referenced and hit.external_id:
            work = await self._get_work(hit.external_id)
            referenced = [
                _short_id(ref) for ref in (work.get("referenced_works") or []) if ref
            ]
        referenced = referenced[:limit]
        if not referenced:
            return []
        endpoint = f"{self.base_url}/works"
        params = {
            "filter": f"openalex:{'|'.join(referenced)}",
            "per-page": min(len(referenced), _MAX_PER_PAGE),
        }
        if self.contact_email:
            params["mailto"] = self.contact_email
        payload = await self.client.get_json(endpoint, provider=self.name, params=params)
        return self._hits(
            payload, endpoint=endpoint, method=RetrievalMethod.REFERENCE_EXPANSION
        )

    async def get_citing_papers(self, hit: SearchHit, *, limit: int = 50) -> list[SearchHit]:
        """Works that cite this one - where replications and rebuttals live."""
        if not hit.external_id:
            return []
        endpoint = f"{self.base_url}/works"
        params: dict[str, Any] = {
            "filter": f"cites:{hit.external_id}",
            "per-page": min(limit, _MAX_PER_PAGE),
            "sort": "cited_by_count:desc",
        }
        if self.contact_email:
            params["mailto"] = self.contact_email
        payload = await self.client.get_json(endpoint, provider=self.name, params=params)
        return self._hits(
            payload, endpoint=endpoint, method=RetrievalMethod.CITATION_EXPANSION
        )

    async def _get_work(self, external_id: str) -> dict[str, Any]:
        params = {"mailto": self.contact_email} if self.contact_email else None
        payload = await self.client.get_json(
            f"{self.base_url}/works/{external_id}", provider=self.name, params=params
        )
        if not isinstance(payload, dict):
            raise SourceUnavailable("unexpected OpenAlex work payload", provider=self.name)
        return payload

    async def lookup_doi(self, doi: str) -> SearchHit | None:
        """Resolve a bare DOI to a hit - the primary-source chase for papers."""
        normalized = normalize_doi(doi)
        if not normalized:
            return None
        endpoint = f"{self.base_url}/works/doi:{normalized}"
        try:
            work = await self.client.get_json(endpoint, provider=self.name)
        except SourceUnavailable:
            raise
        except Exception:
            return None
        if not isinstance(work, dict) or not work.get("id"):
            return None
        return self._to_hit(
            work, endpoint=endpoint, method=RetrievalMethod.PRIMARY_SOURCE_CHASE
        )
