"""Semantic Scholar adapter.

Semantic Scholar carries both citation directions and, usefully for a
skeptic pass, marks the *intent* of a citation where it can. It works
without a key at a low rate limit; a key is used when one is configured.
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
    parse_iso_date,
)
from research.sources.base import SourceCapabilities

_FIELDS = (
    "paperId,externalIds,title,abstract,venue,year,publicationDate,authors,"
    "publicationTypes,openAccessPdf,citationCount,referenceCount,isOpenAccess,url"
)
_MAX_LIMIT = 100


class SemanticScholarSource(AcademicAdapter):
    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1"

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
            max_results_per_query=_MAX_LIMIT,
            tags=frozenset({"citation-graph", "optional-key"}),
            notes="Citation graph in both directions; API key raises rate limits.",
        )

    def _headers(self) -> dict[str, str]:
        # The key is handed to this adapter explicitly by configuration; the
        # client never reads the environment on its own.
        return {"x-api-key": self._api_key} if self._api_key else {}

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        params: dict[str, Any] = {
            "query": query.text,
            "limit": min(query.limit, _MAX_LIMIT),
            "fields": _FIELDS,
        }
        if query.published_after or query.published_before:
            start = query.published_after.year if query.published_after else ""
            end = query.published_before.year if query.published_before else ""
            params["year"] = f"{start}-{end}"
        if query.filters.get("open_access"):
            params["openAccessPdf"] = ""

        endpoint = f"{self.base_url}/paper/search"
        payload = await self.client.get_json(
            endpoint, provider=self.name, params=params, headers=self._headers()
        )
        if not isinstance(payload, dict):
            raise SourceUnavailable("unexpected Semantic Scholar payload", provider=self.name)
        return [
            self._to_hit(paper, endpoint=endpoint, method=RetrievalMethod.SEARCH)
            for paper in payload.get("data", [])
            if isinstance(paper, dict)
        ]

    def _to_hit(
        self, paper: dict[str, Any], *, endpoint: str, method: RetrievalMethod
    ) -> SearchHit:
        external = paper.get("externalIds") or {}
        doi = normalize_doi(external.get("DOI"))
        arxiv_id = normalize_arxiv_id(external.get("ArXiv"))
        venue = paper.get("venue") or None
        publication_types = paper.get("publicationTypes") or []
        raw: dict[str, Any] = {
            "semantic_scholar_id": paper.get("paperId"),
            "venue": venue,
            "publication_types": publication_types,
            "cited_by_count": paper.get("citationCount"),
            "reference_count": paper.get("referenceCount"),
            "is_open_access": paper.get("isOpenAccess"),
            "open_access_pdf": (paper.get("openAccessPdf") or {}).get("url"),
            "_endpoint": endpoint,
            "_retrieval_method": str(method),
        }
        if arxiv_id:
            raw["arxiv_id"] = arxiv_id
        if external.get("PubMed"):
            raw["pmid"] = str(external["PubMed"])
        if "Review" in publication_types:
            # Reviews and meta-analyses are what the academic researcher wants
            # first when entering an unfamiliar literature.
            raw["is_review"] = True

        source_type = classify_work(
            work_type=",".join(publication_types) or None,
            venue=venue,
            doi=doi,
            url=paper.get("url"),
        )
        if arxiv_id and not venue:
            source_type = SourceType.ACADEMIC_PREPRINT

        return SearchHit(
            provider=self.name,
            external_id=paper.get("paperId"),
            url=paper.get("url") or (f"https://doi.org/{doi}" if doi else None),
            title=paper.get("title"),
            snippet=paper.get("abstract"),
            authors=author_names(paper.get("authors") or []),
            published_at=parse_iso_date(paper.get("publicationDate"))
            or (parse_iso_date(str(paper["year"])) if paper.get("year") else None),
            source_type=source_type,
            source_family=SourceFamily.ACADEMIC,
            publisher=venue,
            doi=doi,
            raw=raw,
        )

    async def get_references(self, hit: SearchHit, *, limit: int = 50) -> list[SearchHit]:
        return await self._graph_edge(hit, "references", "citedPaper", limit)

    async def get_citing_papers(self, hit: SearchHit, *, limit: int = 50) -> list[SearchHit]:
        return await self._graph_edge(hit, "citations", "citingPaper", limit)

    async def _graph_edge(
        self, hit: SearchHit, edge: str, key: str, limit: int
    ) -> list[SearchHit]:
        paper_id = hit.raw.get("semantic_scholar_id") or hit.external_id
        if not paper_id and hit.doi:
            paper_id = f"DOI:{hit.doi}"
        if not paper_id:
            return []
        endpoint = f"{self.base_url}/paper/{paper_id}/{edge}"
        payload = await self.client.get_json(
            endpoint,
            provider=self.name,
            params={"limit": min(limit, _MAX_LIMIT), "fields": _FIELDS + ",intents,isInfluential"},
            headers=self._headers(),
        )
        if not isinstance(payload, dict):
            raise SourceUnavailable("unexpected Semantic Scholar payload", provider=self.name)
        method = (
            RetrievalMethod.REFERENCE_EXPANSION
            if edge == "references"
            else RetrievalMethod.CITATION_EXPANSION
        )
        hits: list[SearchHit] = []
        for entry in payload.get("data", []):
            paper = entry.get(key) if isinstance(entry, dict) else None
            if not isinstance(paper, dict) or not paper.get("paperId"):
                continue
            citation_hit = self._to_hit(paper, endpoint=endpoint, method=method)
            if entry.get("intents"):
                citation_hit.raw["citation_intents"] = entry["intents"]
            if entry.get("isInfluential"):
                citation_hit.raw["influential_citation"] = True
            hits.append(citation_hit)
        return hits

    async def lookup_doi(self, doi: str) -> SearchHit | None:
        normalized = normalize_doi(doi)
        if not normalized:
            return None
        endpoint = f"{self.base_url}/paper/DOI:{normalized}"
        try:
            paper = await self.client.get_json(
                endpoint,
                provider=self.name,
                params={"fields": _FIELDS},
                headers=self._headers(),
            )
        except SourceUnavailable:
            raise
        except Exception:
            return None
        if not isinstance(paper, dict) or not paper.get("paperId"):
            return None
        return self._to_hit(
            paper, endpoint=endpoint, method=RetrievalMethod.PRIMARY_SOURCE_CHASE
        )
