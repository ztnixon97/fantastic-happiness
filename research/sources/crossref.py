"""Crossref adapter.

Crossref is the DOI registry: it is the authority for what a DOI points at,
and its records carry the reference lists publishers deposit. It has no
forward citation index, which the capabilities declare so the planner never
asks it for citing works.
"""

from __future__ import annotations

from typing import Any

from research.errors import SourceUnavailable
from research.models.common import RetrievalMethod, SourceFamily, SourceType
from research.models.query import ResearchQuery, SearchHit
from research.normalize.doi import normalize_doi
from research.normalize.text import normalize_whitespace
from research.sources.academic import (
    AcademicAdapter,
    author_names,
    classify_work,
    date_from_parts,
    strip_jats,
)
from research.sources.base import SourceCapabilities

_MAX_ROWS = 50

#: Fields Crossref accepts in ``select`` on the /works route. The list is
#: route-specific and the API rejects the whole request - HTTP 400, no
#: results - if one field is not selectable there. ``language`` is returned in
#: full records but is *not* selectable, which is the kind of thing only a
#: live call finds.
SELECT_FIELDS = (
    "DOI",
    "title",
    "author",
    "issued",
    "created",
    "abstract",
    "type",
    "container-title",
    "publisher",
    "URL",
    "is-referenced-by-count",
    "subject",
)


class CrossrefSource(AcademicAdapter):
    name = "crossref"
    base_url = "https://api.crossref.org"

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(SourceFamily.ACADEMIC,),
            default_source_type=SourceType.ACADEMIC_PEER_REVIEWED,
            returns_inline_documents=True,
            supports_date_filter=True,
            supports_references=True,
            supports_citing_papers=False,
            requires_api_key=False,
            max_results_per_query=_MAX_ROWS,
            tags=frozenset({"keyless", "doi-authority"}),
            notes="DOI registration records and publisher-deposited reference lists.",
        )

    def _headers(self) -> dict[str, str]:
        if self.contact_email:
            return {"User-Agent": f"research-environment/0.1 (mailto:{self.contact_email})"}
        return {}

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        params: dict[str, Any] = {
            "query.bibliographic": query.text,
            "rows": min(query.limit, _MAX_ROWS),
            "select": ",".join(SELECT_FIELDS),
        }
        date_filters = []
        if query.published_after:
            date_filters.append(f"from-pub-date:{query.published_after.date()}")
        if query.published_before:
            date_filters.append(f"until-pub-date:{query.published_before.date()}")
        if date_filters:
            params["filter"] = ",".join(date_filters)
        if self.contact_email:
            params["mailto"] = self.contact_email

        endpoint = f"{self.base_url}/works"
        payload = await self.client.get_json(
            endpoint, provider=self.name, params=params, headers=self._headers()
        )
        items = ((payload or {}).get("message") or {}).get("items")
        if not isinstance(items, list):
            raise SourceUnavailable("unexpected Crossref payload", provider=self.name)
        return [
            self._to_hit(item, endpoint=endpoint, method=RetrievalMethod.SEARCH)
            for item in items
            if isinstance(item, dict)
        ]

    def _to_hit(
        self, item: dict[str, Any], *, endpoint: str, method: RetrievalMethod
    ) -> SearchHit:
        doi = normalize_doi(item.get("DOI"))
        title = _first(item.get("title"))
        venue = _first(item.get("container-title"))
        references = [
            normalize_doi(reference.get("DOI"))
            for reference in (item.get("reference") or [])
            if isinstance(reference, dict) and reference.get("DOI")
        ]
        raw = {
            "venue": venue,
            "work_type": item.get("type"),
            "publisher": item.get("publisher"),
            "cited_by_count": item.get("is-referenced-by-count"),
            "subjects": item.get("subject") or [],
            "reference_dois": [doi for doi in references if doi][:200],
            "reference_count": item.get("reference-count"),
            "_endpoint": endpoint,
            "_retrieval_method": str(method),
        }
        if item.get("update-to"):
            # Crossref records corrections and retractions as update
            # relationships; carry them so a retraction is never lost.
            raw["updates"] = item["update-to"]
        return SearchHit(
            provider=self.name,
            external_id=doi,
            url=item.get("URL") or (f"https://doi.org/{doi}" if doi else None),
            title=normalize_whitespace(title) if title else None,
            snippet=strip_jats(item.get("abstract")),
            authors=author_names(item.get("author") or []),
            published_at=date_from_parts((item.get("issued") or {}).get("date-parts"))
            or date_from_parts((item.get("created") or {}).get("date-parts")),
            source_type=classify_work(
                work_type=item.get("type"), venue=venue, doi=doi, url=item.get("URL")
            ),
            source_family=SourceFamily.ACADEMIC,
            publisher=venue or item.get("publisher"),
            doi=doi,
            language=item.get("language"),
            score=item.get("score"),
            raw=raw,
        )

    async def get_references(self, hit: SearchHit, *, limit: int = 50) -> list[SearchHit]:
        """Resolve the DOIs a work cites.

        Crossref returns reference lists as bare DOIs plus loose text, so each
        resolvable DOI costs one lookup. That is exactly why traversal is
        bounded by document count and not only by depth.
        """
        dois = [doi for doi in (hit.raw.get("reference_dois") or []) if doi]
        if not dois and hit.doi:
            item = await self._get_work(hit.doi)
            dois = [
                normalize_doi(reference.get("DOI"))
                for reference in (item.get("reference") or [])
                if isinstance(reference, dict) and reference.get("DOI")
            ]
            dois = [doi for doi in dois if doi]

        hits: list[SearchHit] = []
        for doi in dois[:limit]:
            resolved = await self.lookup_doi(doi, method=RetrievalMethod.REFERENCE_EXPANSION)
            if resolved:
                hits.append(resolved)
        return hits

    async def get_citing_papers(self, hit: SearchHit, *, limit: int = 50) -> list[SearchHit]:
        """Not available: Crossref exposes no open forward-citation index."""
        return []

    async def _get_work(self, doi: str) -> dict[str, Any]:
        endpoint = f"{self.base_url}/works/{doi}"
        payload = await self.client.get_json(
            endpoint, provider=self.name, headers=self._headers()
        )
        message = (payload or {}).get("message")
        if not isinstance(message, dict):
            raise SourceUnavailable("unexpected Crossref work payload", provider=self.name)
        return message

    async def lookup_doi(
        self, doi: str, *, method: RetrievalMethod = RetrievalMethod.PRIMARY_SOURCE_CHASE
    ) -> SearchHit | None:
        normalized = normalize_doi(doi)
        if not normalized:
            return None
        try:
            item = await self._get_work(normalized)
        except SourceUnavailable:
            raise
        except Exception:
            return None
        return self._to_hit(
            item, endpoint=f"{self.base_url}/works/{normalized}", method=method
        )


def _first(value: Any) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None
