"""Fixture-backed sources.

These serve a fixed corpus instead of a network. They exist so the whole
pipeline - search, citation traversal, fetch, deduplication, persistence,
inspection - can run with no credentials and no outbound requests, which is
what keeps tests deterministic and gives the CLI a demonstration that works
anywhere.

They implement the same interface as the real adapters, so nothing above the
source layer knows the difference.
"""

from __future__ import annotations

from typing import Any

from research.normalize.document import build_document
from research.models.common import (
    Provenance,
    RetrievalMethod,
    SourceFamily,
    SourceType,
)
from research.models.evidence import EvidenceDocument
from research.models.query import ResearchQuery, SearchHit
from research.normalize.doi import normalize_doi
from research.normalize.html import parse_date
from research.normalize.urls import canonicalize_url, registrable_domain
from research.sources.base import SourceAdapter, SourceCapabilities
from research.sources.classify import classify_url


def _score(entry: dict[str, Any], terms: list[str]) -> int:
    haystack = " ".join(
        str(entry.get(key, ""))
        for key in ("title", "abstract", "snippet", "text", "publisher", "venue", "keywords")
    ).casefold()
    return sum(haystack.count(term) for term in terms)


def _terms(text: str) -> list[str]:
    return [term for term in text.casefold().split() if len(term) > 2]


class StaticWebSource(SourceAdapter):
    """Web/news search over a fixed corpus."""

    name = "static_web"
    family = SourceFamily.WEB

    def __init__(
        self,
        entries: list[dict[str, Any]],
        *,
        name: str | None = None,
        family: SourceFamily = SourceFamily.WEB,
    ) -> None:
        self.entries = entries
        self.family = family
        if name:
            self.name = name

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(SourceFamily.WEB, SourceFamily.NEWS),
            default_source_type=SourceType.WEB_PAGE,
            supports_fetch=False,
            returns_inline_documents=False,
            supports_date_filter=True,
            requires_api_key=False,
            max_results_per_query=max(len(self.entries), 1),
            tags=frozenset({"offline", "fixture"}),
            notes="Fixed corpus; pointers only, bodies come from the fetcher.",
        )

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        terms = _terms(query.text)
        wants_news = SourceFamily.NEWS in query.families
        scored: list[tuple[int, SearchHit]] = []
        for entry in self.entries:
            is_news = entry.get("family", "news") == "news"
            if wants_news and not is_news:
                continue
            score = _score(entry, terms)
            if terms and not score:
                continue
            url = canonicalize_url(entry.get("url"))
            if not url:
                continue
            scored.append(
                (
                    score,
                    SearchHit(
                        provider=self.name,
                        url=url,
                        title=entry.get("title"),
                        snippet=entry.get("snippet"),
                        published_at=parse_date(entry.get("published_at")),
                        source_type=SourceType.coerce(
                            entry.get("source_type"),
                            classify_url(url, title=entry.get("title"), is_news_result=is_news),
                        ),
                        source_family=SourceFamily.NEWS if is_news else SourceFamily.WEB,
                        publisher=entry.get("publisher") or registrable_domain(url),
                        raw={"_endpoint": "fixture", "fixture_id": entry.get("id")},
                    ),
                )
            )
        scored.sort(key=lambda pair: (-pair[0], pair[1].url or ""))
        return [hit for _, hit in scored[: query.limit]]

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        raise NotImplementedError("fixture web results are fetched through the URL fetcher")


class StaticAcademicSource(SourceAdapter):
    """Academic search and citation graph over a fixed corpus.

    Corpus entries are dicts with ``id``, ``title``, ``abstract``, ``doi``,
    ``authors``, ``published_at``, ``venue``, ``type`` and ``references``
    (a list of corpus ids). Forward citations are derived by inverting
    ``references``, exactly as a real citation index would present them.
    """

    name = "static_academic"
    family = SourceFamily.ACADEMIC
    source_type = SourceType.ACADEMIC_PEER_REVIEWED

    def __init__(self, entries: list[dict[str, Any]], *, name: str | None = None) -> None:
        self.entries = entries
        self.by_id = {entry["id"]: entry for entry in entries}
        self.cited_by: dict[str, list[str]] = {}
        for entry in entries:
            for reference in entry.get("references", []):
                self.cited_by.setdefault(reference, []).append(entry["id"])
        if name:
            self.name = name

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
            max_results_per_query=max(len(self.entries), 1),
            tags=frozenset({"offline", "fixture", "citation-graph"}),
            notes="Fixed literature corpus with a citation graph.",
        )

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        terms = _terms(query.text)
        scored = []
        for entry in self.entries:
            score = _score(entry, terms)
            if terms and not score:
                continue
            scored.append((score, self._hit(entry, RetrievalMethod.SEARCH)))
        scored.sort(key=lambda pair: (-pair[0], pair[1].external_id or ""))
        return [hit for _, hit in scored[: query.limit]]

    async def get_references(self, hit: SearchHit, *, limit: int = 50) -> list[SearchHit]:
        entry = self.by_id.get(hit.external_id or "")
        if not entry:
            return []
        return [
            self._hit(self.by_id[ref], RetrievalMethod.REFERENCE_EXPANSION)
            for ref in entry.get("references", [])[:limit]
            if ref in self.by_id
        ]

    async def get_citing_papers(self, hit: SearchHit, *, limit: int = 50) -> list[SearchHit]:
        citing = self.cited_by.get(hit.external_id or "", [])
        return [
            self._hit(self.by_id[paper_id], RetrievalMethod.CITATION_EXPANSION)
            for paper_id in citing[:limit]
            if paper_id in self.by_id
        ]

    async def lookup_doi(self, doi: str) -> SearchHit | None:
        normalized = normalize_doi(doi)
        for entry in self.entries:
            if normalize_doi(entry.get("doi")) == normalized:
                return self._hit(entry, RetrievalMethod.PRIMARY_SOURCE_CHASE)
        return None

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        entry = self.by_id.get(hit.external_id or "")
        return build_document(
            provider=self.name,
            source_type=hit.source_type,
            source_family=SourceFamily.ACADEMIC,
            provenance=Provenance(
                provider=self.name,
                retrieval_method=RetrievalMethod.coerce(
                    hit.raw.get("_retrieval_method"), RetrievalMethod.SEARCH
                ),
                provider_endpoint="fixture",
                requested_url=hit.url,
            ),
            title=hit.title,
            abstract=hit.snippet,
            text=(entry or {}).get("text"),
            url=hit.url,
            external_id=hit.external_id,
            authors=list(hit.authors),
            published_at=hit.published_at,
            publisher=hit.publisher,
            doi=hit.doi,
            metadata={
                key: value
                for key, value in hit.raw.items()
                if not key.startswith("_")
            },
        )

    def _hit(self, entry: dict[str, Any], method: RetrievalMethod) -> SearchHit:
        doi = normalize_doi(entry.get("doi"))
        source_type = SourceType.coerce(
            entry.get("source_type"),
            SourceType.ACADEMIC_PREPRINT
            if (entry.get("venue") or "").lower().startswith("arxiv")
            else SourceType.ACADEMIC_PEER_REVIEWED,
        )
        return SearchHit(
            provider=self.name,
            external_id=entry["id"],
            url=entry.get("url") or (f"https://doi.org/{doi}" if doi else None),
            title=entry.get("title"),
            snippet=entry.get("abstract"),
            authors=list(entry.get("authors", [])),
            published_at=parse_date(entry.get("published_at")),
            source_type=source_type,
            source_family=SourceFamily.ACADEMIC,
            publisher=entry.get("venue"),
            doi=doi,
            raw={
                "venue": entry.get("venue"),
                "cited_by_count": len(self.cited_by.get(entry["id"], [])),
                "is_retracted": entry.get("is_retracted", False),
                "fixture_id": entry["id"],
                "_endpoint": "fixture",
                "_retrieval_method": str(method),
            },
        )
