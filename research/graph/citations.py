"""Queries over the stored citation graph.

Read-only views: nothing here retrieves. Acquisition through citations lives
in :mod:`research.operations.citation_chase`, which is where the limits are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research.models.evidence import EvidenceDocument
from research.models.query import SearchHit
from research.storage.store import ResearchStore


@dataclass(frozen=True, slots=True)
class CitationEdge:
    citing_document_id: str | None
    cited_document_id: str | None
    cited_doi: str | None
    cited_external_id: str | None
    cited_title: str | None
    provider: str
    depth: int

    @property
    def resolved(self) -> bool:
        """True when both ends are documents the investigation holds."""
        return bool(self.citing_document_id and self.cited_document_id)


def hit_from_document(document: EvidenceDocument) -> SearchHit:
    """Rebuild the provider pointer for a stored document.

    Traversal starts from documents, but providers answer about *their*
    identifiers, so a stored record has to be turned back into a hit.
    """
    raw = dict(document.metadata)
    raw.setdefault("_endpoint", "")
    if document.external_id and document.provider == "openalex":
        raw.setdefault("openalex_id", document.external_id)
    if document.external_id and document.provider == "semantic_scholar":
        raw.setdefault("semantic_scholar_id", document.external_id)
    return SearchHit(
        provider=document.provider,
        external_id=document.external_id,
        url=document.canonical_url,
        title=document.title,
        authors=list(document.authors),
        published_at=document.published_at,
        source_type=document.source_type,
        source_family=document.source_family,
        publisher=document.publisher,
        doi=document.doi,
        raw=raw,
    )


class CitationGraph:
    """The citation edges an investigation has accumulated."""

    def __init__(self, store: ResearchStore, investigation_id: str) -> None:
        self.store = store
        self.investigation_id = investigation_id

    def references(self, document_id: str) -> list[CitationEdge]:
        rows = self.store.citations.references_of(self.investigation_id, document_id)
        return [_edge(row) for row in rows]

    def citing(self, document_id: str) -> list[CitationEdge]:
        rows = self.store.citations.citing_of(self.investigation_id, document_id)
        return [_edge(row) for row in rows]

    def edges(self, *, resolved_only: bool = False, limit: int = 1000) -> list[CitationEdge]:
        rows = self.store.db.query(
            "SELECT * FROM citations WHERE investigation_id IS ? ORDER BY id LIMIT ?",
            (self.investigation_id, limit),
        )
        edges = [_edge(dict(row)) for row in rows]
        return [edge for edge in edges if edge.resolved] if resolved_only else edges

    def in_degree(self, *, limit: int = 20) -> list[tuple[str, int]]:
        """Documents most cited *by the corpus this investigation holds*.

        A count within the corpus, not a global citation count: it says which
        works this investigation's own literature keeps pointing at, which is
        how foundational work surfaces.
        """
        rows = self.store.db.query(
            "SELECT cited_document_id AS id, COUNT(*) AS n FROM citations "
            "WHERE investigation_id IS ? AND cited_document_id IS NOT NULL "
            "GROUP BY cited_document_id ORDER BY n DESC, cited_document_id LIMIT ?",
            (self.investigation_id, limit),
        )
        return [(row["id"], int(row["n"])) for row in rows]

    def unresolved_frontier(self, *, limit: int = 50) -> list[CitationEdge]:
        """Cited works that are known of but not held.

        This is the traversal frontier, visible without spending anything to
        look at it.
        """
        rows = self.store.db.query(
            "SELECT * FROM citations WHERE investigation_id IS ? "
            "AND cited_document_id IS NULL AND cited_doi != '' ORDER BY id LIMIT ?",
            (self.investigation_id, limit),
        )
        return [_edge(dict(row)) for row in rows]

    def stats(self) -> dict[str, Any]:
        total = self.store.citations.count(self.investigation_id)
        resolved = int(
            self.store.db.scalar(
                "SELECT COUNT(*) FROM citations WHERE investigation_id IS ? "
                "AND cited_document_id IS NOT NULL",
                (self.investigation_id,),
            )
            or 0
        )
        return {
            "edges": total,
            "resolved_edges": resolved,
            "unresolved_edges": total - resolved,
            "max_depth": int(
                self.store.db.scalar(
                    "SELECT COALESCE(MAX(depth), 0) FROM citations WHERE investigation_id IS ?",
                    (self.investigation_id,),
                )
                or 0
            ),
        }


def _edge(row: dict[str, Any]) -> CitationEdge:
    return CitationEdge(
        citing_document_id=row.get("citing_document_id"),
        cited_document_id=row.get("cited_document_id"),
        cited_doi=row.get("cited_doi") or None,
        cited_external_id=row.get("cited_external_id") or None,
        cited_title=row.get("cited_title"),
        provider=row.get("provider", "unknown"),
        depth=int(row.get("depth") or 0),
    )
