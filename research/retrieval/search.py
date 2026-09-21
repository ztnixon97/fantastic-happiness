"""Searching what the investigation already holds.

Until now every search in this system meant calling a provider. That leaves
an investigation unable to answer the cheapest and most common question it
has - "do I already have something about this?" - so workers re-query
providers for material sitting in the store, and a document can only be read
back if its identifier is already known.

This is the retrieval side: lexical matching over the corpus, expansion
through the investigation's own graph, optionally vectors, fused into one
ranking. It spends no provider calls and no budget.

Results respect independence, as everything here does: a syndicated copy does
not occupy a second place in the ranking. It is folded into the result for
the document it copies and named there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from research.models.evidence import EvidenceDocument
from research.retrieval.embeddings import EmbeddingIndex
from research.retrieval.fusion import reciprocal_rank_fusion
from research.retrieval.graph import GraphExpansion
from research.retrieval.lexical import LexicalIndex
from research.storage.store import ResearchStore

#: Lexical results are the spine of the ranking; graph expansion is what a
#: keyword index cannot do; vectors, when enabled, are a third opinion.
DEFAULT_WEIGHTS = {"lexical": 1.0, "graph": 0.8, "vector": 0.9}


@dataclass(slots=True)
class CorpusHit:
    document: EvidenceDocument
    score: float
    snippet: str = ""
    ranks: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    #: Copies of this document that the ranking folded into this result.
    copies: list[str] = field(default_factory=list)

    @property
    def independent(self) -> bool:
        return not (self.document.duplicate_of or self.document.derived_from)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.document.id,
            "title": self.document.title,
            "type": str(self.document.source_type),
            "source": self.document.publisher
            or self.document.canonical_host
            or self.document.provider,
            "published": self.document.published_at.date().isoformat()
            if self.document.published_at
            else None,
            "score": round(self.score, 5),
            "found_by": sorted(self.ranks),
            "why": list(self.reasons),
            "snippet": self.snippet,
            "independent": self.independent,
            "copies": list(self.copies),
        }


class CorpusSearch:
    """Retrieval over held evidence. Costs nothing but a few indexed queries."""

    def __init__(
        self,
        store: ResearchStore,
        investigation_id: str,
        *,
        embeddings: EmbeddingIndex | None = None,
    ) -> None:
        self.store = store
        self.investigation_id = investigation_id
        self.lexical = LexicalIndex(store.db)
        self.graph = GraphExpansion(store, investigation_id)
        self.embeddings = embeddings

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        expand: bool = True,
        collapse_copies: bool = True,
        mode: str = "any",
    ) -> list[CorpusHit]:
        pool = max(limit * 3, 15)
        lexical_hits = self.lexical.search(
            query, investigation_id=self.investigation_id, limit=pool, mode=mode
        )
        rankings: dict[str, Sequence[str]] = {
            "lexical": [hit.document_id for hit in lexical_hits]
        }
        reasons: dict[str, list[str]] = {}
        snippets = {hit.document_id: hit.snippet for hit in lexical_hits}

        if expand and lexical_hits:
            seeds = [hit.document_id for hit in lexical_hits[:5]]
            graph_hits = self.graph.expand(seeds, limit=pool)
            rankings["graph"] = [hit.document_id for hit in graph_hits]
            for hit in graph_hits:
                reasons.setdefault(hit.document_id, []).extend(hit.reasons)

        if self.embeddings is not None:
            candidates = {document_id for ordered in rankings.values() for document_id in ordered}
            vector_hits = await self.embeddings.search(
                query,
                investigation_id=self.investigation_id,
                limit=pool,
                candidates=sorted(candidates) or None,
            )
            rankings["vector"] = [hit.document_id for hit in vector_hits]

        fused = reciprocal_rank_fusion(rankings, weights=DEFAULT_WEIGHTS, reasons=reasons)
        if not fused:
            return []

        documents = {
            document.id: document
            for document in self.store.documents.get_many([hit.document_id for hit in fused])
        }
        hits: list[CorpusHit] = []
        for entry in fused:
            document = documents.get(entry.document_id)
            if document is None:
                continue
            hits.append(
                CorpusHit(
                    document=document,
                    score=entry.score,
                    snippet=snippets.get(entry.document_id, ""),
                    ranks=dict(entry.ranks),
                    reasons=list(entry.reasons),
                )
            )

        if collapse_copies:
            hits = self._collapse(hits)
        return hits[:limit]

    def _collapse(self, hits: list[CorpusHit]) -> list[CorpusHit]:
        """Keep one result per independent source.

        Ten copies of one story should occupy one place in a ranking, for the
        same reason they count as one source in a report. The copies are named
        on the result that stands for them, so nothing is hidden.
        """
        best: dict[str, CorpusHit] = {}
        order: list[str] = []
        for hit in hits:
            key = hit.document.independence_key or hit.document.id
            if key not in best:
                best[key] = hit
                order.append(key)
                continue
            kept = best[key]
            if hit.document.id not in kept.copies:
                kept.copies.append(hit.document.id)
            # Prefer the original over a copy as the representative result.
            if kept.document.duplicate_of and not hit.document.duplicate_of:
                hit.copies = [
                    document_id for document_id in [*kept.copies, kept.document.id]
                    if document_id != hit.document.id
                ]
                hit.score = max(hit.score, kept.score)
                best[key] = hit
        return [best[key] for key in order]

    # -- maintenance ----------------------------------------------------
    def rebuild_index(self) -> int:
        return self.lexical.rebuild(self.investigation_id)

    def stats(self) -> dict[str, Any]:
        return {
            "documents": self.store.documents.count(self.investigation_id),
            "indexed": self.lexical.count(self.investigation_id),
            "embedded": self.embeddings.count(self.investigation_id)
            if self.embeddings
            else 0,
        }
