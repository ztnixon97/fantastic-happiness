"""Bounded traversal of the citation graph.

Following references backwards finds the work a literature rests on;
following citations forwards finds replications, corrections and rebuttals.
Both directions expand combinatorially, so every traversal declares its
limits before it starts and stops on the first one it reaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from research.acquisition.pipeline import AcquisitionResult, EvidenceAcquirer
from research.errors import SourceError
from research.graph.citations import hit_from_document
from research.models.common import RetrievalMethod, SourceFamily
from research.models.query import SearchHit
from research.budgets import BudgetLedger, Resource
from research.sources.base import CitationSource, ResearchSource
from research.sources.registry import SourceRegistry
from research.storage.store import ResearchStore

Direction = Literal["backward", "forward", "both"]


@dataclass(slots=True)
class CitationChaseResult:
    root_document_id: str
    direction: Direction
    max_depth: int
    depth_reached: int = 0
    visited_documents: list[str] = field(default_factory=list)
    added_document_ids: list[str] = field(default_factory=list)
    edges_recorded: int = 0
    already_held: int = 0
    provider_errors: dict[str, str] = field(default_factory=dict)
    stopped_by: str = "traversal complete"

    def summary(self) -> dict[str, Any]:
        return {
            "root": self.root_document_id,
            "direction": self.direction,
            "max_depth": self.max_depth,
            "depth_reached": self.depth_reached,
            "visited": len(self.visited_documents),
            "documents_added": len(self.added_document_ids),
            "edges": self.edges_recorded,
            "already_held": self.already_held,
            "provider_errors": dict(self.provider_errors),
            "stopped_by": self.stopped_by,
        }


class CitationChase:
    """Breadth-first expansion through references and citing works."""

    def __init__(
        self,
        store: ResearchStore,
        registry: SourceRegistry,
        *,
        investigation_id: str,
        ledger: BudgetLedger | None = None,
        acquirer: EvidenceAcquirer | None = None,
        task_id: str | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.investigation_id = investigation_id
        self.ledger = ledger
        self.acquirer = acquirer or EvidenceAcquirer(store, ledger=ledger)
        self.task_id = task_id

    async def chase(
        self,
        document_id: str,
        *,
        direction: Direction = "backward",
        max_depth: int = 1,
        max_documents: int = 25,
        per_node_limit: int = 20,
    ) -> CitationChaseResult:
        if self.ledger is not None:
            max_depth = min(max_depth, self.ledger.policy.max_citation_depth)
        result = CitationChaseResult(
            root_document_id=document_id, direction=direction, max_depth=max_depth
        )
        if max_depth < 1:
            result.stopped_by = "citation depth limit is zero"
            return result

        frontier: list[tuple[str, int]] = [(document_id, 0)]
        seen: set[str] = {document_id}

        while frontier:
            current_id, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            document = self.store.documents.get(current_id)
            source = self._source_for(document.provider)
            if source is None:
                result.stopped_by = "no configured provider exposes a citation graph"
                break

            result.visited_documents.append(current_id)
            hit = hit_from_document(document)

            for edge_direction in _directions(direction):
                if len(result.added_document_ids) >= max_documents:
                    result.stopped_by = f"document limit reached ({max_documents})"
                    return result
                neighbours = await self._expand(
                    source, hit, edge_direction, per_node_limit, result
                )
                for neighbour in neighbours:
                    if len(result.added_document_ids) >= max_documents:
                        result.stopped_by = f"document limit reached ({max_documents})"
                        return result
                    acquisition = await self._persist(
                        source, neighbour, current_id, edge_direction, depth + 1, result
                    )
                    if acquisition is None:
                        continue
                    if acquisition.refused:
                        result.stopped_by = acquisition.refused
                        return result
                    neighbour_id = acquisition.document.id
                    if acquisition.created and acquisition.verdict.independent:
                        result.added_document_ids.append(neighbour_id)
                    else:
                        result.already_held += 1
                    result.depth_reached = max(result.depth_reached, depth + 1)
                    if neighbour_id not in seen and depth + 1 < max_depth:
                        seen.add(neighbour_id)
                        frontier.append((neighbour_id, depth + 1))

        return result

    # -- internals ------------------------------------------------------
    def _source_for(self, provider: str) -> ResearchSource | None:
        """Prefer the provider that supplied the document, else any capable one."""
        preferred = self.registry.get(provider)
        if isinstance(preferred, CitationSource):
            capability = preferred.capabilities()
            if capability.supports_references or capability.supports_citing_papers:
                return preferred
        candidates = self.registry.citation_sources()
        return candidates[0] if candidates else None

    async def _expand(
        self,
        source: ResearchSource,
        hit: SearchHit,
        edge_direction: str,
        limit: int,
        result: CitationChaseResult,
    ) -> list[SearchHit]:
        capability = source.capabilities()
        if edge_direction == "backward" and not capability.supports_references:
            return []
        if edge_direction == "forward" and not capability.supports_citing_papers:
            return []
        if self.ledger is not None and not self.ledger.try_spend(Resource.PROVIDER_CALLS):
            result.stopped_by = "provider call budget exhausted"
            return []
        try:
            if edge_direction == "backward":
                return await source.get_references(hit, limit=limit)
            return await source.get_citing_papers(hit, limit=limit)
        except SourceError as exc:
            result.provider_errors[source.name] = str(exc)
            if self.ledger is not None:
                self.ledger.try_spend(Resource.FAILED_SOURCE_CALLS)
            return []

    async def _persist(
        self,
        source: ResearchSource,
        neighbour: SearchHit,
        current_id: str,
        edge_direction: str,
        depth: int,
        result: CitationChaseResult,
    ) -> AcquisitionResult | None:
        if self.ledger is not None and not self.ledger.can_spend(Resource.CITATION_DOCUMENTS):
            result.stopped_by = "citation document budget exhausted"
            return None

        document = await source.fetch(neighbour)
        document.source_family = SourceFamily.ACADEMIC
        if document.provenance is not None:
            object.__setattr__(document.provenance, "parent_document_id", current_id)
            object.__setattr__(
                document.provenance,
                "retrieval_method",
                RetrievalMethod.REFERENCE_EXPANSION
                if edge_direction == "backward"
                else RetrievalMethod.CITATION_EXPANSION,
            )
            object.__setattr__(document.provenance, "task_id", self.task_id)

        acquisition = self.acquirer.persist(document, investigation_id=self.investigation_id)
        if acquisition.refused:
            return acquisition
        if self.ledger is not None:
            self.ledger.try_spend(Resource.CITATION_DOCUMENTS)

        neighbour_id = acquisition.document.id
        citing, cited = (
            (current_id, neighbour_id)
            if edge_direction == "backward"
            else (neighbour_id, current_id)
        )
        edge_id = self.store.citations.add(
            self.investigation_id,
            citing_document_id=citing,
            cited_document_id=cited,
            provider=source.name,
            cited_external_id=acquisition.document.external_id
            if edge_direction == "backward"
            else self.store.documents.get(current_id).external_id,
            cited_doi=acquisition.document.doi
            if edge_direction == "backward"
            else self.store.documents.get(current_id).doi,
            cited_title=acquisition.document.title if edge_direction == "backward" else None,
            depth=depth,
        )
        if edge_id is not None:
            result.edges_recorded += 1
        return acquisition


def _directions(direction: Direction) -> tuple[str, ...]:
    if direction == "both":
        return ("backward", "forward")
    return (direction,)
