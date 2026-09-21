"""Read models for the UI.

The UI gets its own assembling layer so the research core never grows a
presentation concern. Everything here is a projection of stored state into
JSON; nothing here can write, search or spend.
"""

from __future__ import annotations

from typing import Any

from research.budgets import BudgetLedger
from research.config import BudgetPolicy, ResearchConfig
from research.graph.citations import CitationGraph
from research.graph.claims import ClaimGraph
from research.graph.entities import EntityRegistry
from research.graph.independence import independent_documents
from research.graph.timelines import Timeline
from research.models.claim import EvidenceStance
from research.normalize.text import truncate
from research.operations.claims import ClaimOperations
from research.orchestration.stopping import StoppingRules
from research.storage.store import ResearchStore
from research.synthesis.report import collect


class InvestigationView:
    """Everything the UI can ask about one investigation."""

    def __init__(
        self,
        store: ResearchStore,
        investigation_id: str,
        *,
        config: ResearchConfig | None = None,
    ) -> None:
        self.store = store
        self.investigation_id = investigation_id
        self.config = config or ResearchConfig()

    # -- overview -------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        investigation = self.store.investigations.get(self.investigation_id)
        documents = self.store.documents.list(self.investigation_id, limit=2000)
        groups = independent_documents(self.store.documents, [d.id for d in documents])
        ledger = self._ledger(investigation.budget)
        rules = StoppingRules(self.store, self.investigation_id, ledger)
        return {
            "id": investigation.id,
            "question": investigation.question,
            "brief": investigation.metadata.get("plan_brief") or investigation.brief,
            "status": str(investigation.status),
            "stop_reason": str(investigation.stop_reason) if investigation.stop_reason else None,
            "stop_detail": investigation.stop_detail,
            "created_at": (
                investigation.created_at.isoformat() if investigation.created_at else None
            ),
            "counts": {
                "documents": len(documents),
                "independent_sources": len(groups),
                "claims": len(self.store.claims.list(self.investigation_id, hydrate=False)),
                "entities": len(self.store.entities.list(self.investigation_id, limit=1000)),
                "events": len(self.store.events.timeline(self.investigation_id, limit=1000)),
                "tasks": self.store.tasks.count(self.investigation_id),
                "searches": self.store.queries.count(self.investigation_id),
                "failed_fetches": self.store.fetches.failure_count(self.investigation_id),
            },
            "budget": ledger.snapshot(),
            "stopping_rules": rules.describe(),
        }

    def _ledger(self, budget: dict[str, Any]) -> BudgetLedger:
        policy = BudgetPolicy.from_dict(budget) if budget else self.config.budget
        return BudgetLedger(self.store.budget, self.investigation_id, policy)

    # -- tasks ----------------------------------------------------------
    def tasks(self) -> dict[str, Any]:
        tasks = self.store.tasks.list(self.investigation_id, limit=500)
        return {
            "tasks": [
                {
                    "id": task.id,
                    "parent": task.parent_task_id,
                    "role": str(task.role),
                    "operation": str(task.operation),
                    "status": str(task.status),
                    "depth": task.depth,
                    "priority": task.priority,
                    "objective": task.objective,
                    "summary": task.result.summary if task.result else None,
                    "evidence": task.result.evidence_ids if task.result else [],
                    "claims": task.result.claim_ids if task.result else [],
                    "open_questions": task.result.open_questions if task.result else [],
                    "metrics": task.result.metrics if task.result else {},
                    "error": task.error,
                }
                for task in tasks
            ]
        }

    # -- claims ---------------------------------------------------------
    def claims(self) -> dict[str, Any]:
        graph = ClaimGraph(self.store, self.investigation_id)
        payload = []
        for assessment in graph.assess_all(limit=300):
            links = self.store.claims.evidence_links(assessment.claim_id)
            payload.append(
                {
                    **assessment.to_dict(),
                    "links": [
                        {
                            "document_id": link.document_id,
                            "stance": str(link.stance),
                            "excerpt": link.excerpt,
                            "analysis": link.analysis,
                            "task_id": link.created_by_task_id,
                        }
                        for link in links
                    ],
                }
            )
        return {"claims": payload}

    def open_questions(self) -> dict[str, Any]:
        operations = ClaimOperations(self.store, investigation_id=self.investigation_id)
        return {"open_questions": operations.open_questions(limit=100)}

    # -- graph ----------------------------------------------------------
    def graph(self) -> dict[str, Any]:
        """Nodes and edges for the claim/evidence graph.

        Copies are folded into the document they copy, because drawing them
        separately is the visual form of the same mistake as counting them
        separately.
        """
        documents = self.store.documents.list(self.investigation_id, limit=2000)
        by_id = {document.id: document for document in documents}
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for group in independent_documents(self.store.documents, [d.id for d in documents]):
            members = [by_id[document_id] for document_id in group if document_id in by_id]
            if not members:
                continue
            head = next((member for member in members if not member.duplicate_of), members[0])
            nodes.append(
                {
                    "id": head.id,
                    "kind": "evidence",
                    "label": truncate(head.title or head.id, 70),
                    "type": str(head.source_type),
                    "source": head.publisher or head.canonical_host or head.provider,
                    "published": (
                        head.published_at.date().isoformat() if head.published_at else None
                    ),
                    "url": head.canonical_url,
                    "copies": [member.id for member in members if member.id != head.id],
                    "retracted": bool(head.metadata.get("is_retracted")),
                }
            )

        represented = {node["id"]: node["id"] for node in nodes}
        for node in nodes:
            for copy_id in node["copies"]:
                represented[copy_id] = node["id"]

        for assessment in ClaimGraph(self.store, self.investigation_id).assess_all(limit=300):
            nodes.append(
                {
                    "id": assessment.claim_id,
                    "kind": "claim",
                    "label": truncate(assessment.text, 90),
                    "status": str(assessment.status),
                    "explanation": assessment.explanation,
                    "independent_support": assessment.support.independent_count,
                    "independent_against": assessment.contradiction.independent_count,
                }
            )
            for link in self.store.claims.evidence_links(assessment.claim_id):
                target = represented.get(link.document_id)
                if not target:
                    continue
                edges.append(
                    {
                        "source": target,
                        "target": assessment.claim_id,
                        "kind": str(link.stance),
                        "via_copy": target != link.document_id,
                    }
                )

        citations = CitationGraph(self.store, self.investigation_id)
        for edge in citations.edges(resolved_only=True, limit=500):
            source = represented.get(edge.citing_document_id or "")
            target = represented.get(edge.cited_document_id or "")
            if source and target and source != target:
                edges.append({"source": source, "target": target, "kind": "cites"})

        return {"nodes": nodes, "edges": _dedupe_edges(edges)}

    # -- entities and timeline ------------------------------------------
    def entities(self) -> dict[str, Any]:
        registry = EntityRegistry(self.store, self.investigation_id)
        entities = self.store.entities.list(self.investigation_id, limit=500)
        return {
            "entities": [
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": str(entity.entity_type),
                    "identifiers": [identifier.key() for identifier in entity.identifiers],
                    "aliases": entity.aliases,
                    "documents": registry.documents_for(entity.id),
                    "claims": registry.claims_for(entity.id),
                    "review_needed": bool(entity.metadata.get("review_needed")),
                }
                for entity in entities
            ]
        }

    def timeline(self) -> dict[str, Any]:
        entries = Timeline(self.store, self.investigation_id).build(
            include_publications=True, limit=300
        )
        return {"timeline": [entry.to_dict() for entry in entries]}

    # -- sources and activity -------------------------------------------
    def sources(self) -> dict[str, Any]:
        data = collect(self.store, self.investigation_id)
        return {
            "sources": [
                {
                    "id": source.document.id,
                    "citation": source.citation,
                    "title": source.document.title,
                    "type": str(source.document.source_type),
                    "family": str(source.document.source_family),
                    "url": source.document.canonical_url,
                    "doi": source.document.doi,
                    "published": source.document.published_at.date().isoformat()
                    if source.document.published_at
                    else None,
                    "provenance": source.document.provenance.describe()
                    if source.document.provenance
                    else None,
                    "copies": [
                        {"id": copy.id, "relation": str(copy.duplicate_relation),
                         "source": copy.canonical_host or copy.provider}
                        for copy in source.copies
                    ],
                }
                for source in data.sources
            ]
        }

    def activity(self) -> dict[str, Any]:
        return {
            "searches": self.store.queries.list(self.investigation_id, limit=200),
            "fetches": self.store.fetches.list(self.investigation_id, limit=200),
        }

    def document(self, document_id: str) -> dict[str, Any]:
        document = self.store.documents.get(document_id)
        links = self.store.claims.claims_for_document(document_id)
        return {
            "id": document.id,
            "title": document.title,
            "type": str(document.source_type),
            "family": str(document.source_family),
            "provider": document.provider,
            "url": document.canonical_url,
            "doi": document.doi,
            "authors": document.authors,
            "publisher": document.publisher,
            "published": document.published_at.isoformat() if document.published_at else None,
            "fetched": document.fetched_at.isoformat() if document.fetched_at else None,
            "content_hash": document.content_hash,
            "provenance": document.provenance.describe() if document.provenance else None,
            "duplicate_of": document.duplicate_of,
            "derived_from": document.derived_from,
            "duplicate_relation": str(document.duplicate_relation)
            if document.duplicate_relation
            else None,
            "duplicate_reason": document.metadata.get("duplicate_reason"),
            "independence_key": document.independence_key,
            "retracted": bool(document.metadata.get("is_retracted")),
            "claims": [
                {"claim_id": link.claim_id, "stance": str(link.stance), "excerpt": link.excerpt}
                for link in links
            ],
            # Sent as plain text and rendered as text by the client. Retrieved
            # content is never handed to a browser as markup.
            "text": truncate(document.best_text, 20000),
        }


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["kind"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def investigations(store: ResearchStore) -> dict[str, Any]:
    return {
        "investigations": [
            {
                "id": investigation.id,
                "question": investigation.question,
                "status": str(investigation.status),
                "stop_reason": str(investigation.stop_reason)
                if investigation.stop_reason
                else None,
                "created_at": investigation.created_at.isoformat()
                if investigation.created_at
                else None,
                "documents": store.documents.count(investigation.id),
                "claims": len(store.claims.list(investigation.id, hydrate=False)),
            }
            for investigation in store.investigations.list(limit=100)
        ]
    }


STANCE_KINDS = {str(stance) for stance in EvidenceStance} | {"cites"}
