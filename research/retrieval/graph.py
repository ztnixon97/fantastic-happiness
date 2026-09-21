"""Retrieval through the investigation's own graph.

Lexical search finds documents that use a word. Graph expansion finds the
documents that *matter* given those: the work they cite, the papers that cite
them, the other evidence attached to the same claim, the material about the
same entities. It is the part of retrieval that a keyword index cannot do,
and it costs one indexed query per relation rather than a model call.

Every expansion carries its reason, so a result can say why it surfaced -
"cited by evidence:12", "also supports claim:3" - rather than arriving with
an unexplained score.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from research.storage.store import ResearchStore

#: Relative weights. A document attached to the same claim is more relevant
#: to a query that surfaced its neighbour than one merely naming the same
#: organisation; a copy is barely relevant at all, since it is the same
#: source, but is worth surfacing so a reader knows it exists.
WEIGHTS = {
    "cites": 1.0,
    "cited_by": 1.0,
    "same_claim": 1.2,
    "same_entity": 0.6,
    "copy": 0.3,
}


@dataclass(slots=True)
class GraphHit:
    document_id: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def add(self, relation: str, reason: str) -> None:
        self.score += WEIGHTS.get(relation, 0.5)
        if reason not in self.reasons:
            self.reasons.append(reason)


class GraphExpansion:
    """Finds documents connected to a set of seeds."""

    def __init__(self, store: ResearchStore, investigation_id: str) -> None:
        self.store = store
        self.investigation_id = investigation_id

    def expand(
        self,
        seeds: Sequence[str],
        *,
        limit: int = 20,
        relations: Sequence[str] = ("cites", "cited_by", "same_claim", "same_entity", "copy"),
    ) -> list[GraphHit]:
        if not seeds:
            return []
        seed_set = set(seeds)
        hits: dict[str, GraphHit] = defaultdict(lambda: GraphHit(document_id=""))

        def record(document_id: str, relation: str, reason: str) -> None:
            if not document_id or document_id in seed_set:
                return
            hit = hits[document_id]
            hit.document_id = document_id
            hit.add(relation, reason)

        placeholders = ",".join("?" for _ in seed_set)
        params = tuple(seed_set)

        if "cites" in relations or "cited_by" in relations:
            for row in self.store.db.query(
                "SELECT citing_document_id, cited_document_id FROM citations "
                f"WHERE investigation_id IS ? AND (citing_document_id IN ({placeholders}) "
                f"OR cited_document_id IN ({placeholders})) AND cited_document_id IS NOT NULL",
                (self.investigation_id, *params, *params),
            ):
                citing, cited = row["citing_document_id"], row["cited_document_id"]
                if citing in seed_set and "cites" in relations:
                    record(cited, "cites", f"cited by {citing}")
                if cited in seed_set and "cited_by" in relations:
                    record(citing, "cited_by", f"cites {cited}")

        if "same_claim" in relations:
            for row in self.store.db.query(
                "SELECT other.document_id AS document_id, other.claim_id AS claim_id, "
                "       other.stance AS stance "
                "FROM claim_evidence AS seed JOIN claim_evidence AS other "
                "  ON seed.claim_id = other.claim_id "
                f"WHERE seed.document_id IN ({placeholders})",
                params,
            ):
                record(
                    row["document_id"],
                    "same_claim",
                    f"also {row['stance']} {row['claim_id']}",
                )

        if "same_entity" in relations:
            for row in self.store.db.query(
                "SELECT other.subject_id AS document_id, other.object_id AS entity_id "
                "FROM relationships AS seed JOIN relationships AS other "
                "  ON seed.object_id = other.object_id "
                "WHERE seed.investigation_id = ? AND seed.object_type = 'entity' "
                f"  AND other.subject_type = 'document' AND seed.subject_id IN ({placeholders})",
                (self.investigation_id, *params),
            ):
                record(row["document_id"], "same_entity", f"mentions {row['entity_id']}")

        if "copy" in relations:
            for row in self.store.db.query(
                "SELECT id, independence_key FROM documents WHERE investigation_id = ? "
                "AND independence_key IN ("
                "  SELECT COALESCE(independence_key, id) FROM documents "
                f"  WHERE id IN ({placeholders})"
                ")",
                (self.investigation_id, *params),
            ):
                record(row["id"], "copy", f"copy of the same source as {row['independence_key']}")

        ranked = sorted(hits.values(), key=lambda hit: (-hit.score, hit.document_id))
        return ranked[:limit]
