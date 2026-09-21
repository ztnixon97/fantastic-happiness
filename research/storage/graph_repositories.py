"""Repositories for the claim / entity / event / citation graph."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from research.errors import NotFound
from research.ids import CITATION, CLAIM, ENTITY, EVENT, LINK, RELATIONSHIP
from research.models.claim import (
    Claim,
    ClaimEvidenceLink,
    ClaimStatus,
    EvidenceStance,
)
from research.models.entity import Entity, EntityIdentifier, EntityType
from research.models.event import DatePrecision, Event
from research.models.common import utcnow
from research.normalize.text import normalize_title
from research.storage.database import Database, encode_dt, to_json
from research.storage.rows import (
    row_to_claim,
    row_to_claim_link,
    row_to_entity,
    row_to_event,
)


class ClaimRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        investigation_id: str,
        text: str,
        *,
        status: ClaimStatus = ClaimStatus.UNVERIFIED,
        notes: str | None = None,
        created_by_task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Claim:
        claim = Claim(
            id=self.db.next_id(CLAIM),
            investigation_id=investigation_id,
            text=text,
            status=status,
            notes=notes,
            created_by_task_id=created_by_task_id,
            metadata=dict(metadata or {}),
        )
        self.db.execute(
            "INSERT INTO claims(id, investigation_id, text, status, notes, metadata, "
            "created_by_task_id, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                claim.id,
                claim.investigation_id,
                claim.text,
                str(claim.status),
                claim.notes,
                to_json(claim.metadata),
                claim.created_by_task_id,
                encode_dt(claim.created_at),
                encode_dt(claim.updated_at),
            ),
        )
        return claim

    def get(self, claim_id: str, *, hydrate: bool = True) -> Claim:
        row = self.db.query_one("SELECT * FROM claims WHERE id = ?", (claim_id,))
        if row is None:
            raise NotFound(f"no such claim: {claim_id}")
        claim = row_to_claim(row)
        if hydrate:
            self._hydrate(claim)
        return claim

    def _hydrate(self, claim: Claim) -> None:
        for link in self.evidence_links(claim.id):
            if link.stance is EvidenceStance.SUPPORTS:
                claim.supporting_evidence_ids.append(link.document_id)
            elif link.stance is EvidenceStance.CONTRADICTS:
                claim.contradicting_evidence_ids.append(link.document_id)
        claim.related_claim_ids = [
            row["related_claim_id"]
            for row in self.db.query(
                "SELECT related_claim_id FROM claim_links WHERE claim_id = ?",
                (claim.id,),
            )
        ]
        claim.entity_ids = [
            row["object_id"]
            for row in self.db.query(
                "SELECT object_id FROM relationships WHERE subject_id = ? "
                "AND predicate = 'ABOUT' AND object_type = 'entity'",
                (claim.id,),
            )
        ]

    def list(
        self,
        investigation_id: str,
        *,
        status: ClaimStatus | None = None,
        limit: int = 200,
        hydrate: bool = True,
    ) -> list[Claim]:
        sql = "SELECT * FROM claims WHERE investigation_id = ?"
        params: list[Any] = [investigation_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(str(status))
        sql += " ORDER BY CAST(SUBSTR(id, INSTR(id, ':') + 1) AS INTEGER) LIMIT ?"
        params.append(limit)
        claims = [row_to_claim(row) for row in self.db.query(sql, tuple(params))]
        if hydrate:
            for claim in claims:
                self._hydrate(claim)
        return claims

    def link_evidence(self, link: ClaimEvidenceLink) -> str:
        link_id = self.db.next_id(LINK)
        self.db.execute(
            "INSERT OR IGNORE INTO claim_evidence(id, claim_id, document_id, stance, "
            "excerpt, analysis, created_by_task_id, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                link_id,
                link.claim_id,
                link.document_id,
                str(link.stance),
                link.excerpt,
                link.analysis,
                link.created_by_task_id,
                encode_dt(link.created_at),
            ),
        )
        return link_id

    def evidence_links(self, claim_id: str) -> list[ClaimEvidenceLink]:
        rows = self.db.query(
            "SELECT * FROM claim_evidence WHERE claim_id = ? ORDER BY created_at, id",
            (claim_id,),
        )
        return [row_to_claim_link(row) for row in rows]

    def claims_for_document(self, document_id: str) -> list[ClaimEvidenceLink]:
        rows = self.db.query(
            "SELECT * FROM claim_evidence WHERE document_id = ? ORDER BY claim_id",
            (document_id,),
        )
        return [row_to_claim_link(row) for row in rows]

    def set_status(self, claim_id: str, status: ClaimStatus, *, notes: str | None = None) -> None:
        self.db.execute(
            "UPDATE claims SET status = ?, notes = COALESCE(?, notes), updated_at = ? "
            "WHERE id = ?",
            (str(status), notes, encode_dt(utcnow()), claim_id),
        )

    def link_claims(self, claim_id: str, related_claim_id: str, relation: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO claim_links(claim_id, related_claim_id, relation, "
            "created_at) VALUES(?,?,?,?)",
            (claim_id, related_claim_id, relation, encode_dt(utcnow())),
        )


class EntityRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        investigation_id: str | None,
        entity_type: EntityType,
        name: str,
        *,
        identifiers: Sequence[EntityIdentifier] = (),
        aliases: Sequence[str] = (),
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Entity:
        entity = Entity(
            id=self.db.next_id(ENTITY),
            investigation_id=investigation_id,
            entity_type=entity_type,
            name=name,
            identifiers=list(identifiers),
            aliases=list(aliases),
            description=description,
            metadata=dict(metadata or {}),
        )
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO entities(id, investigation_id, entity_type, name, name_key, "
                "description, metadata, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    entity.id,
                    investigation_id,
                    str(entity_type),
                    name,
                    normalize_title(name),
                    description,
                    to_json(entity.metadata),
                    encode_dt(entity.created_at),
                    encode_dt(entity.updated_at),
                ),
            )
            self._write_identifiers(conn, entity)
            self._write_aliases(conn, entity.id, aliases)
        return entity

    def _write_identifiers(self, conn: Any, entity: Entity) -> None:
        conn.executemany(
            "INSERT OR IGNORE INTO entity_identifiers(entity_id, investigation_id, "
            "scheme, value) VALUES(?,?,?,?)",
            [
                (entity.id, entity.investigation_id, ident.scheme, ident.value)
                for ident in entity.identifiers
            ],
        )

    def _write_aliases(self, conn: Any, entity_id: str, aliases: Iterable[str]) -> None:
        conn.executemany(
            "INSERT OR IGNORE INTO entity_aliases(entity_id, alias, alias_key, source) "
            "VALUES(?,?,?,?)",
            [
                (entity_id, alias, normalize_title(alias), None)
                for alias in aliases
                if normalize_title(alias)
            ],
        )

    def get(self, entity_id: str) -> Entity:
        row = self.db.query_one("SELECT * FROM entities WHERE id = ?", (entity_id,))
        if row is None:
            raise NotFound(f"no such entity: {entity_id}")
        return self._hydrate(row)

    def _hydrate(self, row: Any) -> Entity:
        identifiers = [
            EntityIdentifier(scheme=ident["scheme"], value=ident["value"])
            for ident in self.db.query(
                "SELECT scheme, value FROM entity_identifiers WHERE entity_id = ? "
                "ORDER BY scheme",
                (row["id"],),
            )
        ]
        aliases = [
            alias["alias"]
            for alias in self.db.query(
                "SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY alias",
                (row["id"],),
            )
        ]
        return row_to_entity(row, identifiers, aliases)

    def find_by_identifier(
        self, investigation_id: str | None, scheme: str, value: str
    ) -> Entity | None:
        row = self.db.query_one(
            "SELECT e.* FROM entities e JOIN entity_identifiers i ON i.entity_id = e.id "
            "WHERE i.scheme = ? AND i.value = ? AND (e.investigation_id IS ? OR ? IS NULL) "
            "ORDER BY e.id LIMIT 1",
            (scheme, value, investigation_id, investigation_id),
        )
        return self._hydrate(row) if row else None

    def find_by_name(
        self,
        investigation_id: str | None,
        entity_type: EntityType,
        name: str,
        *,
        limit: int = 10,
    ) -> list[Entity]:
        name_key = normalize_title(name)
        if not name_key:
            return []
        rows = self.db.query(
            "SELECT DISTINCT e.* FROM entities e "
            "LEFT JOIN entity_aliases a ON a.entity_id = e.id "
            "WHERE e.investigation_id IS ? AND e.entity_type = ? "
            "AND (e.name_key = ? OR a.alias_key = ?) ORDER BY e.id LIMIT ?",
            (investigation_id, str(entity_type), name_key, name_key, limit),
        )
        return [self._hydrate(row) for row in rows]

    def add_identifier(self, entity_id: str, identifier: EntityIdentifier) -> None:
        investigation_id = self.db.scalar(
            "SELECT investigation_id FROM entities WHERE id = ?", (entity_id,)
        )
        self.db.execute(
            "INSERT OR IGNORE INTO entity_identifiers(entity_id, investigation_id, "
            "scheme, value) VALUES(?,?,?,?)",
            (entity_id, investigation_id, identifier.scheme, identifier.value),
        )

    def add_alias(self, entity_id: str, alias: str, *, source: str | None = None) -> None:
        alias_key = normalize_title(alias)
        if not alias_key:
            return
        self.db.execute(
            "INSERT OR IGNORE INTO entity_aliases(entity_id, alias, alias_key, source) "
            "VALUES(?,?,?,?)",
            (entity_id, alias, alias_key, source),
        )

    def update_metadata(self, entity_id: str, metadata: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE entities SET metadata = ?, updated_at = ? WHERE id = ?",
            (to_json(metadata), encode_dt(utcnow()), entity_id),
        )

    def list(self, investigation_id: str, *, limit: int = 200) -> list[Entity]:
        rows = self.db.query(
            "SELECT * FROM entities WHERE investigation_id = ? "
            "ORDER BY CAST(SUBSTR(id, INSTR(id, ':') + 1) AS INTEGER) LIMIT ?",
            (investigation_id, limit),
        )
        return [self._hydrate(row) for row in rows]


class EventRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        investigation_id: str,
        description: str,
        *,
        date_start: Any = None,
        date_end: Any = None,
        date_precision: DatePrecision = DatePrecision.UNKNOWN,
        entity_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        created_by_task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        event = Event(
            id=self.db.next_id(EVENT),
            investigation_id=investigation_id,
            description=description,
            date_start=date_start,
            date_end=date_end,
            date_precision=date_precision,
            entity_ids=list(entity_ids),
            evidence_ids=list(evidence_ids),
            created_by_task_id=created_by_task_id,
            metadata=dict(metadata or {}),
        )
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO events(id, investigation_id, description, date_start, "
                "date_end, date_precision, metadata, created_by_task_id, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    event.id,
                    investigation_id,
                    description,
                    encode_dt(event.date_start),
                    encode_dt(event.date_end),
                    str(date_precision),
                    to_json(event.metadata),
                    created_by_task_id,
                    encode_dt(event.created_at),
                ),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO event_entities(event_id, entity_id) VALUES(?,?)",
                [(event.id, entity_id) for entity_id in entity_ids],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO event_evidence(event_id, document_id) VALUES(?,?)",
                [(event.id, document_id) for document_id in evidence_ids],
            )
        return event

    def get(self, event_id: str) -> Event:
        row = self.db.query_one("SELECT * FROM events WHERE id = ?", (event_id,))
        if row is None:
            raise NotFound(f"no such event: {event_id}")
        return self._hydrate(row)

    def _hydrate(self, row: Any) -> Event:
        entity_ids = [
            item["entity_id"]
            for item in self.db.query(
                "SELECT entity_id FROM event_entities WHERE event_id = ? ORDER BY entity_id",
                (row["id"],),
            )
        ]
        evidence_ids = [
            item["document_id"]
            for item in self.db.query(
                "SELECT document_id FROM event_evidence WHERE event_id = ? "
                "ORDER BY document_id",
                (row["id"],),
            )
        ]
        return row_to_event(row, entity_ids, evidence_ids)

    def add_evidence(self, event_id: str, document_id: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO event_evidence(event_id, document_id) VALUES(?,?)",
            (event_id, document_id),
        )

    def timeline(self, investigation_id: str, *, limit: int = 200) -> list[Event]:
        """Events ordered by date; undated events sort last, never invented."""
        rows = self.db.query(
            "SELECT * FROM events WHERE investigation_id = ? "
            "ORDER BY date_start IS NULL, date_start, id LIMIT ?",
            (investigation_id, limit),
        )
        return [self._hydrate(row) for row in rows]


class RelationshipRepository:
    """Generic provenance-carrying edges (MENTIONS, ABOUT, PARTICIPATED_IN...)."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        investigation_id: str,
        *,
        subject_type: str,
        subject_id: str,
        predicate: str,
        object_type: str,
        object_id: str,
        evidence_document_id: str | None = None,
        created_by_task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        relationship_id = self.db.next_id(RELATIONSHIP)
        self.db.execute(
            "INSERT OR IGNORE INTO relationships(id, investigation_id, subject_type, "
            "subject_id, predicate, object_type, object_id, evidence_document_id, "
            "created_by_task_id, metadata, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                relationship_id,
                investigation_id,
                subject_type,
                subject_id,
                predicate,
                object_type,
                object_id,
                evidence_document_id,
                created_by_task_id,
                to_json(metadata or {}),
                encode_dt(utcnow()),
            ),
        )
        return relationship_id

    def for_subject(self, investigation_id: str, subject_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.query(
                "SELECT * FROM relationships WHERE investigation_id = ? AND subject_id = ? "
                "ORDER BY id",
                (investigation_id, subject_id),
            )
        ]

    def for_object(self, investigation_id: str, object_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.query(
                "SELECT * FROM relationships WHERE investigation_id = ? AND object_id = ? "
                "ORDER BY id",
                (investigation_id, object_id),
            )
        ]

    def list(self, investigation_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.query(
                "SELECT * FROM relationships WHERE investigation_id = ? ORDER BY id LIMIT ?",
                (investigation_id, limit),
            )
        ]


class CitationRepository:
    """Document-to-document citation edges.

    An edge can exist before its target is fetched: the cited work is
    recorded by DOI/provider id, and resolved to a document id if and when
    that work is acquired. That is what makes bounded traversal possible -
    the frontier is visible without fetching it.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        investigation_id: str | None,
        *,
        citing_document_id: str | None,
        provider: str,
        cited_document_id: str | None = None,
        citing_external_id: str | None = None,
        cited_external_id: str | None = None,
        cited_doi: str | None = None,
        cited_title: str | None = None,
        relation: str = "cites",
        depth: int = 0,
    ) -> str | None:
        """Record a citation edge, or return ``None`` if it was already held.

        Traversal revisits the same pair by different routes; the caller
        counts edges it actually added rather than edges it tried to add.
        """
        citation_id = self.db.next_id(CITATION)
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO citations(id, investigation_id, citing_document_id, "
            "cited_document_id, citing_external_id, cited_external_id, cited_doi, "
            "cited_title, relation, provider, depth, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                citation_id,
                investigation_id,
                citing_document_id,
                cited_document_id,
                citing_external_id,
                # SQLite treats NULLs as distinct in UNIQUE constraints, which
                # would defeat citation de-duplication; empty string instead.
                cited_external_id or "",
                cited_doi or "",
                cited_title,
                relation,
                provider,
                depth,
            ) + (encode_dt(utcnow()),),
        )
        return citation_id if cursor.rowcount else None

    def resolve_document(
        self,
        investigation_id: str | None,
        *,
        document_id: str,
        doi: str | None = None,
        external_id: str | None = None,
    ) -> int:
        """Attach a newly acquired document to citation edges pointing at it."""
        updated = 0
        if doi:
            cursor = self.db.execute(
                "UPDATE citations SET cited_document_id = ? WHERE investigation_id IS ? "
                "AND cited_document_id IS NULL AND cited_doi = ?",
                (document_id, investigation_id, doi),
            )
            updated += cursor.rowcount or 0
        if external_id:
            cursor = self.db.execute(
                "UPDATE citations SET cited_document_id = ? WHERE investigation_id IS ? "
                "AND cited_document_id IS NULL AND cited_external_id = ?",
                (document_id, investigation_id, external_id),
            )
            updated += cursor.rowcount or 0
        return updated

    def references_of(self, investigation_id: str | None, document_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.query(
                "SELECT * FROM citations WHERE investigation_id IS ? AND "
                "citing_document_id = ? AND relation = 'cites' ORDER BY id",
                (investigation_id, document_id),
            )
        ]

    def citing_of(self, investigation_id: str | None, document_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.query(
                "SELECT * FROM citations WHERE investigation_id IS ? AND "
                "cited_document_id = ? AND relation = 'cites' ORDER BY id",
                (investigation_id, document_id),
            )
        ]

    def count(self, investigation_id: str | None) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM citations WHERE investigation_id IS ?",
                (investigation_id,),
            )
            or 0
        )
