"""Repositories for core investigation state."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from research.errors import NotFound
from research.ids import EVIDENCE, FETCH, INVESTIGATION, QUERY, SNAPSHOT, TASK
from research.models.common import DuplicateRelation, SourceFamily, utcnow
from research.models.evidence import EvidenceDocument
from research.models.investigation import (
    Investigation,
    InvestigationStatus,
    StopReason,
)
from research.models.query import ResearchQuery
from research.models.task import ResearchTask, TaskResult, TaskStatus
from research.normalize.fingerprint import simhash_bands
from research.normalize.urls import url_identity_key
from research.storage.database import Database, encode_dt, from_json, to_json
from research.storage.fts import index_document
from research.storage.rows import (
    document_to_params,
    row_to_document,
    row_to_investigation,
    row_to_task,
)

_DOCUMENT_COLUMNS = (
    "id, investigation_id, source_type, source_family, provider, external_id, "
    "canonical_url, url_key, title, title_key, authors, published_at, fetched_at, "
    "text, abstract, metadata, content_hash, doi, canonical_host, publisher, "
    "language, simhash, duplicate_of, duplicate_relation, derived_from, "
    "independence_key, provenance"
)


class InvestigationRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        question: str,
        *,
        brief: str | None = None,
        budget: dict[str, Any] | None = None,
        tags: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> Investigation:
        investigation = Investigation(
            id=self.db.next_id(INVESTIGATION),
            question=question,
            brief=brief,
            budget=dict(budget or {}),
            tags=list(tags),
            metadata=dict(metadata or {}),
        )
        self.db.execute(
            "INSERT INTO investigations(id, question, brief, status, budget, tags, "
            "metadata, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                investigation.id,
                investigation.question,
                investigation.brief,
                str(investigation.status),
                to_json(investigation.budget),
                to_json(investigation.tags),
                to_json(investigation.metadata),
                encode_dt(investigation.created_at),
                encode_dt(investigation.updated_at),
            ),
        )
        return investigation

    def get(self, investigation_id: str) -> Investigation:
        row = self.db.query_one(
            "SELECT * FROM investigations WHERE id = ?", (investigation_id,)
        )
        if row is None:
            raise NotFound(f"no such investigation: {investigation_id}")
        return row_to_investigation(row)

    def list(self, *, limit: int = 50) -> list[Investigation]:
        rows = self.db.query(
            "SELECT * FROM investigations ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [row_to_investigation(row) for row in rows]

    def set_status(
        self,
        investigation_id: str,
        status: InvestigationStatus,
        *,
        stop_reason: StopReason | None = None,
        stop_detail: str | None = None,
    ) -> None:
        completed = status in (
            InvestigationStatus.COMPLETED,
            InvestigationStatus.FAILED,
        )
        self.db.execute(
            "UPDATE investigations SET status = ?, stop_reason = ?, stop_detail = ?, "
            "updated_at = ?, completed_at = CASE WHEN ? THEN ? ELSE completed_at END "
            "WHERE id = ?",
            (
                str(status),
                str(stop_reason) if stop_reason else None,
                stop_detail,
                encode_dt(utcnow()),
                1 if completed else 0,
                encode_dt(utcnow()),
                investigation_id,
            ),
        )

    def update_metadata(self, investigation_id: str, metadata: dict[str, Any]) -> None:
        """Merge keys into an investigation's metadata.

        Merging here rather than at each call site: every caller wanted it,
        each was doing it by hand, and one that forgot would silently drop
        whatever another had recorded.
        """
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT metadata FROM investigations WHERE id = ?", (investigation_id,)
            ).fetchone()
            existing = from_json(row["metadata"], {}) if row else {}
            connection.execute(
                "UPDATE investigations SET metadata = ?, updated_at = ? WHERE id = ?",
                (
                    to_json({**(existing or {}), **metadata}),
                    encode_dt(utcnow()),
                    investigation_id,
                ),
            )

    def set_brief(self, investigation_id: str, brief: str | None) -> None:
        """Replace the standing context the planner reads."""
        self.db.execute(
            "UPDATE investigations SET brief = ?, updated_at = ? WHERE id = ?",
            (brief, encode_dt(utcnow()), investigation_id),
        )


class TaskRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, task: ResearchTask) -> ResearchTask:
        if not task.id:
            task.id = self.db.next_id(TASK)
        self.db.execute(
            "INSERT INTO research_tasks(id, investigation_id, parent_task_id, role, "
            "operation, objective, status, depth, priority, parameters, result, error, "
            "created_at, started_at, finished_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                task.id,
                task.investigation_id,
                task.parent_task_id,
                str(task.role),
                str(task.operation),
                task.objective,
                str(task.status),
                task.depth,
                task.priority,
                to_json(task.parameters),
                to_json(task.result.to_dict()) if task.result else None,
                task.error,
                encode_dt(task.created_at),
                encode_dt(task.started_at),
                encode_dt(task.finished_at),
            ),
        )
        return task

    def get(self, task_id: str) -> ResearchTask:
        row = self.db.query_one("SELECT * FROM research_tasks WHERE id = ?", (task_id,))
        if row is None:
            raise NotFound(f"no such task: {task_id}")
        return row_to_task(row)

    def list(
        self,
        investigation_id: str,
        *,
        status: TaskStatus | None = None,
        limit: int = 200,
    ) -> list[ResearchTask]:
        sql = "SELECT * FROM research_tasks WHERE investigation_id = ?"
        params: list[Any] = [investigation_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(str(status))
        sql += " ORDER BY priority ASC, created_at ASC LIMIT ?"
        params.append(limit)
        return [row_to_task(row) for row in self.db.query(sql, tuple(params))]

    def children(self, task_id: str) -> list[ResearchTask]:
        rows = self.db.query(
            "SELECT * FROM research_tasks WHERE parent_task_id = ? ORDER BY created_at",
            (task_id,),
        )
        return [row_to_task(row) for row in rows]

    def claim_pending(self, investigation_id: str, *, limit: int = 1) -> list[ResearchTask]:
        """Take up to ``limit`` pending tasks and mark them running.

        Selection and marking happen together so that a concurrent batch
        cannot hand the same task to two workers. Ordering is the same as
        ``next_pending``: priority first, then creation order.
        """
        if limit <= 0:
            return []
        rows = self.db.query(
            "SELECT * FROM research_tasks WHERE investigation_id = ? AND status = ? "
            "ORDER BY priority ASC, created_at ASC, id ASC LIMIT ?",
            (investigation_id, str(TaskStatus.PENDING), int(limit)),
        )
        tasks = [row_to_task(row) for row in rows]
        for task in tasks:
            self.mark_running(task.id)
        return tasks

    def next_pending(self, investigation_id: str) -> ResearchTask | None:
        row = self.db.query_one(
            "SELECT * FROM research_tasks WHERE investigation_id = ? AND status = ? "
            "ORDER BY priority ASC, created_at ASC LIMIT 1",
            (investigation_id, str(TaskStatus.PENDING)),
        )
        return row_to_task(row) if row else None

    def mark_running(self, task_id: str) -> None:
        self.db.execute(
            "UPDATE research_tasks SET status = ?, started_at = ? WHERE id = ?",
            (str(TaskStatus.RUNNING), encode_dt(utcnow()), task_id),
        )

    def finish(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: TaskResult | None = None,
        error: str | None = None,
    ) -> None:
        self.db.execute(
            "UPDATE research_tasks SET status = ?, result = ?, error = ?, finished_at = ? "
            "WHERE id = ?",
            (
                str(status),
                to_json(result.to_dict()) if result else None,
                error,
                encode_dt(utcnow()),
                task_id,
            ),
        )

    def count(self, investigation_id: str) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM research_tasks WHERE investigation_id = ?",
                (investigation_id,),
            )
            or 0
        )


class DocumentRepository:
    """Persistence and lookup primitives for evidence.

    Deduplication *policy* lives in :mod:`research.acquisition.deduplicate`;
    this repository only offers the lookups that policy needs.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def new_id(self) -> str:
        return self.db.next_id(EVIDENCE)

    def add(self, document: EvidenceDocument) -> EvidenceDocument:
        if not document.id:
            document.id = self.new_id()
        if document.independence_key is None:
            document.independence_key = document.id
        params = document_to_params(document, url_identity_key(document.canonical_url))
        columns = ", ".join(params)
        placeholders = ", ".join(f":{key}" for key in params)
        with self.db.transaction() as conn:
            conn.execute(
                f"INSERT INTO documents({columns}, created_at) "
                f"VALUES({placeholders}, :created_at)",
                {**params, "created_at": encode_dt(utcnow())},
            )
            self._index(conn, document)
        return document

    def _index(self, conn: Any, document: EvidenceDocument) -> None:
        # Full-text index first: written in the same transaction as the row
        # it describes, so the two cannot fall out of step.
        index_document(conn, document)
        conn.executemany(
            "INSERT OR IGNORE INTO document_identities"
            "(document_id, investigation_id, scheme, value) VALUES(?,?,?,?)",
            [
                (document.id, document.investigation_id, scheme, value)
                for scheme, value in document.identity_keys()
            ],
        )
        url_key = url_identity_key(document.canonical_url)
        if url_key:
            conn.execute(
                "INSERT OR IGNORE INTO document_identities"
                "(document_id, investigation_id, scheme, value) VALUES(?,?,?,?)",
                (document.id, document.investigation_id, "url_key", url_key),
            )
        conn.executemany(
            "INSERT OR IGNORE INTO document_fingerprints"
            "(document_id, investigation_id, band) VALUES(?,?,?)",
            [
                (document.id, document.investigation_id, band)
                for band in simhash_bands(document.simhash)
            ],
        )

    def update(self, document: EvidenceDocument) -> EvidenceDocument:
        """Persist an enriched document (e.g. full text added after a fetch)."""
        params = document_to_params(document, url_identity_key(document.canonical_url))
        assignments = ", ".join(f"{key} = :{key}" for key in params if key != "id")
        with self.db.transaction() as conn:
            conn.execute(f"UPDATE documents SET {assignments} WHERE id = :id", params)
            conn.execute(
                "DELETE FROM document_fingerprints WHERE document_id = ?", (document.id,)
            )
            self._index(conn, document)
        return document

    def get(self, document_id: str) -> EvidenceDocument:
        row = self.db.query_one(
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE id = ?", (document_id,)
        )
        if row is None:
            raise NotFound(f"no such document: {document_id}")
        return row_to_document(row)

    def get_many(self, document_ids: Iterable[str]) -> list[EvidenceDocument]:
        ids = [doc_id for doc_id in document_ids]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.db.query(
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE id IN ({placeholders})",
            tuple(ids),
        )
        by_id = {row["id"]: row_to_document(row) for row in rows}
        return [by_id[doc_id] for doc_id in ids if doc_id in by_id]

    def list(
        self,
        investigation_id: str,
        *,
        family: SourceFamily | None = None,
        originals_only: bool = False,
        limit: int = 500,
        offset: int = 0,
    ) -> list[EvidenceDocument]:
        sql = f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE investigation_id = ?"
        params: list[Any] = [investigation_id]
        if family is not None:
            sql += " AND source_family = ?"
            params.append(str(family))
        if originals_only:
            sql += " AND duplicate_of IS NULL"
        sql += " ORDER BY CAST(SUBSTR(id, INSTR(id, ':') + 1) AS INTEGER) LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return [row_to_document(row) for row in self.db.query(sql, tuple(params))]

    def find_by_identity(
        self, investigation_id: str | None, scheme: str, value: str
    ) -> str | None:
        """Return the id of an existing document carrying this identity key."""
        return self.db.scalar(
            "SELECT document_id FROM document_identities WHERE "
            "(investigation_id IS ? OR ? IS NULL) AND scheme = ? AND value = ? "
            "ORDER BY document_id LIMIT 1",
            (investigation_id, investigation_id, scheme, value),
        )

    def find_by_bands(
        self, investigation_id: str | None, bands: Sequence[str], *, exclude: str | None = None
    ) -> list[str]:
        if not bands:
            return []
        placeholders = ",".join("?" for _ in bands)
        sql = (
            "SELECT DISTINCT document_id FROM document_fingerprints "
            f"WHERE investigation_id IS ? AND band IN ({placeholders})"
        )
        params: list[Any] = [investigation_id, *bands]
        if exclude:
            sql += " AND document_id != ?"
            params.append(exclude)
        return [row["document_id"] for row in self.db.query(sql, tuple(params))]

    def mark_duplicate(
        self,
        document_id: str,
        *,
        duplicate_of: str,
        relation: DuplicateRelation,
        independence_key: str | None = None,
    ) -> None:
        self.db.execute(
            "UPDATE documents SET duplicate_of = ?, duplicate_relation = ?, "
            "independence_key = COALESCE(?, independence_key) WHERE id = ?",
            (duplicate_of, str(relation), independence_key, document_id),
        )

    def set_derived_from(self, document_id: str, parent_document_id: str) -> None:
        self.db.execute(
            "UPDATE documents SET derived_from = ? WHERE id = ?",
            (parent_document_id, document_id),
        )

    def count(
        self,
        investigation_id: str,
        *,
        family: SourceFamily | None = None,
        originals_only: bool = False,
    ) -> int:
        sql = "SELECT COUNT(*) FROM documents WHERE investigation_id = ?"
        params: list[Any] = [investigation_id]
        if family is not None:
            sql += " AND source_family = ?"
            params.append(str(family))
        if originals_only:
            sql += " AND duplicate_of IS NULL"
        return int(self.db.scalar(sql, tuple(params)) or 0)

    def independence_groups(self, document_ids: Sequence[str]) -> dict[str, list[str]]:
        """Group documents by independence key.

        Two documents in the same group are one piece of evidence, however
        many outlets carried it.
        """
        if not document_ids:
            return {}
        placeholders = ",".join("?" for _ in document_ids)
        rows = self.db.query(
            f"SELECT id, COALESCE(independence_key, id) AS key FROM documents "
            f"WHERE id IN ({placeholders})",
            tuple(document_ids),
        )
        groups: dict[str, list[str]] = {}
        for row in rows:
            groups.setdefault(row["key"], []).append(row["id"])
        return groups

    def duplicates_of(self, document_id: str) -> list[EvidenceDocument]:
        rows = self.db.query(
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE duplicate_of = ? "
            "ORDER BY id",
            (document_id,),
        )
        return [row_to_document(row) for row in rows]


class SearchQueryLog:
    """Every provider search is recorded: what was asked, and why."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def record(
        self,
        query: ResearchQuery,
        *,
        provider: str,
        family: SourceFamily,
        status: str,
        result_count: int = 0,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> str:
        query_id = self.db.next_id(QUERY)
        self.db.execute(
            "INSERT INTO search_queries(id, investigation_id, task_id, provider, family, "
            "query_text, objective, filters, max_results, result_count, status, error, "
            "duration_ms, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                query_id,
                query.investigation_id,
                query.task_id,
                provider,
                str(family),
                query.text,
                query.objective,
                to_json(query.filters),
                query.limit,
                result_count,
                status,
                error,
                duration_ms,
                encode_dt(utcnow()),
            ),
        )
        return query_id

    def list(self, investigation_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM search_queries WHERE investigation_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (investigation_id, limit),
        )
        return [dict(row) for row in rows]

    def consecutive_failures(self, investigation_id: str, provider: str) -> int:
        """How many times in a row this provider has failed, most recent first.

        Durable rather than in-process: a provider that is refusing this
        investigation is still refusing it after a restart, and the query log
        is where that fact already lives.
        """
        rows = self.db.query(
            "SELECT status FROM search_queries WHERE investigation_id = ? AND provider = ? "
            "ORDER BY id DESC LIMIT 20",
            (investigation_id, provider),
        )
        failures = 0
        for row in rows:
            if row["status"] == "ok":
                break
            failures += 1
        return failures

    def count(self, investigation_id: str) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM search_queries WHERE investigation_id = ?",
                (investigation_id,),
            )
            or 0
        )


class SourceFetchLog:
    """Every outbound document fetch is recorded, successes and failures alike."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def record(
        self,
        *,
        provider: str,
        url: str,
        investigation_id: str | None = None,
        task_id: str | None = None,
        final_url: str | None = None,
        status_code: int | None = None,
        ok: bool = False,
        error: str | None = None,
        content_type: str | None = None,
        num_bytes: int | None = None,
        duration_ms: int | None = None,
        document_id: str | None = None,
    ) -> str:
        fetch_id = self.db.next_id(FETCH)
        self.db.execute(
            "INSERT INTO source_fetches(id, investigation_id, task_id, provider, url, "
            "final_url, status_code, ok, error, content_type, bytes, duration_ms, "
            "document_id, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                fetch_id,
                investigation_id,
                task_id,
                provider,
                url,
                final_url,
                status_code,
                1 if ok else 0,
                error,
                content_type,
                num_bytes,
                duration_ms,
                document_id,
                encode_dt(utcnow()),
            ),
        )
        return fetch_id

    def begin(
        self,
        *,
        provider: str,
        url: str,
        investigation_id: str | None = None,
        task_id: str | None = None,
    ) -> str:
        """Open a fetch record before the call is made.

        The row exists whether or not the request succeeds, so a failed or
        hung provider leaves a trace instead of a gap.
        """
        return self.record(
            provider=provider,
            url=url,
            investigation_id=investigation_id,
            task_id=task_id,
            ok=False,
            error="in flight",
        )

    def complete(
        self,
        fetch_id: str,
        *,
        ok: bool,
        final_url: str | None = None,
        status_code: int | None = None,
        error: str | None = None,
        content_type: str | None = None,
        num_bytes: int | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.db.execute(
            "UPDATE source_fetches SET ok = ?, final_url = ?, status_code = ?, "
            "error = ?, content_type = ?, bytes = ?, duration_ms = ? WHERE id = ?",
            (
                1 if ok else 0,
                final_url,
                status_code,
                error,
                content_type,
                num_bytes,
                duration_ms,
                fetch_id,
            ),
        )

    def attach_document(self, fetch_id: str, document_id: str) -> None:
        self.db.execute(
            "UPDATE source_fetches SET document_id = ? WHERE id = ?",
            (document_id, fetch_id),
        )

    def list(self, investigation_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM source_fetches WHERE investigation_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (investigation_id, limit),
        )
        return [dict(row) for row in rows]

    def failure_count(self, investigation_id: str) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM source_fetches WHERE investigation_id = ? AND ok = 0",
                (investigation_id,),
            )
            or 0
        )


class BudgetRepository:
    """Durable budget counters, so a resumed run cannot start spending afresh."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def usage(self, investigation_id: str) -> dict[str, float]:
        rows = self.db.query(
            "SELECT resource, used FROM budget_usage WHERE investigation_id = ?",
            (investigation_id,),
        )
        return {row["resource"]: float(row["used"]) for row in rows}

    def add(self, investigation_id: str, resource: str, amount: float = 1) -> float:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO budget_usage(investigation_id, resource, used) VALUES(?,?,?) "
                "ON CONFLICT(investigation_id, resource) DO UPDATE SET used = used + ?",
                (investigation_id, resource, amount, amount),
            )
            row = conn.execute(
                "SELECT used FROM budget_usage WHERE investigation_id = ? AND resource = ?",
                (investigation_id, resource),
            ).fetchone()
        return float(row["used"]) if row else 0.0

    def get(self, investigation_id: str, resource: str) -> float:
        return float(
            self.db.scalar(
                "SELECT used FROM budget_usage WHERE investigation_id = ? AND resource = ?",
                (investigation_id, resource),
            )
            or 0.0
        )

    def reset(self, investigation_id: str) -> None:
        self.db.execute(
            "DELETE FROM budget_usage WHERE investigation_id = ?", (investigation_id,)
        )

class SnapshotRepository:
    """Stored states of an investigation, for comparing with later ones.

    Deals in plain payloads rather than a snapshot type: what a snapshot
    *means* is a question about claims and evidence, which is decided a
    layer up. Storage only has to keep it and give it back unchanged.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def save(
        self, investigation_id: str, payload: dict[str, Any], *, label: str | None = None
    ) -> str:
        snapshot_id = self.db.next_id(SNAPSHOT)
        self.db.execute(
            "INSERT INTO snapshots(id, investigation_id, label, taken_at, payload) "
            "VALUES(?,?,?,?,?)",
            (snapshot_id, investigation_id, label, encode_dt(utcnow()), to_json(payload)),
        )
        return snapshot_id

    def get(self, snapshot_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,))
        if row is None:
            raise NotFound(f"no snapshot {snapshot_id}")
        return self._read(row)

    def latest(self, investigation_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM snapshots WHERE investigation_id = ? "
            "ORDER BY taken_at DESC, id DESC LIMIT 1",
            (investigation_id,),
        )
        return self._read(row) if row else None

    def list(self, investigation_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM snapshots WHERE investigation_id = ? "
            "ORDER BY taken_at DESC, id DESC LIMIT ?",
            (investigation_id, limit),
        )
        return [self._read(row) for row in rows]

    def count(self, investigation_id: str) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM snapshots WHERE investigation_id = ?",
                (investigation_id,),
            )
            or 0
        )

    @staticmethod
    def _read(row: Any) -> dict[str, Any]:
        payload = from_json(row["payload"], {}) or {}
        payload["snapshot_id"] = row["id"]
        payload["label"] = row["label"]
        payload["taken_at"] = row["taken_at"]
        return payload
