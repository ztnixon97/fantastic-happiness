"""SQLite-backed investigation store.

Investigation state lives here, outside any model context. A resumed run
reads tasks, evidence and budget from this store; it never replays a
conversation to work out what is known.

Access is synchronous. The store is local, single-process and fast relative
to network acquisition, so the complexity of an async driver buys nothing at
this stage; the repository interface is the seam to change if that stops
being true.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from research.errors import StorageError

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS id_sequences (
    prefix     TEXT PRIMARY KEY,
    next_value INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS investigations (
    id           TEXT PRIMARY KEY,
    question     TEXT NOT NULL,
    brief        TEXT,
    status       TEXT NOT NULL,
    stop_reason  TEXT,
    stop_detail  TEXT,
    budget       TEXT NOT NULL DEFAULT '{}',
    tags         TEXT NOT NULL DEFAULT '[]',
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS research_tasks (
    id               TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    parent_task_id   TEXT REFERENCES research_tasks(id) ON DELETE SET NULL,
    role             TEXT NOT NULL,
    operation        TEXT NOT NULL,
    objective        TEXT NOT NULL,
    status           TEXT NOT NULL,
    depth            INTEGER NOT NULL DEFAULT 0,
    priority         INTEGER NOT NULL DEFAULT 5,
    parameters       TEXT NOT NULL DEFAULT '{}',
    result           TEXT,
    error            TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_investigation ON research_tasks(investigation_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON research_tasks(parent_task_id);

CREATE TABLE IF NOT EXISTS documents (
    id                 TEXT PRIMARY KEY,
    investigation_id   TEXT REFERENCES investigations(id) ON DELETE CASCADE,
    source_type        TEXT NOT NULL,
    source_family      TEXT NOT NULL,
    provider           TEXT NOT NULL,
    external_id        TEXT,
    canonical_url      TEXT,
    url_key            TEXT,
    title              TEXT,
    title_key          TEXT,
    authors            TEXT NOT NULL DEFAULT '[]',
    published_at       TEXT,
    fetched_at         TEXT NOT NULL,
    text               TEXT,
    abstract           TEXT,
    metadata           TEXT NOT NULL DEFAULT '{}',
    content_hash       TEXT NOT NULL,
    doi                TEXT,
    canonical_host     TEXT,
    publisher          TEXT,
    language           TEXT,
    simhash            TEXT,  -- 64-bit unsigned, hex: SQLite INTEGER is signed
    duplicate_of       TEXT REFERENCES documents(id) ON DELETE SET NULL,
    duplicate_relation TEXT,
    derived_from       TEXT REFERENCES documents(id) ON DELETE SET NULL,
    independence_key   TEXT,
    provenance         TEXT NOT NULL DEFAULT '{}',
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_investigation ON documents(investigation_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(investigation_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(investigation_id, doi);
CREATE INDEX IF NOT EXISTS idx_documents_urlkey ON documents(investigation_id, url_key);
CREATE INDEX IF NOT EXISTS idx_documents_independence
    ON documents(investigation_id, independence_key);

-- Exact-identity index used by deduplication (doi, provider id, url, hash...).
CREATE TABLE IF NOT EXISTS document_identities (
    document_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    investigation_id TEXT,
    scheme           TEXT NOT NULL,
    value            TEXT NOT NULL,
    PRIMARY KEY (document_id, scheme, value)
);
CREATE INDEX IF NOT EXISTS idx_identities_lookup
    ON document_identities(investigation_id, scheme, value);

-- SimHash bands: near-duplicate candidate retrieval without a full scan.
CREATE TABLE IF NOT EXISTS document_fingerprints (
    document_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    investigation_id TEXT,
    band             TEXT NOT NULL,
    PRIMARY KEY (document_id, band)
);
CREATE INDEX IF NOT EXISTS idx_fingerprints_band
    ON document_fingerprints(investigation_id, band);

CREATE TABLE IF NOT EXISTS claims (
    id                 TEXT PRIMARY KEY,
    investigation_id   TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    text               TEXT NOT NULL,
    status             TEXT NOT NULL,
    notes              TEXT,
    metadata           TEXT NOT NULL DEFAULT '{}',
    created_by_task_id TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_investigation ON claims(investigation_id, status);

CREATE TABLE IF NOT EXISTS claim_evidence (
    id                 TEXT PRIMARY KEY,
    claim_id           TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    document_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    stance             TEXT NOT NULL,
    excerpt            TEXT,
    analysis           TEXT,
    created_by_task_id TEXT,
    created_at         TEXT NOT NULL,
    UNIQUE (claim_id, document_id, stance)
);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_claim ON claim_evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_document ON claim_evidence(document_id);

CREATE TABLE IF NOT EXISTS claim_links (
    claim_id         TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    related_claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    relation         TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (claim_id, related_claim_id, relation)
);

CREATE TABLE IF NOT EXISTS entities (
    id               TEXT PRIMARY KEY,
    investigation_id TEXT REFERENCES investigations(id) ON DELETE CASCADE,
    entity_type      TEXT NOT NULL,
    name             TEXT NOT NULL,
    name_key         TEXT NOT NULL,
    description      TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_namekey
    ON entities(investigation_id, entity_type, name_key);

CREATE TABLE IF NOT EXISTS entity_identifiers (
    entity_id        TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    investigation_id TEXT,
    scheme           TEXT NOT NULL,
    value            TEXT NOT NULL,
    PRIMARY KEY (entity_id, scheme, value)
);
CREATE INDEX IF NOT EXISTS idx_entity_identifiers
    ON entity_identifiers(investigation_id, scheme, value);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id  TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias      TEXT NOT NULL,
    alias_key  TEXT NOT NULL,
    source     TEXT,
    PRIMARY KEY (entity_id, alias_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases ON entity_aliases(alias_key);

CREATE TABLE IF NOT EXISTS events (
    id                 TEXT PRIMARY KEY,
    investigation_id   TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    description        TEXT NOT NULL,
    date_start         TEXT,
    date_end           TEXT,
    date_precision     TEXT NOT NULL,
    metadata           TEXT NOT NULL DEFAULT '{}',
    created_by_task_id TEXT,
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_investigation ON events(investigation_id, date_start);

CREATE TABLE IF NOT EXISTS event_entities (
    event_id  TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (event_id, entity_id)
);

CREATE TABLE IF NOT EXISTS event_evidence (
    event_id    TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    PRIMARY KEY (event_id, document_id)
);

-- Generic provenance-carrying graph edge (MENTIONS, ABOUT, PARTICIPATED_IN...).
CREATE TABLE IF NOT EXISTS relationships (
    id                   TEXT PRIMARY KEY,
    investigation_id     TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    subject_type         TEXT NOT NULL,
    subject_id           TEXT NOT NULL,
    predicate            TEXT NOT NULL,
    object_type          TEXT NOT NULL,
    object_id            TEXT NOT NULL,
    evidence_document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
    created_by_task_id   TEXT,
    metadata             TEXT NOT NULL DEFAULT '{}',
    created_at           TEXT NOT NULL,
    UNIQUE (investigation_id, subject_id, predicate, object_id, evidence_document_id)
);
CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(investigation_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(investigation_id, object_id);

CREATE TABLE IF NOT EXISTS citations (
    id                  TEXT PRIMARY KEY,
    investigation_id    TEXT REFERENCES investigations(id) ON DELETE CASCADE,
    citing_document_id  TEXT REFERENCES documents(id) ON DELETE CASCADE,
    cited_document_id   TEXT REFERENCES documents(id) ON DELETE SET NULL,
    citing_external_id  TEXT,
    cited_external_id   TEXT,
    cited_doi           TEXT,
    cited_title         TEXT,
    relation            TEXT NOT NULL DEFAULT 'cites',
    provider            TEXT NOT NULL,
    depth               INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    UNIQUE (investigation_id, citing_document_id, cited_external_id, cited_doi, relation)
);
CREATE INDEX IF NOT EXISTS idx_citations_citing ON citations(investigation_id, citing_document_id);
CREATE INDEX IF NOT EXISTS idx_citations_cited ON citations(investigation_id, cited_document_id);

CREATE TABLE IF NOT EXISTS search_queries (
    id               TEXT PRIMARY KEY,
    investigation_id TEXT REFERENCES investigations(id) ON DELETE CASCADE,
    task_id          TEXT,
    provider         TEXT NOT NULL,
    family           TEXT NOT NULL,
    query_text       TEXT NOT NULL,
    objective        TEXT,
    filters          TEXT NOT NULL DEFAULT '{}',
    max_results      INTEGER NOT NULL DEFAULT 0,
    result_count     INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL,
    error            TEXT,
    duration_ms      INTEGER,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_queries_investigation
    ON search_queries(investigation_id, created_at);

CREATE TABLE IF NOT EXISTS source_fetches (
    id               TEXT PRIMARY KEY,
    investigation_id TEXT REFERENCES investigations(id) ON DELETE CASCADE,
    task_id          TEXT,
    provider         TEXT NOT NULL,
    url              TEXT NOT NULL,
    final_url        TEXT,
    status_code      INTEGER,
    ok               INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    content_type     TEXT,
    bytes            INTEGER,
    duration_ms      INTEGER,
    document_id      TEXT REFERENCES documents(id) ON DELETE SET NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fetches_investigation
    ON source_fetches(investigation_id, created_at);

-- Full-text index over held evidence. A separate table rather than an
-- external-content one: the store is small, and keeping it standalone means
-- a corrupt or missing index can be rebuilt from the documents table without
-- touching either.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    document_id UNINDEXED,
    investigation_id UNINDEXED,
    title,
    abstract,
    body,
    tokenize = 'porter unicode61'
);

-- Optional and off by default: vectors are only worth their cost once
-- lexical retrieval is demonstrably failing.
CREATE TABLE IF NOT EXISTS document_embeddings (
    document_id      TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    investigation_id TEXT,
    model            TEXT NOT NULL,
    dimensions       INTEGER NOT NULL,
    vector           BLOB NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_investigation
    ON document_embeddings(investigation_id, model);

CREATE TABLE IF NOT EXISTS budget_usage (
    investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    resource         TEXT NOT NULL,
    used             REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (investigation_id, resource)
);
"""


def to_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot serialise {type(value)!r} for storage")


def from_json(raw: str | None, default: Any = None) -> Any:
    if raw in (None, ""):
        return {} if default is None else default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt row
        raise StorageError(f"corrupt JSON column: {exc}") from exc


def encode_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def decode_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class Database:
    """Thin connection/transaction manager over SQLite."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._depth = 0
        self.migrate()

    # -- lifecycle ------------------------------------------------------
    def migrate(self) -> None:
        # executescript() commits any open transaction, so it runs outside the
        # transaction helper rather than inside it.
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )
        self._backfill_index()

    def _backfill_index(self) -> None:
        """Populate the full-text index for a store written before it existed.

        ``CREATE TABLE IF NOT EXISTS`` gives an older database the new table
        but not its contents, and an empty index is worse than no index: it
        answers every search with nothing at all. The index is derived from
        the documents table, so it can simply be built.
        """
        with self._lock:
            has_documents = self._conn.execute("SELECT 1 FROM documents LIMIT 1").fetchone()
            if not has_documents:
                return
            if self._conn.execute("SELECT 1 FROM documents_fts LIMIT 1").fetchone():
                return
        from research.storage import fts

        fts.rebuild(self)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- access ---------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Reentrant transaction; nested uses join the outermost one."""
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield self._conn
                finally:
                    self._depth -= 1
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
            finally:
                self._depth = 0

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        return list(self.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        return self.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: tuple | dict = ()) -> Any:
        row = self.query_one(sql, params)
        return row[0] if row else None

    # -- identifiers ----------------------------------------------------
    def next_id(self, prefix: str) -> str:
        """Allocate the next stable identifier for ``prefix``.

        Sequential rather than random: ``evidence:42`` is something a person
        can hold in their head while reading a report and type back into the
        CLI to inspect it.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT next_value FROM id_sequences WHERE prefix = ?", (prefix,)
            ).fetchone()
            value = int(row["next_value"]) if row else 1
            conn.execute(
                "INSERT INTO id_sequences(prefix, next_value) VALUES(?, ?) "
                "ON CONFLICT(prefix) DO UPDATE SET next_value = excluded.next_value",
                (prefix, value + 1),
            )
        return f"{prefix}:{value}"
