"""Maintenance of the full-text index.

Writing the index is a storage concern: it is a derived view of the documents
table, written in the same transaction as the row it describes so the two
cannot disagree. Querying it is a retrieval concern and lives in
:mod:`research.retrieval.lexical`.

Keeping the halves apart is what stops storage from depending on retrieval,
which would make the document repository depend on the thing that searches it.
"""

from __future__ import annotations

from typing import Any

from research.storage.database import Database


def index_document(connection: Any, document: Any) -> None:
    """Write one document into the index, replacing any earlier copy."""
    connection.execute("DELETE FROM documents_fts WHERE document_id = ?", (document.id,))
    connection.execute(
        "INSERT INTO documents_fts(document_id, investigation_id, title, abstract, body) "
        "VALUES(?,?,?,?,?)",
        (
            document.id,
            document.investigation_id,
            document.title or "",
            document.abstract or "",
            document.text or "",
        ),
    )


def remove_document(db: Database, document_id: str) -> None:
    db.execute("DELETE FROM documents_fts WHERE document_id = ?", (document_id,))


def rebuild(db: Database, investigation_id: str | None = None) -> int:
    """Rebuild the index from the documents table.

    Needed for stores written before the index existed, and useful whenever a
    caller wants to be certain the two agree. The index is a cache: it is
    always safe to throw away and rebuild.
    """
    sql = "SELECT id, investigation_id, title, abstract, text FROM documents"
    params: tuple[Any, ...] = ()
    if investigation_id is not None:
        sql += " WHERE investigation_id = ?"
        params = (investigation_id,)
    rows = db.query(sql, params)
    with db.transaction() as connection:
        if investigation_id is None:
            connection.execute("DELETE FROM documents_fts")
        else:
            connection.execute(
                "DELETE FROM documents_fts WHERE investigation_id = ?", (investigation_id,)
            )
        connection.executemany(
            "INSERT INTO documents_fts(document_id, investigation_id, title, abstract, body) "
            "VALUES(?,?,?,?,?)",
            [
                (
                    row["id"],
                    row["investigation_id"],
                    row["title"] or "",
                    row["abstract"] or "",
                    row["text"] or "",
                )
                for row in rows
            ],
        )
    return len(rows)


def count(db: Database, investigation_id: str | None = None) -> int:
    if investigation_id is None:
        return int(db.scalar("SELECT COUNT(*) FROM documents_fts") or 0)
    return int(
        db.scalar(
            "SELECT COUNT(*) FROM documents_fts WHERE investigation_id = ?",
            (investigation_id,),
        )
        or 0
    )
