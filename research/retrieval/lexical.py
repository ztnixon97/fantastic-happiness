"""Full-text retrieval over held evidence.

SQLite's FTS5 with BM25 ranking and a porter stemmer. No index server, no
embedding model, no new dependency - which is the point: the brief's rule is
that vector search waits until lexical retrieval is demonstrably failing, and
a corpus of a few hundred documents is not where that happens.

The index is written by the document repository as documents are stored, so
it cannot drift from the corpus. It can also be rebuilt from the documents
table at any time, which is what makes it safe to treat as a cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from research.normalize.text import truncate
from research.storage import fts
from research.storage.database import Database

#: FTS5 treats these as syntax. A research query is prose, not a query
#: language, so they are stripped rather than passed through - a stray quote
#: would otherwise turn a search into a syntax error.
_FTS_SPECIAL = re.compile(r'["*():^{}\[\]~\\-]')
_TOKEN = re.compile(r"[\w][\w\'-]*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class LexicalHit:
    document_id: str
    score: float
    snippet: str


def prepare_query(text: str, *, mode: str = "any") -> str:
    """Turn prose into an FTS5 query.

    ``any`` (the default) matches documents containing any of the terms and
    lets BM25 rank them, which is what recall wants. ``all`` requires every
    term, for when a caller knows what it is looking for.
    """
    candidates = _TOKEN.findall(_FTS_SPECIAL.sub(" ", text or ""))
    tokens = [token for token in candidates if len(token) > 1]
    if not tokens:
        return ""
    quoted = [f'"{token}"' for token in tokens]
    return " AND ".join(quoted) if mode == "all" else " OR ".join(quoted)


class LexicalIndex:
    """Maintains and queries the full-text index."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # -- maintenance ----------------------------------------------------
    # The index is written by the document repository as documents are
    # stored; these are here so a caller can repair or inspect it.
    def rebuild(self, investigation_id: str | None = None) -> int:
        return fts.rebuild(self.db, investigation_id)

    def count(self, investigation_id: str | None = None) -> int:
        return fts.count(self.db, investigation_id)

    def remove(self, document_id: str) -> None:
        fts.remove_document(self.db, document_id)

    # -- querying -------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        investigation_id: str | None = None,
        limit: int = 20,
        mode: str = "any",
    ) -> list[LexicalHit]:
        """Rank held documents against a query by BM25.

        Title and abstract are weighted above the body: a paper whose title is
        about the subject is more on-point than one that mentions it in
        passing, and BM25 alone does not know that.
        """
        match = prepare_query(query, mode=mode)
        if not match:
            return []
        sql = (
            "SELECT document_id, "
            "  bm25(documents_fts, 0.0, 0.0, 8.0, 4.0, 1.0) AS score, "
            "  snippet(documents_fts, 4, '', '', '…', 14) AS snippet "
            "FROM documents_fts WHERE documents_fts MATCH ?"
        )
        params: list[Any] = [match]
        if investigation_id is not None:
            sql += " AND investigation_id = ?"
            params.append(investigation_id)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        try:
            rows = self.db.query(sql, tuple(params))
        except Exception:
            # A malformed query is a caller error, not a crash: fall back to
            # the terms alone rather than failing the research step.
            rows = self.db.query(
                sql, tuple([prepare_query(query, mode="any"), *params[1:]])
            )
        return [
            LexicalHit(
                document_id=row["document_id"],
                # BM25 returns a negative score where lower is better; flip it
                # so every retriever here agrees that higher means better.
                score=-float(row["score"]),
                snippet=truncate(str(row["snippet"] or "").strip(), 300),
            )
            for row in rows
        ]
