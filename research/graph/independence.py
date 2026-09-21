"""Counting sources rather than documents.

Deduplication decides *whether* two documents are independent; this decides
what follows from that. It lives below the acquisition layer because
everything that reports on evidence - claim assessment, timelines, the CLI -
needs it, and none of those should have to depend on the machinery that
persists documents.
"""

from __future__ import annotations

from typing import Sequence

from research.storage.repositories import DocumentRepository


def independent_documents(
    documents: DocumentRepository, document_ids: Sequence[str]
) -> list[list[str]]:
    """Partition documents into independence groups.

    Each returned group counts as *one* source, whatever its size. Anything
    that says "two independent sources" in a report must count groups here,
    not rows in the document table.
    """
    groups = documents.independence_groups(document_ids)
    return [sorted(members, key=_id_sort_key) for _, members in sorted(groups.items())]


def count_independent(
    documents: DocumentRepository, document_ids: Sequence[str]
) -> int:
    return len(independent_documents(documents, document_ids))


def _id_sort_key(document_id: str) -> tuple[str, int]:
    prefix, _, number = document_id.partition(":")
    return (prefix, int(number) if number.isdigit() else 0)
