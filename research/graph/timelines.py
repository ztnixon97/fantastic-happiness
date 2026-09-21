"""Evidence-backed timelines.

Two kinds of entry, kept apart because they mean different things:

``event``
    Something that happened, asserted by a research worker and backed by at
    least one document.
``publication``
    When material entered the record. Derived deterministically from the
    corpus, one entry per independent source, so ten copies of a wire story
    do not become ten points on a chronology.

Undated material sorts last and is labelled undated. A timeline never invents
a date to make itself look complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from research.graph.independence import independent_documents
from research.graph.claims import PRIMARY_SOURCE_TYPES
from research.models.event import DatePrecision, Event
from research.normalize.text import truncate
from research.storage.store import ResearchStore


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    date: datetime | None
    precision: DatePrecision
    description: str
    kind: str
    evidence_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    event_id: str | None = None
    #: Independent sources behind this entry, after copies are collapsed.
    independent_sources: int = 0
    has_primary_source: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.date().isoformat() if self.date else None,
            "precision": str(self.precision),
            "description": self.description,
            "kind": self.kind,
            "evidence": list(self.evidence_ids),
            "entities": list(self.entity_ids),
            "event_id": self.event_id,
            "independent_sources": self.independent_sources,
            "primary_source": self.has_primary_source,
        }


class Timeline:
    """Builds a chronology from stored events and the corpus itself."""

    def __init__(self, store: ResearchStore, investigation_id: str) -> None:
        self.store = store
        self.investigation_id = investigation_id

    def build(
        self,
        *,
        include_publications: bool = False,
        limit: int = 200,
    ) -> list[TimelineEntry]:
        entries = [self._event_entry(event) for event in self.store.events.timeline(
            self.investigation_id, limit=limit
        )]
        if include_publications:
            entries.extend(self._publication_entries(limit=limit))
        # Undated entries keep their place at the end rather than being
        # dropped: "we do not know when this happened" is information.
        return sorted(entries, key=lambda entry: (entry.date is None, entry.date or datetime.min))

    def _event_entry(self, event: Event) -> TimelineEntry:
        documents = self.store.documents.get_many(event.evidence_ids)
        groups = independent_documents(self.store.documents, [d.id for d in documents])
        return TimelineEntry(
            date=event.date_start,
            precision=event.date_precision,
            description=event.description,
            kind="event",
            evidence_ids=list(event.evidence_ids),
            entity_ids=list(event.entity_ids),
            event_id=event.id,
            independent_sources=len(groups),
            has_primary_source=any(
                document.source_type in PRIMARY_SOURCE_TYPES for document in documents
            ),
        )

    def _publication_entries(self, *, limit: int) -> list[TimelineEntry]:
        """One entry per independent source, dated by its earliest copy."""
        documents = [
            document
            for document in self.store.documents.list(self.investigation_id, limit=1000)
            if document.published_at
        ]
        by_group: dict[str, list[Any]] = {}
        for document in documents:
            by_group.setdefault(document.independence_key or document.id, []).append(document)

        entries: list[TimelineEntry] = []
        for members in by_group.values():
            first = min(members, key=lambda document: document.published_at)
            copies = len(members) - 1
            label = truncate(first.title or first.canonical_url or first.id, 90)
            if copies:
                label += f" (+{copies} further cop{'y' if copies == 1 else 'ies'})"
            entries.append(
                TimelineEntry(
                    date=first.published_at,
                    precision=DatePrecision.DAY,
                    description=label,
                    kind="publication",
                    evidence_ids=[document.id for document in members],
                    independent_sources=1,
                    has_primary_source=first.source_type in PRIMARY_SOURCE_TYPES,
                )
            )
        entries.sort(key=lambda entry: entry.date or datetime.min)
        return entries[:limit]

    def first_appearance(self, document_id: str) -> TimelineEntry | None:
        """When the story behind a document first entered the record.

        Follows the independence group rather than the document: the answer
        for a syndicated copy is the date the original ran.
        """
        document = self.store.documents.get(document_id)
        group_key = document.independence_key or document.id
        members = [
            candidate
            for candidate in self.store.documents.list(self.investigation_id, limit=1000)
            if (candidate.independence_key or candidate.id) == group_key
            and candidate.published_at
        ]
        if not members:
            return None
        first = min(members, key=lambda candidate: candidate.published_at)
        return TimelineEntry(
            date=first.published_at,
            precision=DatePrecision.DAY,
            description=truncate(first.title or first.id, 90),
            kind="publication",
            evidence_ids=[first.id],
            independent_sources=1,
            has_primary_source=first.source_type in PRIMARY_SOURCE_TYPES,
        )
