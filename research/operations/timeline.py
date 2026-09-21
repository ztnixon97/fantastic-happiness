"""Recording events and building timelines.

An event is an assertion about the world, so it obeys the same rule as a
claim: it must be backed by evidence the investigation actually holds. An
event with no evidence is refused rather than stored, because an unsupported
point on a chronology is indistinguishable from a supported one once it is
drawn.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from research.errors import IntegrityError
from research.graph.timelines import Timeline, TimelineEntry
from research.models.event import DatePrecision, Event
from research.normalize.text import normalize_whitespace
from research.storage.store import ResearchStore


class TimelineOperations:
    def __init__(
        self,
        store: ResearchStore,
        *,
        investigation_id: str,
        task_id: str | None = None,
    ) -> None:
        self.store = store
        self.investigation_id = investigation_id
        self.task_id = task_id
        self.timeline = Timeline(store, investigation_id)

    def record_event(
        self,
        description: str,
        *,
        evidence_ids: Sequence[str],
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        date_precision: DatePrecision = DatePrecision.UNKNOWN,
        entity_ids: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        text = normalize_whitespace(description)
        if not text:
            raise IntegrityError("an event needs a description")
        if not evidence_ids:
            raise IntegrityError(
                "an event needs at least one evidence document; an undocumented "
                "event does not belong on a timeline"
            )
        for document_id in evidence_ids:
            document = self.store.documents.get(document_id)
            if document.investigation_id != self.investigation_id:
                raise IntegrityError(
                    f"{document_id} belongs to another investigation"
                )
        if date_start is None and date_precision is not DatePrecision.UNKNOWN:
            raise IntegrityError(
                "a date precision was given without a date; say UNKNOWN instead"
            )
        if date_start is not None and date_precision is DatePrecision.UNKNOWN:
            raise IntegrityError(
                "a dated event needs its precision stated (exact, day, month, "
                "quarter or year)"
            )
        if date_end is not None and date_start is not None and date_end < date_start:
            raise IntegrityError("an event cannot end before it starts")

        for entity_id in entity_ids:
            self.store.entities.get(entity_id)

        event = self.store.events.create(
            self.investigation_id,
            text,
            date_start=date_start,
            date_end=date_end,
            date_precision=date_precision,
            entity_ids=list(entity_ids),
            evidence_ids=list(evidence_ids),
            created_by_task_id=self.task_id,
            metadata=dict(metadata or {}),
        )
        for entity_id in entity_ids:
            self.store.relationships.add(
                self.investigation_id,
                subject_type="entity",
                subject_id=entity_id,
                predicate="PARTICIPATED_IN",
                object_type="event",
                object_id=event.id,
                evidence_document_id=evidence_ids[0],
                created_by_task_id=self.task_id,
            )
        for document_id in evidence_ids:
            self.store.relationships.add(
                self.investigation_id,
                subject_type="document",
                subject_id=document_id,
                predicate="ABOUT",
                object_type="event",
                object_id=event.id,
                evidence_document_id=document_id,
                created_by_task_id=self.task_id,
            )
        return event

    def build_timeline(
        self, *, include_publications: bool = False, limit: int = 200
    ) -> list[TimelineEntry]:
        return self.timeline.build(include_publications=include_publications, limit=limit)
