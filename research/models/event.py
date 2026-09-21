"""Evidence-backed events, the substrate for timelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from research.models.common import StrEnum, utcnow


class DatePrecision(StrEnum):
    """How precisely an event is dated.

    Publication dates and reported dates are frequently vaguer than a
    timestamp suggests; recording precision avoids inventing certainty.
    """

    EXACT = "exact"
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Event:
    id: str
    investigation_id: str
    description: str
    date_start: datetime | None = None
    date_end: datetime | None = None
    date_precision: DatePrecision = DatePrecision.UNKNOWN
    entity_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    created_by_task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
