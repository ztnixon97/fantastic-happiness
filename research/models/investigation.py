"""Top-level unit of research work."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from research.models.common import StrEnum, utcnow


class InvestigationStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class StopReason(StrEnum):
    """Why an investigation stopped. Always recorded, never inferred later."""

    EVIDENCE_SUFFICIENT = "evidence_sufficient"
    DIMINISHING_RETURNS = "diminishing_returns"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RUNTIME_EXHAUSTED = "runtime_exhausted"
    UNRESOLVABLE_UNCERTAINTY = "unresolvable_uncertainty"
    #: The sources needed were unreachable - rate-limited, unconfigured or
    #: down. Distinct from diminishing returns: nothing was exhausted, the
    #: looking simply could not happen.
    SOURCES_UNAVAILABLE = "sources_unavailable"
    NO_OPEN_TASKS = "no_open_tasks"
    OPERATOR_STOPPED = "operator_stopped"
    ERROR = "error"


@dataclass(slots=True)
class Investigation:
    """A question under investigation, plus everything learned about it.

    The investigation is durable state. Resuming one never requires replaying
    a conversation: the store holds the tasks, evidence, claims and budget.
    """

    id: str
    question: str
    status: InvestigationStatus = InvestigationStatus.CREATED
    brief: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    completed_at: datetime | None = None
    stop_reason: StopReason | None = None
    stop_detail: str | None = None
    #: Snapshot of the budget policy this investigation runs under, so a
    #: resumed run cannot silently inherit different limits.
    budget: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status in (
            InvestigationStatus.CREATED,
            InvestigationStatus.PLANNING,
            InvestigationStatus.RUNNING,
            InvestigationStatus.PAUSED,
        )
