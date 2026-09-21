"""Research tasks: the recursive unit of investigation work."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from research.models.common import StrEnum, utcnow


class ResearchRole(StrEnum):
    """Specialised roles sharing one research infrastructure."""

    PLANNER = "planner"
    SCOUT = "scout"
    ACADEMIC = "academic"
    NEWS = "news"
    PRIMARY_SOURCE = "primary_source"
    SOCIAL = "social"
    SKEPTIC = "skeptic"
    SYNTHESIZER = "synthesizer"


class Operation(StrEnum):
    """Research-native operations a task can perform.

    These are the verbs the planner reasons in. They are intentionally about
    research intent ('find the original source') rather than transport
    ('GET /works?filter=...').
    """

    SEARCH_ACADEMIC = "search_academic"
    SEARCH_NEWS = "search_news"
    SEARCH_WEB = "search_web"
    SEARCH_SOCIAL = "search_social"
    FETCH_SOURCE = "fetch_source"
    FOLLOW_CITATIONS = "follow_citations"
    FIND_PRIMARY_SOURCE = "find_primary_source"
    FIND_COUNTEREVIDENCE = "find_counterevidence"
    RESOLVE_ENTITY = "resolve_entity"
    BUILD_TIMELINE = "build_timeline"
    SYNTHESIZE = "synthesize"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"  # e.g. budget exhausted before it started


@dataclass(slots=True)
class FollowUp:
    """A lead worth pursuing, produced by a task for the planner."""

    operation: Operation
    objective: str
    rationale: str | None = None
    priority: int = 5  # 1 = highest
    seed_document_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["operation"] = str(self.operation)
        return data


@dataclass(slots=True)
class TaskResult:
    """Structured return value of a research task.

    Child tasks persist evidence and return *this*. They never hand their
    retrieved text back up to a parent model context.
    """

    summary: str = ""
    claim_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    recommended_followups: list[FollowUp] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "claims": self.claim_ids,
            "entities": self.entity_ids,
            "evidence": self.evidence_ids,
            "open_questions": list(self.open_questions),
            "recommended_followups": [f.to_dict() for f in self.recommended_followups],
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskResult":
        followups = []
        for raw in data.get("recommended_followups", []):
            op = Operation.coerce(raw.get("operation"), Operation.SEARCH_WEB)
            followups.append(
                FollowUp(
                    operation=op,
                    objective=raw.get("objective", ""),
                    rationale=raw.get("rationale"),
                    priority=int(raw.get("priority", 5)),
                    seed_document_ids=list(raw.get("seed_document_ids", [])),
                )
            )
        return cls(
            summary=data.get("summary", ""),
            claim_ids=list(data.get("claims", [])),
            entity_ids=list(data.get("entities", [])),
            evidence_ids=list(data.get("evidence", [])),
            open_questions=list(data.get("open_questions", [])),
            recommended_followups=followups,
            metrics=dict(data.get("metrics", {})),
        )


@dataclass(slots=True)
class ResearchTask:
    """One tractable piece of an investigation."""

    id: str
    investigation_id: str
    role: ResearchRole
    operation: Operation
    objective: str
    status: TaskStatus = TaskStatus.PENDING
    parent_task_id: str | None = None
    depth: int = 0
    priority: int = 5
    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: TaskResult | None = None
    error: str | None = None
    #: Free-form parameters for the operation (query strings, seed doc ids,
    #: date ranges). Kept as data so a task is resumable.
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.SKIPPED,
        )
