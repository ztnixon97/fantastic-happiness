"""Running an investigation.

Plan, then work the queue: take the highest-priority pending task, run it,
turn its follow-ups into new tasks within the depth and budget limits, and
check the stopping rules before starting the next one. Stop for a stated
reason and record it.

The scheduler holds no research knowledge. It decides what runs next and when
to stop, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.agents.planner import Plan, Planner
from research.agents.worker import ResearchWorker, WorkerOutcome
from research.errors import BudgetExceeded, ModelError
from research.llm.base import ModelClient
from research.models.investigation import InvestigationStatus, StopReason
from research.models.task import ResearchTask, TaskStatus
from research.budgets import BudgetLedger
from research.orchestration.stopping import StopDecision, StoppingRules
from research.sources.registry import SourceRegistry
from research.storage.store import ResearchStore


@dataclass(slots=True)
class InvestigationRun:
    investigation_id: str
    plan: Plan | None = None
    outcomes: list[WorkerOutcome] = field(default_factory=list)
    stop_reason: StopReason = StopReason.NO_OPEN_TASKS
    stop_detail: str = ""
    error: str | None = None

    @property
    def tasks_run(self) -> int:
        return len(self.outcomes)

    def summary(self) -> dict[str, Any]:
        return {
            "investigation": self.investigation_id,
            "tasks_run": self.tasks_run,
            "stop_reason": str(self.stop_reason),
            "stop_detail": self.stop_detail,
            "tasks": [outcome.summary() for outcome in self.outcomes],
            **({"error": self.error} if self.error else {}),
        }


def _ignore_event(event: str, payload: dict[str, Any]) -> None:
    """Default progress sink: a run that nobody is watching still runs."""


class Scheduler:
    def __init__(
        self,
        store: ResearchStore,
        registry: SourceRegistry,
        model: ModelClient,
        *,
        investigation_id: str,
        ledger: BudgetLedger,
        max_steps_per_task: int = 8,
        on_event: Any = None,
        embeddings: Any = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.model = model
        self.investigation_id = investigation_id
        self.ledger = ledger
        self.max_steps_per_task = max_steps_per_task
        #: Vector index for workers' own searches of held evidence, so a
        #: worker ranks its corpus the same way the CLI does.
        self.embeddings = embeddings
        self.planner = Planner(
            store, model, investigation_id=investigation_id, ledger=ledger
        )
        self.stopping = StoppingRules(store, investigation_id, ledger)
        #: Optional callback for progress output; the CLI passes a printer.
        self.on_event = on_event or _ignore_event

    async def run(self, *, max_tasks: int | None = None, plan_size: int = 5) -> InvestigationRun:
        investigation = self.store.investigations.get(self.investigation_id)
        run = InvestigationRun(investigation_id=self.investigation_id)
        self.store.investigations.set_status(
            self.investigation_id, InvestigationStatus.PLANNING
        )

        try:
            run.plan = await self.planner.plan(investigation, max_tasks=plan_size)
            self.on_event("planned", run.plan.summary())
        except (ModelError, BudgetExceeded) as exc:
            run.error = str(exc)
            return self._finish(run, StopDecision(
                should_stop=True, reason=StopReason.ERROR, detail=str(exc)
            ))

        self.store.investigations.set_status(
            self.investigation_id, InvestigationStatus.RUNNING
        )
        worker = ResearchWorker(
            self.store,
            self.registry,
            self.model,
            investigation_id=self.investigation_id,
            ledger=self.ledger,
            max_steps=self.max_steps_per_task,
            embeddings=self.embeddings,
        )

        while True:
            decision = self.stopping.evaluate()
            if decision.should_stop:
                return self._finish(run, decision)
            if max_tasks is not None and run.tasks_run >= max_tasks:
                return self._finish(run, StopDecision(
                    should_stop=True,
                    reason=StopReason.OPERATOR_STOPPED,
                    detail=f"reached the requested limit of {max_tasks} tasks",
                ))

            task = self.store.tasks.next_pending(self.investigation_id)
            if task is None:
                return self._finish(run, StopDecision(
                    should_stop=True,
                    reason=StopReason.NO_OPEN_TASKS,
                    detail="no tasks remain",
                ))

            self.on_event("task_started", {"task": task.id, "role": str(task.role),
                                           "objective": task.objective})
            try:
                outcome = await worker.run(task)
            except BudgetExceeded as exc:
                self.store.tasks.finish(task.id, TaskStatus.SKIPPED, error=str(exc))
                return self._finish(run, StopDecision(
                    should_stop=True, reason=StopReason.BUDGET_EXHAUSTED, detail=str(exc)
                ))
            except ModelError as exc:
                self.store.tasks.finish(task.id, TaskStatus.FAILED, error=str(exc))
                run.error = str(exc)
                return self._finish(run, StopDecision(
                    should_stop=True, reason=StopReason.ERROR, detail=str(exc)
                ))

            run.outcomes.append(outcome)
            self.on_event("task_finished", outcome.summary())
            self._ingest_followups(task, outcome)

    def _ingest_followups(self, task: ResearchTask, outcome: WorkerOutcome) -> None:
        followups = outcome.result.recommended_followups
        if not followups:
            return
        created = self.planner.ingest_followups(task, followups)
        if created:
            self.on_event(
                "followups",
                {
                    "parent": task.id,
                    "created": [
                        {"task_id": child.id, "role": str(child.role),
                         "objective": child.objective, "depth": child.depth}
                        for child in created
                    ],
                    "proposed": len(followups),
                },
            )

    def _finish(self, run: InvestigationRun, decision: StopDecision) -> InvestigationRun:
        run.stop_reason = decision.reason or StopReason.NO_OPEN_TASKS
        run.stop_detail = decision.detail
        status = (
            InvestigationStatus.FAILED
            if decision.reason is StopReason.ERROR
            else InvestigationStatus.COMPLETED
        )
        self.store.investigations.set_status(
            self.investigation_id,
            status,
            stop_reason=run.stop_reason,
            stop_detail=decision.detail,
        )
        # Anything still queued is cancelled explicitly rather than left
        # pending, so a resumed run does not silently continue a stopped one.
        for pending in self.store.tasks.list(
            self.investigation_id, status=TaskStatus.PENDING
        ):
            self.store.tasks.finish(
                pending.id, TaskStatus.SKIPPED, error=f"investigation stopped: {run.stop_reason}"
            )
        self.on_event("stopped", {"reason": str(run.stop_reason), "detail": decision.detail})
        return run
