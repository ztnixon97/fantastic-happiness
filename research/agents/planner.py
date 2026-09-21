"""Turning a question into tasks.

The planner is the only role that writes tasks directly. Everything it
produces is validated before it is stored: unknown roles, unknown operations
and empty objectives are dropped rather than persisted, and the number of
tasks is capped by the budget rather than by the model's enthusiasm.

It also converts the follow-ups workers report into new tasks, which is what
makes the system recursive - and refuses the ones that repeat work already
planned, which is what stops it from being infinite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.agents.prompts import planner_prompt
from research.errors import ModelError, ModelOutputError
from research.llm.base import Message, ModelClient, extract_json
from research.models.investigation import Investigation
from research.models.task import (
    FollowUp,
    Operation,
    ResearchRole,
    ResearchTask,
    TaskStatus,
)
from research.normalize.text import normalize_title, normalize_whitespace
from research.budgets import BudgetLedger, Resource
from research.storage.store import ResearchStore

PLANNER_SYSTEM = """\
You plan research. You do not carry it out.

Given a question, decide how to break it into tasks that specialised workers
can complete: academic literature, current news, primary records, public
statements, and a skeptic whose job is to attack whatever the others
conclude. Name the entities that will recur and the propositions the answer
turns on.

Good tasks are specific enough that a worker knows when it is finished, and
distinct enough that two workers will not do the same work twice.

Reply with one JSON object and nothing else.
"""


@dataclass(slots=True)
class Plan:
    brief: str = ""
    entities: list[str] = field(default_factory=list)
    claims_to_verify: list[str] = field(default_factory=list)
    tasks: list[ResearchTask] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "brief": self.brief,
            "entities": list(self.entities),
            "claims_to_verify": list(self.claims_to_verify),
            "tasks": [
                {
                    "task_id": task.id,
                    "role": str(task.role),
                    "operation": str(task.operation),
                    "objective": task.objective,
                    "priority": task.priority,
                }
                for task in self.tasks
            ],
            "rejected": list(self.rejected),
        }


class Planner:
    def __init__(
        self,
        store: ResearchStore,
        model: ModelClient,
        *,
        investigation_id: str,
        ledger: BudgetLedger,
    ) -> None:
        self.store = store
        self.model = model
        self.investigation_id = investigation_id
        self.ledger = ledger

    async def plan(self, investigation: Investigation, *, max_tasks: int = 5) -> Plan:
        """Ask for a decomposition, validate it, and store the tasks."""
        allowance = min(max_tasks, int(self.ledger.state(Resource.TASKS).remaining))
        if allowance <= 0:
            return Plan(brief="task budget exhausted before planning began")

        messages = [
            Message(role="system", content=PLANNER_SYSTEM),
            Message(
                role="user",
                content=planner_prompt(
                    investigation.question,
                    brief=investigation.brief,
                    budget={
                        name: values["remaining"]
                        for name, values in self.ledger.snapshot().items()
                        if name in ("searches", "documents", "tasks", "model_calls")
                    },
                    max_tasks=allowance,
                ),
            ),
        ]
        try:
            response = await self.model.complete(messages)
        except ModelError as exc:
            raise ModelError(f"the planner could not run: {exc}", provider="planner") from exc

        self.ledger.try_spend(Resource.MODEL_CALLS)
        self.ledger.try_spend(Resource.TOKENS, response.usage.total_tokens)

        payload = extract_json(response.text)
        if not isinstance(payload, dict):
            raise ModelOutputError("the planner did not return a JSON object")

        plan = Plan(
            brief=normalize_whitespace(str(payload.get("brief", ""))),
            entities=[str(value) for value in payload.get("entities", [])][:20],
            claims_to_verify=[str(value) for value in payload.get("claims_to_verify", [])][:20],
        )
        for raw in payload.get("tasks", [])[: allowance * 2]:
            task = self._task_from(raw, plan)
            if task is None:
                continue
            plan.tasks.append(task)
            if len(plan.tasks) >= allowance:
                break

        if plan.brief:
            self.store.investigations.update_metadata(
                self.investigation_id,
                {
                    **investigation.metadata,
                    "plan_brief": plan.brief,
                    "planned_entities": plan.entities,
                    "claims_to_verify": plan.claims_to_verify,
                },
            )
        return plan

    def _task_from(self, raw: Any, plan: Plan) -> ResearchTask | None:
        if not isinstance(raw, dict):
            plan.rejected.append("a task entry was not an object")
            return None
        objective = normalize_whitespace(str(raw.get("objective", "")))
        if not objective:
            plan.rejected.append("a task had no objective")
            return None
        role = ResearchRole.coerce(raw.get("role"), None)
        if role is None or role is ResearchRole.PLANNER:
            plan.rejected.append(f"unusable role for {objective[:60]!r}")
            return None
        if self._duplicate_objective(objective):
            plan.rejected.append(f"already planned: {objective[:60]!r}")
            return None
        if not self.ledger.try_spend(Resource.TASKS):
            plan.rejected.append("task budget exhausted")
            return None

        operation = Operation.coerce(raw.get("operation"), _default_operation(role))
        priority = raw.get("priority")
        return self.store.tasks.create(
            ResearchTask(
                id="",
                investigation_id=self.investigation_id,
                role=role,
                operation=operation,
                objective=objective,
                depth=1,
                priority=_clamp(priority, default=5),
                parameters={"rationale": raw.get("rationale")} if raw.get("rationale") else {},
            )
        )

    # -- recursion ------------------------------------------------------
    def ingest_followups(
        self,
        parent: ResearchTask,
        followups: list[FollowUp],
        *,
        max_new_tasks: int = 3,
    ) -> list[ResearchTask]:
        """Turn a worker's recommended follow-ups into tasks.

        Bounded three ways: by depth, by the task budget, and by refusing
        objectives that repeat something already planned. Without the third,
        workers recommend each other in circles.
        """
        created: list[ResearchTask] = []
        depth = parent.depth + 1
        if not self.ledger.allows_depth(depth):
            return created

        for followup in followups:
            if len(created) >= max_new_tasks:
                break
            objective = normalize_whitespace(followup.objective)
            if not objective or self._duplicate_objective(objective):
                continue
            if not self.ledger.can_spend(Resource.TASKS):
                break
            role = _role_for_operation(followup.operation)
            self.ledger.try_spend(Resource.TASKS)
            created.append(
                self.store.tasks.create(
                    ResearchTask(
                        id="",
                        investigation_id=self.investigation_id,
                        parent_task_id=parent.id,
                        role=role,
                        operation=followup.operation,
                        objective=objective,
                        depth=depth,
                        priority=followup.priority,
                        parameters={
                            "rationale": followup.rationale,
                            "seed_document_ids": followup.seed_document_ids,
                        },
                    )
                )
            )
        return created

    def _duplicate_objective(self, objective: str) -> bool:
        key = normalize_title(objective)
        if not key:
            return True
        for task in self.store.tasks.list(self.investigation_id, limit=500):
            if normalize_title(task.objective) == key:
                return True
        return False

    def pending(self) -> list[ResearchTask]:
        return self.store.tasks.list(self.investigation_id, status=TaskStatus.PENDING)


def _default_operation(role: ResearchRole) -> Operation:
    return {
        ResearchRole.ACADEMIC: Operation.SEARCH_ACADEMIC,
        ResearchRole.NEWS: Operation.SEARCH_NEWS,
        ResearchRole.PRIMARY_SOURCE: Operation.FIND_PRIMARY_SOURCE,
        ResearchRole.SKEPTIC: Operation.FIND_COUNTEREVIDENCE,
        ResearchRole.SOCIAL: Operation.SEARCH_SOCIAL,
    }.get(role, Operation.SEARCH_WEB)


def _role_for_operation(operation: Operation) -> ResearchRole:
    return {
        Operation.SEARCH_ACADEMIC: ResearchRole.ACADEMIC,
        Operation.FOLLOW_CITATIONS: ResearchRole.ACADEMIC,
        Operation.SEARCH_NEWS: ResearchRole.NEWS,
        Operation.FIND_PRIMARY_SOURCE: ResearchRole.PRIMARY_SOURCE,
        Operation.FIND_COUNTEREVIDENCE: ResearchRole.SKEPTIC,
        Operation.SEARCH_SOCIAL: ResearchRole.SOCIAL,
    }.get(operation, ResearchRole.SCOUT)


def _clamp(value: Any, *, default: int) -> int:
    try:
        return max(1, min(int(value), 9))
    except (TypeError, ValueError):
        return default
