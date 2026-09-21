"""Running one research task.

A worker is a loop: look at the state, choose one action, see a compact
observation, repeat, and finish with a structured report. It ends for one of
four reasons, all recorded: it completed, it ran out of steps, the budget
stopped it, or the model failed to produce a usable action often enough to
give up on.

What a worker returns to its parent is a :class:`TaskResult` - a summary and
identifiers. Retrieved text stays in the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.agents.actions import ACTIONS
from research.agents.prompts import system_prompt, task_prompt
from research.agents.runtime import ActionRequest, ActionRuntime, Observation, document_brief
from research.retrieval.search import CorpusSearch
from research.errors import BudgetExceeded, ModelError, ModelOutputError
from research.graph.independence import independent_documents
from research.llm.base import Message, ModelClient, extract_json
from research.models.task import ResearchTask, TaskResult, TaskStatus
from research.normalize.text import truncate
from research.operations.claims import ClaimOperations
from research.budgets import BudgetLedger, Resource
from research.sources.registry import SourceRegistry
from research.storage.store import ResearchStore

#: How many malformed replies to tolerate before abandoning the task.
MAX_PARSE_FAILURES = 2


@dataclass(slots=True)
class WorkerOutcome:
    task: ResearchTask
    result: TaskResult
    status: TaskStatus
    steps: int = 0
    stopped_by: str = "completed"
    spawned_task_ids: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task.id,
            "role": str(self.task.role),
            "status": str(self.status),
            "steps": self.steps,
            "stopped_by": self.stopped_by,
            "evidence": len(self.result.evidence_ids),
            "claims": len(self.result.claim_ids),
            "spawned": list(self.spawned_task_ids),
        }


class ResearchWorker:
    """Executes one task with one role's action vocabulary."""

    def __init__(
        self,
        store: ResearchStore,
        registry: SourceRegistry,
        model: ModelClient,
        *,
        investigation_id: str,
        ledger: BudgetLedger,
        max_steps: int = 8,
        embeddings: Any = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.model = model
        self.investigation_id = investigation_id
        self.ledger = ledger
        self.max_steps = max_steps
        self.embeddings = embeddings

    async def run(self, task: ResearchTask) -> WorkerOutcome:
        investigation = self.store.investigations.get(self.investigation_id)
        runtime = ActionRuntime(
            self.store,
            self.registry,
            investigation_id=self.investigation_id,
            task=task,
            ledger=self.ledger,
            corpus=CorpusSearch(
                self.store, self.investigation_id, embeddings=self.embeddings
            ),
        )
        self.store.tasks.mark_running(task.id)

        messages: list[Message] = [
            Message(role="system", content=system_prompt(task.role)),
            Message(
                role="user",
                content=task_prompt(
                    question=investigation.question,
                    objective=task.objective,
                    state=self._state_summary(),
                    budget=self._budget_summary(),
                    steps_remaining=self.max_steps,
                ),
            ),
        ]

        observations: list[Observation] = []
        parse_failures = 0
        stopped_by = "step limit reached"
        started_documents = set(self._document_ids())

        for step in range(1, self.max_steps + 1):
            if not self.ledger.can_spend(Resource.MODEL_CALLS):
                stopped_by = "model call budget exhausted"
                break
            try:
                self.ledger.checkpoint_runtime()
            except BudgetExceeded as exc:
                stopped_by = str(exc)
                break

            try:
                response = await self.model.complete(messages)
            except ModelError as exc:
                stopped_by = f"model unavailable: {exc}"
                break
            except Exception as exc:  # noqa: BLE001 - third-party client
                # An adapter can raise anything. One misbehaving call should
                # cost this task, not the whole investigation, and the reason
                # is recorded on the task either way.
                stopped_by = f"model client raised {type(exc).__name__}: {exc}"
                break

            self.ledger.try_spend(Resource.MODEL_CALLS)
            self.ledger.try_spend(Resource.TOKENS, response.usage.total_tokens)
            messages.append(Message(role="assistant", content=response.text))

            try:
                request = self._parse_action(response.text)
            except ModelOutputError as exc:
                parse_failures += 1
                if parse_failures > MAX_PARSE_FAILURES:
                    stopped_by = f"model did not produce a usable action: {exc}"
                    break
                messages.append(
                    Message(
                        role="user",
                        content=(
                            f"That was not usable: {exc}. Reply with one JSON object "
                            'of the form {"thought": "...", "action": "...", '
                            '"arguments": {...}} and nothing else.'
                        ),
                    )
                )
                continue

            observation = await runtime.execute(request)
            observations.append(observation)
            messages.append(
                Message(role="user", content=self._observation_message(observation, step))
            )

            if observation.terminal:
                stopped_by = "completed"
                break
            if observation.ok is False and observation.error and "budget" in observation.error:
                stopped_by = observation.error
                break
        else:
            stopped_by = "step limit reached"

        result = self._build_result(
            runtime, observations, started_documents, stopped_by=stopped_by
        )
        status = TaskStatus.COMPLETED if runtime.result_payload else TaskStatus.FAILED
        if not runtime.result_payload and result.evidence_ids:
            # Work was done even though the worker never reported: keep it,
            # and say why the report is missing.
            status = TaskStatus.COMPLETED
        self.store.tasks.finish(
            task.id,
            status,
            result=result,
            error=None if status is TaskStatus.COMPLETED else stopped_by,
        )
        return WorkerOutcome(
            task=task,
            result=result,
            status=status,
            steps=len(observations),
            stopped_by=stopped_by,
            spawned_task_ids=[child.id for child in runtime.spawned],
            observations=observations,
        )

    # -- helpers --------------------------------------------------------
    def _parse_action(self, text: str) -> ActionRequest:
        payload = extract_json(text)
        if not isinstance(payload, dict):
            raise ModelOutputError("expected a JSON object")
        name = payload.get("action") or payload.get("name")
        if not isinstance(name, str) or not name:
            raise ModelOutputError("no 'action' field")
        if name not in ACTIONS:
            raise ModelOutputError(f"unknown action {name!r}")
        arguments = payload.get("arguments") or payload.get("args") or {}
        if not isinstance(arguments, dict):
            raise ModelOutputError("'arguments' must be an object")
        thought = payload.get("thought")
        return ActionRequest(
            name=name,
            arguments=arguments,
            thought=str(thought) if isinstance(thought, str) else None,
        )

    def _observation_message(self, observation: Observation, step: int) -> str:
        remaining = self.max_steps - step
        lines = [observation.render()]
        if remaining <= 1:
            lines.append(
                "This is your last step: use complete_research_task now and report "
                "what you established."
            )
        elif remaining <= 3:
            lines.append(f"{remaining} steps remain.")
        return "\n".join(lines)

    def _document_ids(self) -> list[str]:
        return [
            document.id
            for document in self.store.documents.list(self.investigation_id, limit=1000)
        ]

    def _state_summary(self) -> dict[str, Any]:
        documents = self.store.documents.list(self.investigation_id, limit=1000)
        claims = self.store.claims.list(self.investigation_id, hydrate=False)
        operations = ClaimOperations(self.store, investigation_id=self.investigation_id)
        open_questions = operations.open_questions(limit=10)
        recent = documents[-6:]
        return {
            "documents": len(documents),
            "independent_sources": len(
                independent_documents(self.store.documents, [d.id for d in documents])
            ),
            "claims": len(claims),
            "open_questions": len(open_questions),
            "open_question_list": open_questions,
            "recent_documents": [document_brief(document) for document in recent],
        }

    def _budget_summary(self) -> dict[str, float]:
        snapshot = self.ledger.snapshot()
        return {
            name: values["remaining"]
            for name, values in snapshot.items()
            if name in ("searches", "documents", "provider_calls", "model_calls", "tasks")
        }

    def _build_result(
        self,
        runtime: ActionRuntime,
        observations: list[Observation],
        started_documents: set[str],
        *,
        stopped_by: str,
    ) -> TaskResult:
        payload = dict(runtime.result_payload or {})
        acquired = [
            document_id
            for document_id in self._document_ids()
            if document_id not in started_documents
        ]
        if not payload:
            payload = {
                "summary": (
                    f"Task ended without a report ({stopped_by}). "
                    f"{len(acquired)} document(s) were acquired and are held."
                ),
                "claims": [],
                "evidence": acquired,
                "open_questions": [],
                "recommended_followups": [],
            }
        else:
            # Evidence the worker forgot to mention is still evidence it found.
            reported = set(payload.get("evidence", []))
            payload["evidence"] = list(payload.get("evidence", [])) + [
                document_id for document_id in acquired if document_id not in reported
            ]
        payload.setdefault("metrics", {})
        payload["metrics"].update(
            {
                "steps": len(observations),
                "actions": [observation.action for observation in observations],
                "failed_actions": sum(1 for observation in observations if not observation.ok),
                "stopped_by": stopped_by,
                "model": getattr(self.model.spec, "provider", "unknown"),
            }
        )
        payload["summary"] = truncate(str(payload.get("summary", "")), 2000)
        return TaskResult.from_dict(payload)
