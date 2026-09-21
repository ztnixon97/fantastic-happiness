"""When to stop.

Research should end because it decided to, not because a context window ran
out. These rules make that decision explicit and record which one fired, so
'why did it stop there?' has an answer months later.

The rules are evaluated in order of authority: a hard limit stops work
immediately; evidence-based reasons only apply once there is enough evidence
to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.graph.independence import independent_documents
from research.models.investigation import StopReason
from research.models.task import TaskStatus
from research.operations.claims import ClaimOperations
from research.budgets import BudgetLedger, Resource
from research.storage.store import ResearchStore

#: Searches to look back over when judging whether new work is still finding
#: anything. Short, because an investigation is short.
DILIGENCE_WINDOW = 4


@dataclass(frozen=True, slots=True)
class StopDecision:
    should_stop: bool
    reason: StopReason | None = None
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.should_stop


CONTINUE = StopDecision(should_stop=False)


@dataclass(slots=True)
class StoppingRules:
    store: ResearchStore
    investigation_id: str
    ledger: BudgetLedger
    #: Fraction of recent searches that may return nothing new before the
    #: investigation is judged to be repeating itself.
    diminishing_threshold: float = 0.75

    def evaluate(self) -> StopDecision:
        for rule in (
            self._budget_exhausted,
            self._runtime_exhausted,
            self._no_open_tasks,
            self._diminishing_returns,
            self._evidence_sufficient,
        ):
            decision = rule()
            if decision.should_stop:
                return decision
        return CONTINUE

    # -- hard limits ----------------------------------------------------
    def _budget_exhausted(self) -> StopDecision:
        exhausted = [
            resource
            for resource in self.ledger.exhausted_resources()
            if resource is not Resource.RUNTIME_SECONDS
        ]
        blocking = [
            resource
            for resource in exhausted
            if resource in (Resource.DOCUMENTS, Resource.SEARCHES, Resource.MODEL_CALLS,
                            Resource.PROVIDER_CALLS, Resource.TASKS)
        ]
        if not blocking:
            return CONTINUE
        return StopDecision(
            should_stop=True,
            reason=StopReason.BUDGET_EXHAUSTED,
            detail="exhausted: " + ", ".join(str(resource) for resource in blocking),
            evidence={"exhausted": [str(resource) for resource in blocking]},
        )

    def _runtime_exhausted(self) -> StopDecision:
        state = self.ledger.state(Resource.RUNTIME_SECONDS)
        if not state.exhausted:
            return CONTINUE
        return StopDecision(
            should_stop=True,
            reason=StopReason.RUNTIME_EXHAUSTED,
            detail=f"ran for {state.used:.0f}s of an allowed {state.limit:.0f}s",
        )

    def _no_open_tasks(self) -> StopDecision:
        pending = self.store.tasks.list(
            self.investigation_id, status=TaskStatus.PENDING, limit=1
        )
        if pending:
            return CONTINUE
        return StopDecision(
            should_stop=True,
            reason=StopReason.NO_OPEN_TASKS,
            detail="every planned task has run and none proposed further work",
        )

    # -- evidence-based -------------------------------------------------
    def _diminishing_returns(self) -> StopDecision:
        """New work is mostly returning material already held.

        Two independent signals, checked separately because they can occur
        apart: a corpus that has become mostly copies, and a run of searches
        that returned nothing at all.
        """
        documents = self.store.documents.count(self.investigation_id)
        originals = self.store.documents.count(self.investigation_id, originals_only=True)
        if documents >= 10 and originals / max(documents, 1) < 0.4:
            return StopDecision(
                should_stop=True,
                reason=StopReason.DIMINISHING_RETURNS,
                detail=(
                    f"{documents - originals} of {documents} documents are copies or "
                    "rewrites of material already held"
                ),
                evidence={"documents": documents, "independent": originals},
            )

        recent = self.store.queries.list(self.investigation_id, limit=DILIGENCE_WINDOW)
        if len(recent) < DILIGENCE_WINDOW:
            return CONTINUE
        barren = sum(1 for query in recent if not query["result_count"])
        if barren / len(recent) >= self.diminishing_threshold:
            return StopDecision(
                should_stop=True,
                reason=StopReason.DIMINISHING_RETURNS,
                detail=f"{barren} of the last {len(recent)} searches returned nothing",
            )
        return CONTINUE

    def _evidence_sufficient(self) -> StopDecision:
        """Major claims are evidenced, traced and tested."""
        operations = ClaimOperations(self.store, investigation_id=self.investigation_id)
        claims = operations.list_claims(limit=200)
        if len(claims) < 2:
            return CONTINUE
        open_questions = operations.open_questions(limit=200)
        if open_questions:
            return CONTINUE
        documents = self.store.documents.list(self.investigation_id, limit=1000)
        groups = independent_documents(self.store.documents, [d.id for d in documents])
        return StopDecision(
            should_stop=True,
            reason=StopReason.EVIDENCE_SUFFICIENT,
            detail=(
                f"{len(claims)} claims are evidenced with no outstanding gaps, across "
                f"{len(groups)} independent sources"
            ),
            evidence={"claims": len(claims), "independent_sources": len(groups)},
        )

    # -- reporting ------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """What each rule currently sees - used by the CLI to explain a run."""
        return {
            "budget": self._budget_exhausted().detail or "within limits",
            "runtime": self._runtime_exhausted().detail or "within limits",
            "open_tasks": len(
                self.store.tasks.list(self.investigation_id, status=TaskStatus.PENDING)
            ),
            "diminishing_returns": (
                self._diminishing_returns().detail or "still finding new material"
            ),
            "evidence": self._evidence_sufficient().detail or "claims still have gaps",
        }
