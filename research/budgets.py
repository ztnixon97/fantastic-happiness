"""Research budgets.

This sits beside :mod:`research.config` rather than inside the orchestration
package because every layer charges against it - acquisition, operations,
workers and the scheduler alike - and none of them should have to depend on
the code that decides what runs next in order to do so.

Budgets are durable counters, not in-process ones: an investigation resumed
tomorrow continues spending the same allowance it started with. Every
expenditure goes through :class:`BudgetLedger`, so 'why did this stop?' has
an answer that does not depend on remembering the run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from research.config import BudgetPolicy
from research.errors import BudgetExceeded
from research.models.common import SourceFamily, StrEnum
from research.storage.repositories import BudgetRepository


class Resource(StrEnum):
    TASKS = "tasks"
    SEARCHES = "searches"
    DOCUMENTS = "documents"
    ACADEMIC_DOCUMENTS = "academic_documents"
    NEWS_DOCUMENTS = "news_documents"
    CITATION_DOCUMENTS = "citation_documents"
    PROVIDER_CALLS = "provider_calls"
    MODEL_CALLS = "model_calls"
    TOKENS = "tokens"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    #: Money, in whatever currency the configured prices are in. Counted
    #: only for models the configuration has priced; see ModelSettings.prices.
    COST = "cost"
    FAILED_SOURCE_CALLS = "failed_source_calls"
    RUNTIME_SECONDS = "runtime_seconds"


_LIMIT_FIELDS: dict[Resource, str] = {
    Resource.TASKS: "max_tasks",
    Resource.SEARCHES: "max_searches",
    Resource.DOCUMENTS: "max_documents",
    Resource.ACADEMIC_DOCUMENTS: "max_academic_documents",
    Resource.NEWS_DOCUMENTS: "max_news_documents",
    Resource.CITATION_DOCUMENTS: "max_citation_documents",
    Resource.PROVIDER_CALLS: "max_provider_calls",
    Resource.MODEL_CALLS: "max_model_calls",
    Resource.TOKENS: "max_tokens",
    Resource.COST: "max_cost",
    Resource.FAILED_SOURCE_CALLS: "max_failed_source_calls",
}

#: Documents also count against a per-family allowance, so one prolific
#: source family cannot consume the whole document budget.
FAMILY_RESOURCE = {
    SourceFamily.ACADEMIC: Resource.ACADEMIC_DOCUMENTS,
    SourceFamily.NEWS: Resource.NEWS_DOCUMENTS,
}


@dataclass(frozen=True, slots=True)
class BudgetState:
    resource: Resource
    used: float
    limit: float

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit


class BudgetLedger:
    """Enforces and records an investigation's resource limits."""

    def __init__(
        self,
        repository: BudgetRepository,
        investigation_id: str,
        policy: BudgetPolicy,
        *,
        clock: Any = time.monotonic,
    ) -> None:
        self.repository = repository
        self.investigation_id = investigation_id
        self.policy = policy
        self._clock = clock
        self._started = clock()
        self._runtime_at_start = repository.get(investigation_id, Resource.RUNTIME_SECONDS)

    # -- limits ---------------------------------------------------------
    def limit(self, resource: Resource) -> float:
        if resource is Resource.RUNTIME_SECONDS:
            return float(self.policy.max_runtime_minutes * 60)
        field = _LIMIT_FIELDS.get(resource)
        if field is None:
            # A counter with no ceiling. Input and output tokens are tracked
            # so cost can be computed from them; the ceiling that matters is
            # on their total, and on money.
            return float("inf")
        return float(getattr(self.policy, field))

    def used(self, resource: Resource) -> float:
        if resource is Resource.RUNTIME_SECONDS:
            return self.elapsed_seconds()
        return self.repository.get(self.investigation_id, str(resource))

    def state(self, resource: Resource) -> BudgetState:
        return BudgetState(resource, self.used(resource), self.limit(resource))

    def elapsed_seconds(self) -> float:
        return self._runtime_at_start + (self._clock() - self._started)

    # -- spending -------------------------------------------------------
    def charge_model(self, usage: Any, *, price: tuple[float, float] | None = None) -> float:
        """Record one model call: the call, its tokens, and its cost.

        One place does all of it so the counters cannot disagree with each
        other. ``price`` is (input, output) per million tokens, or None for a
        model the configuration has not priced - in which case the tokens are
        still counted and the money is not guessed at.
        """
        self.try_spend(Resource.MODEL_CALLS)
        input_tokens = float(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = float(getattr(usage, "output_tokens", 0) or 0)
        self.try_spend(Resource.TOKENS, input_tokens + output_tokens)
        self.try_spend(Resource.INPUT_TOKENS, input_tokens)
        self.try_spend(Resource.OUTPUT_TOKENS, output_tokens)
        if price is None:
            return 0.0
        cost = (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
        if cost:
            self.try_spend(Resource.COST, cost)
        return cost

    def can_spend(self, resource: Resource, amount: float = 1) -> bool:
        return self.used(resource) + amount <= self.limit(resource)

    def spend(self, resource: Resource, amount: float = 1) -> float:
        """Record expenditure, raising :class:`BudgetExceeded` if over limit.

        The limit is checked before the counter moves, so a stored count is
        always a count of work actually permitted.
        """
        self.checkpoint_runtime()
        limit = self.limit(resource)
        used = self.used(resource)
        if used + amount > limit:
            raise BudgetExceeded(str(resource), limit, used)
        return self.repository.add(self.investigation_id, str(resource), amount)

    def try_spend(self, resource: Resource, amount: float = 1) -> bool:
        try:
            self.spend(resource, amount)
        except BudgetExceeded:
            return False
        return True

    def spend_document(self, family: SourceFamily) -> None:
        """Charge a document against the global and per-family allowances.

        Both allowances are checked before either is charged, so a document
        refused by the family limit does not consume the global one.
        """
        family_resource = FAMILY_RESOURCE.get(family)
        for resource in filter(None, (Resource.DOCUMENTS, family_resource)):
            state = self.state(resource)
            if state.used + 1 > state.limit:
                raise BudgetExceeded(str(resource), state.limit, state.used)
        self.spend(Resource.DOCUMENTS)
        if family_resource is not None:
            self.spend(family_resource)

    def can_spend_document(self, family: SourceFamily) -> bool:
        if not self.can_spend(Resource.DOCUMENTS):
            return False
        family_resource = FAMILY_RESOURCE.get(family)
        return family_resource is None or self.can_spend(family_resource)

    def checkpoint_runtime(self) -> None:
        """Persist elapsed runtime and enforce the wall-clock limit."""
        elapsed = self.elapsed_seconds()
        self.repository.add(
            self.investigation_id,
            str(Resource.RUNTIME_SECONDS),
            elapsed - self.repository.get(self.investigation_id, str(Resource.RUNTIME_SECONDS)),
        )
        limit = self.limit(Resource.RUNTIME_SECONDS)
        if elapsed > limit:
            raise BudgetExceeded(str(Resource.RUNTIME_SECONDS), limit, elapsed)

    # -- depth ----------------------------------------------------------
    def allows_depth(self, depth: int) -> bool:
        return depth <= self.policy.max_depth

    def allows_citation_depth(self, depth: int) -> bool:
        return depth <= self.policy.max_citation_depth

    # -- reporting ------------------------------------------------------
    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            str(resource): {
                "used": round(self.used(resource), 3),
                "limit": self.limit(resource),
                "remaining": round(self.state(resource).remaining, 3),
            }
            for resource in Resource
        }

    def exhausted_resources(self) -> list[Resource]:
        return [resource for resource in Resource if self.state(resource).exhausted]

    def near_exhaustion(self, threshold: float = 0.85) -> list[Resource]:
        """Resources past ``threshold`` of their limit - a stopping signal."""
        out = []
        for resource in Resource:
            state = self.state(resource)
            if state.limit and state.used / state.limit >= threshold:
                out.append(resource)
        return out
