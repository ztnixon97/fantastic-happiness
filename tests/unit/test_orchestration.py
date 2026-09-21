"""Model tiers, cost accounting, and running tasks concurrently.

These three arrived together because they meet in the same place: the
scheduler decides which model a role gets, what that call costs, and how
many of them run at once.
"""

from __future__ import annotations

import asyncio

import pytest

from research.budgets import BudgetLedger, Resource
from research.config import BudgetPolicy, ModelSettings, ModelTier, ResearchConfig
from research.errors import BudgetExceeded, ConfigError
from research.llm.base import ModelUsage
from research.llm.factory import ModelPool
from research.llm.scripted import ScriptedModel
from research.models.task import Operation, ResearchRole, ResearchTask
from research.orchestration.scheduler import Scheduler
from research.storage.store import ResearchStore


class TestModelTiers:
    def test_one_model_unless_told_otherwise(self) -> None:
        settings = ModelSettings(model="one")
        assert {settings.resolve(role).model for role in ("planner", "academic", None)} == {"one"}

    def test_planning_and_synthesis_get_the_strategic_tier(self) -> None:
        settings = ModelSettings(
            model="middle", fast=ModelTier(model="small"), strategic=ModelTier(model="large")
        )
        assert settings.resolve("planner").model == "large"
        assert settings.resolve("synthesizer").model == "large"
        assert settings.resolve("academic").model == "small"
        assert settings.resolve("skeptic").model == "middle", "attacking is not grunt work"

    def test_a_tier_written_as_a_bare_name_is_a_model(self) -> None:
        settings = ModelSettings.from_dict({"model": "m", "fast": "tiny"})
        assert settings.fast == ModelTier(model="tiny")

    def test_a_tier_inherits_everything_it_does_not_state(self) -> None:
        settings = ModelSettings(
            provider="anthropic", model="m", max_tokens=1000, base_url="http://x",
            fast=ModelTier(model="tiny"),
        )
        resolved = settings.resolve("news")
        assert (resolved.provider, resolved.max_tokens, resolved.base_url) == (
            "anthropic", 1000, "http://x",
        )

    def test_roles_can_be_remapped(self) -> None:
        settings = ModelSettings(
            model="m", fast=ModelTier(model="tiny"), roles={"skeptic": "fast"}
        )
        assert settings.resolve("skeptic").model == "tiny"

    def test_a_role_can_use_a_different_provider_entirely(self) -> None:
        config = ResearchConfig(
            model=ModelSettings(
                provider="anthropic",
                model="claude",
                fast=ModelTier(provider="openai", model="local-thing", base_url="http://local"),
            )
        )
        pool = ModelPool(config)
        with pytest.raises(ConfigError):
            pool.for_role(ResearchRole.PLANNER)  # no anthropic credential here
        gatherer = pool.for_role(ResearchRole.ACADEMIC)
        assert gatherer.spec.model == "local-thing"

    def test_the_pool_builds_one_client_per_tier(self) -> None:
        config = ResearchConfig(
            model=ModelSettings(
                provider="openai", model="m", fast=ModelTier(provider="openai", model="tiny")
            )
        )
        pool = ModelPool(config)
        assert pool.for_role(ResearchRole.ACADEMIC) is pool.for_role(ResearchRole.NEWS)
        assert pool.for_role(ResearchRole.ACADEMIC) is not pool.for_role(ResearchRole.SKEPTIC)

    def test_a_fixed_client_answers_for_every_role(self) -> None:
        model = ScriptedModel(["{}"])
        pool = ModelPool(fixed=model)
        assert pool.for_role(ResearchRole.PLANNER) is model
        assert pool.price_for(ResearchRole.PLANNER) is None


class TestCost:
    def test_tokens_are_counted_whether_or_not_a_price_is_known(
        self, store: ResearchStore, investigation
    ) -> None:
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy())
        cost = ledger.charge_model(ModelUsage(input_tokens=1000, output_tokens=500))
        assert cost == 0.0
        assert ledger.used(Resource.TOKENS) == 1500
        assert ledger.used(Resource.INPUT_TOKENS) == 1000
        assert ledger.used(Resource.OUTPUT_TOKENS) == 500
        assert ledger.used(Resource.COST) == 0.0

    def test_a_priced_model_reports_money(
        self, store: ResearchStore, investigation
    ) -> None:
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy())
        # $3 per million in, $15 per million out.
        cost = ledger.charge_model(
            ModelUsage(input_tokens=1_000_000, output_tokens=100_000), price=(3.0, 15.0)
        )
        assert cost == pytest.approx(4.5)
        assert ledger.used(Resource.COST) == pytest.approx(4.5)

    def test_there_is_no_built_in_price_table(self) -> None:
        """A stale price reported as this run's cost would be a made-up figure."""
        assert ModelSettings().prices == {}
        assert ModelSettings().price("claude-sonnet-5") is None

    def test_money_can_be_a_ceiling(self, store: ResearchStore, investigation) -> None:
        ledger = BudgetLedger(
            store.budget, investigation.id, BudgetPolicy(max_cost=0.01)
        )
        ledger.charge_model(ModelUsage(input_tokens=2_000, output_tokens=0), price=(3.0, 15.0))
        with pytest.raises(BudgetExceeded):
            for _ in range(50):
                ledger.spend(Resource.COST, 0.004)

    def test_unlimited_counters_do_not_break_the_snapshot(
        self, store: ResearchStore, investigation
    ) -> None:
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy())
        snapshot = ledger.snapshot()
        assert snapshot["input_tokens"]["limit"] == float("inf")
        assert ledger.can_spend(Resource.INPUT_TOKENS, 10**9)


def _task(store: ResearchStore, investigation_id: str, objective: str, role=ResearchRole.SCOUT):
    return store.tasks.create(
        ResearchTask(
            id="",
            investigation_id=investigation_id,
            role=role,
            operation=Operation.SEARCH_WEB,
            objective=objective,
            depth=1,
        )
    )


class TestClaimingTasks:
    def test_a_claimed_task_is_not_offered_twice(
        self, store: ResearchStore, investigation
    ) -> None:
        for index in range(3):
            _task(store, investigation.id, f"objective {index}")

        first = store.tasks.claim_pending(investigation.id, limit=2)
        second = store.tasks.claim_pending(investigation.id, limit=2)

        assert len(first) == 2
        assert len(second) == 1
        assert {task.id for task in first} & {task.id for task in second} == set()

    def test_claiming_marks_them_running(self, store: ResearchStore, investigation) -> None:
        _task(store, investigation.id, "objective")
        claimed = store.tasks.claim_pending(investigation.id, limit=5)
        assert str(store.tasks.get(claimed[0].id).status) == "running"

    def test_priority_still_decides_the_order(
        self, store: ResearchStore, investigation
    ) -> None:
        low = store.tasks.create(
            ResearchTask(
                id="", investigation_id=investigation.id, role=ResearchRole.SCOUT,
                operation=Operation.SEARCH_WEB, objective="later", depth=1, priority=9,
            )
        )
        high = store.tasks.create(
            ResearchTask(
                id="", investigation_id=investigation.id, role=ResearchRole.SCOUT,
                operation=Operation.SEARCH_WEB, objective="sooner", depth=1, priority=1,
            )
        )
        claimed = store.tasks.claim_pending(investigation.id, limit=1)
        assert [task.id for task in claimed] == [high.id]
        assert low.id not in {task.id for task in claimed}

    def test_asking_for_nothing_returns_nothing(
        self, store: ResearchStore, investigation
    ) -> None:
        _task(store, investigation.id, "objective")
        assert store.tasks.claim_pending(investigation.id, limit=0) == []


class TestConcurrency:
    """The scheduler's fan-out, with a worker that records the overlap."""

    def _scheduler(self, store, investigation, registry, *, concurrency: int, running: list):
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy())
        scheduler = Scheduler(
            store,
            registry,
            ScriptedModel(["{}"]),
            investigation_id=investigation.id,
            ledger=ledger,
            max_concurrent_tasks=concurrency,
        )

        class Recorder:
            """Stands in for a worker, and notes how many ran at once."""

            def __init__(self, *args, **kwargs) -> None:
                self.task = None

            async def run(self, task):
                running.append(len(running) + 1)
                peak = len(running)
                await asyncio.sleep(0.01)
                running.pop()
                from research.agents.worker import WorkerOutcome
                from research.models.task import TaskResult, TaskStatus

                store.tasks.finish(task.id, TaskStatus.COMPLETED)
                return WorkerOutcome(
                    task=task,
                    result=TaskResult(summary=f"peak {peak}"),
                    status=TaskStatus.COMPLETED,
                    steps=1,
                )

        return scheduler, Recorder

    async def test_tasks_run_in_parallel_up_to_the_limit(
        self, store: ResearchStore, investigation, monkeypatch
    ) -> None:
        from research.sources.registry import SourceRegistry

        for index in range(6):
            _task(store, investigation.id, f"objective {index}")
        running: list[int] = []
        scheduler, Recorder = self._scheduler(
            store, investigation, SourceRegistry(), concurrency=3, running=running
        )
        monkeypatch.setattr("research.orchestration.scheduler.ResearchWorker", Recorder)
        monkeypatch.setattr(scheduler.planner, "plan", _no_plan)

        run = await scheduler.run()

        assert run.tasks_run == 6
        peaks = [int(outcome.result.summary.split()[-1]) for outcome in run.outcomes]
        assert max(peaks) == 3, "three at a time, not one and not six"

    async def test_one_at_a_time_is_still_available(
        self, store: ResearchStore, investigation, monkeypatch
    ) -> None:
        from research.sources.registry import SourceRegistry

        for index in range(3):
            _task(store, investigation.id, f"objective {index}")
        running: list[int] = []
        scheduler, Recorder = self._scheduler(
            store, investigation, SourceRegistry(), concurrency=1, running=running
        )
        monkeypatch.setattr("research.orchestration.scheduler.ResearchWorker", Recorder)
        monkeypatch.setattr(scheduler.planner, "plan", _no_plan)

        run = await scheduler.run()
        peaks = [int(outcome.result.summary.split()[-1]) for outcome in run.outcomes]
        assert max(peaks) == 1

    async def test_the_task_limit_is_not_overshot_by_the_batch(
        self, store: ResearchStore, investigation, monkeypatch
    ) -> None:
        from research.sources.registry import SourceRegistry

        for index in range(10):
            _task(store, investigation.id, f"objective {index}")
        running: list[int] = []
        scheduler, Recorder = self._scheduler(
            store, investigation, SourceRegistry(), concurrency=4, running=running
        )
        monkeypatch.setattr("research.orchestration.scheduler.ResearchWorker", Recorder)
        monkeypatch.setattr(scheduler.planner, "plan", _no_plan)

        run = await scheduler.run(max_tasks=6)
        assert run.tasks_run == 6

    async def test_a_sibling_failing_does_not_lose_the_work_that_succeeded(
        self, store: ResearchStore, investigation, monkeypatch
    ) -> None:
        """One task hitting the budget must not discard its batch-mates."""
        from research.agents.worker import WorkerOutcome
        from research.models.task import TaskResult, TaskStatus
        from research.sources.registry import SourceRegistry

        for index in range(3):
            _task(store, investigation.id, f"objective {index}")
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy())
        scheduler = Scheduler(
            store,
            SourceRegistry(),
            ScriptedModel(["{}"]),
            investigation_id=investigation.id,
            ledger=ledger,
            max_concurrent_tasks=3,
        )
        monkeypatch.setattr(scheduler.planner, "plan", _no_plan)

        class OneFails:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def run(self, task):
                if task.objective.endswith("1"):
                    raise BudgetExceeded("documents", 10, 10)
                store.tasks.finish(task.id, TaskStatus.COMPLETED)
                return WorkerOutcome(
                    task=task, result=TaskResult(summary="ok"),
                    status=TaskStatus.COMPLETED, steps=1,
                )

        monkeypatch.setattr("research.orchestration.scheduler.ResearchWorker", OneFails)
        run = await scheduler.run()

        assert run.tasks_run == 2, "the two that finished are recorded"
        assert str(run.stop_reason) == "budget_exhausted"


async def _no_plan(investigation, *, max_tasks: int = 5):
    from research.agents.planner import Plan

    return Plan(brief="tasks were seeded by the test")
