"""Budget enforcement and stopping signals."""

from __future__ import annotations

import pytest

from research.config import BudgetPolicy, load_config
from research.errors import BudgetExceeded
from research.models.common import SourceFamily
from research.orchestration.budgets import BudgetLedger, Resource
from research.storage.store import ResearchStore


@pytest.fixture
def clock():
    class Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    return Clock()


def make_ledger(store, investigation_id, policy, clock=None) -> BudgetLedger:
    return BudgetLedger(
        store.budget, investigation_id, policy, clock=clock or (lambda: 0.0)
    )


class TestSpending:
    def test_spending_stops_at_the_limit(self, store, investigation) -> None:
        ledger = make_ledger(store, investigation.id, BudgetPolicy(max_searches=2))
        assert ledger.try_spend(Resource.SEARCHES)
        assert ledger.try_spend(Resource.SEARCHES)
        assert not ledger.try_spend(Resource.SEARCHES)
        with pytest.raises(BudgetExceeded) as exc:
            ledger.spend(Resource.SEARCHES)
        assert exc.value.resource == "searches"

    def test_a_refused_spend_does_not_move_the_counter(self, store, investigation) -> None:
        ledger = make_ledger(store, investigation.id, BudgetPolicy(max_searches=1))
        ledger.spend(Resource.SEARCHES)
        ledger.try_spend(Resource.SEARCHES)
        assert ledger.used(Resource.SEARCHES) == 1

    def test_family_limits_are_charged_alongside_the_global_one(self, store, investigation) -> None:
        ledger = make_ledger(
            store, investigation.id, BudgetPolicy(max_documents=10, max_academic_documents=1)
        )
        ledger.spend_document(SourceFamily.ACADEMIC)
        with pytest.raises(BudgetExceeded):
            ledger.spend_document(SourceFamily.ACADEMIC)
        # The refusal must not have consumed the global document allowance.
        assert ledger.used(Resource.DOCUMENTS) == 1
        ledger.spend_document(SourceFamily.NEWS)
        assert ledger.used(Resource.DOCUMENTS) == 2

    def test_families_without_a_sub_budget_only_charge_the_global_one(self, store, investigation) -> None:
        ledger = make_ledger(store, investigation.id, BudgetPolicy(max_documents=2))
        ledger.spend_document(SourceFamily.WEB)
        assert ledger.used(Resource.DOCUMENTS) == 1
        assert ledger.can_spend_document(SourceFamily.WEB)


class TestPersistence:
    def test_usage_survives_a_restart(self, tmp_path) -> None:
        policy = BudgetPolicy(max_searches=3)
        path = tmp_path / "r.sqlite3"
        with ResearchStore.open(path) as store:
            investigation = store.investigations.create("q")
            first = make_ledger(store, investigation.id, policy)
            first.spend(Resource.SEARCHES)
            first.spend(Resource.SEARCHES)
        with ResearchStore.open(path) as store:
            resumed = make_ledger(store, investigation.id, policy)
            assert resumed.used(Resource.SEARCHES) == 2
            assert resumed.try_spend(Resource.SEARCHES)
            assert not resumed.try_spend(Resource.SEARCHES)


class TestRuntimeAndDepth:
    def test_runtime_limit_terminates_work(self, store, investigation, clock) -> None:
        ledger = make_ledger(
            store, investigation.id, BudgetPolicy(max_runtime_minutes=1), clock=clock
        )
        clock.now = 59
        ledger.checkpoint_runtime()
        clock.now = 61
        with pytest.raises(BudgetExceeded) as exc:
            ledger.checkpoint_runtime()
        assert exc.value.resource == "runtime_seconds"

    def test_elapsed_runtime_accumulates_across_sessions(self, store, investigation, clock) -> None:
        policy = BudgetPolicy(max_runtime_minutes=10)
        first = make_ledger(store, investigation.id, policy, clock=clock)
        clock.now = 30
        first.checkpoint_runtime()
        second_clock = type(clock)()
        second = BudgetLedger(store.budget, investigation.id, policy, clock=second_clock)
        second_clock.now = 10
        assert second.elapsed_seconds() == pytest.approx(40)

    def test_depth_limits_bound_recursion(self, store, investigation) -> None:
        ledger = make_ledger(
            store, investigation.id, BudgetPolicy(max_depth=2, max_citation_depth=1)
        )
        assert ledger.allows_depth(2)
        assert not ledger.allows_depth(3)
        assert ledger.allows_citation_depth(1)
        assert not ledger.allows_citation_depth(2)


class TestReporting:
    def test_snapshot_covers_every_resource(self, store, investigation) -> None:
        ledger = make_ledger(store, investigation.id, BudgetPolicy())
        snapshot = ledger.snapshot()
        assert set(snapshot) == {str(resource) for resource in Resource}
        assert snapshot["searches"]["remaining"] == BudgetPolicy().max_searches

    def test_near_exhaustion_is_a_stopping_signal(self, store, investigation) -> None:
        ledger = make_ledger(store, investigation.id, BudgetPolicy(max_searches=10))
        for _ in range(9):
            ledger.spend(Resource.SEARCHES)
        assert Resource.SEARCHES in ledger.near_exhaustion()
        assert Resource.SEARCHES not in ledger.exhausted_resources()
        ledger.spend(Resource.SEARCHES)
        assert Resource.SEARCHES in ledger.exhausted_resources()


class TestConfiguration:
    def test_yaml_config_sets_budgets(self, tmp_path) -> None:
        path = tmp_path / "research.yaml"
        path.write_text(
            "research:\n  budget:\n    max_depth: 2\n    max_documents: 40\n"
            "providers:\n  arxiv:\n    enabled: false\n",
            encoding="utf-8",
        )
        config = load_config(path, environ={})
        assert config.budget.max_depth == 2
        assert config.budget.max_documents == 40
        assert not config.is_enabled("arxiv")
        assert config.is_enabled("openalex")

    def test_only_allowlisted_environment_variables_are_read(self) -> None:
        config = load_config(
            environ={
                "RESEARCH_BRAVE_API_KEY": "brave-key",
                "AWS_SECRET_ACCESS_KEY": "should-not-be-visible",
                "OPENAI_API_KEY": "should-not-be-visible",
            }
        )
        assert config.secret("brave") == "brave-key"
        assert config.secret("tavily") is None
        assert "should-not-be-visible" not in repr(config.to_dict())

    def test_serialised_config_never_carries_secrets(self) -> None:
        config = load_config(environ={"RESEARCH_BRAVE_API_KEY": "brave-key"})
        assert "brave-key" not in str(config.to_dict())
