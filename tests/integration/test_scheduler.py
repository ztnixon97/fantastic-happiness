"""Stopping rules, recursion, and a whole autonomous run offline."""

from __future__ import annotations

import pytest

from research.agents.offline_model import offline_research_model
from research.budgets import BudgetLedger, Resource
from research.config import AcquisitionPolicy, BudgetPolicy, ResearchConfig
from research.graph.independence import independent_documents
from research.llm.scripted import ScriptedModel
from research.models.common import Provenance, SourceFamily, SourceType
from research.models.investigation import InvestigationStatus, StopReason
from research.models.task import Operation, ResearchRole, ResearchTask, TaskStatus
from research.normalize.document import build_document
from research.acquisition.pipeline import EvidenceAcquirer
from research.orchestration.scheduler import Scheduler
from research.orchestration.stopping import StoppingRules
from research.sources.offline import build_offline_registry, load_corpus
from research.storage.store import ResearchStore


@pytest.fixture
def config() -> ResearchConfig:
    return ResearchConfig(
        acquisition=AcquisitionPolicy(per_host_min_interval_seconds=0.0, retry_backoff_seconds=0.0)
    )


@pytest.fixture
def offline_environment(config):
    corpus = load_corpus()
    registry, client = build_offline_registry(corpus, config)
    with ResearchStore.in_memory() as store:
        investigation = store.investigations.create(
            corpus["question"], budget=config.budget.to_dict()
        )
        yield store, registry, investigation, config


def ledger_for(store, investigation, policy: BudgetPolicy | None = None) -> BudgetLedger:
    return BudgetLedger(store.budget, investigation.id, policy or BudgetPolicy())


class TestStoppingRules:
    def test_no_open_tasks_stops_the_run(self, store, investigation) -> None:
        rules = StoppingRules(store, investigation.id, ledger_for(store, investigation))
        decision = rules.evaluate()
        assert decision.should_stop
        assert decision.reason is StopReason.NO_OPEN_TASKS

    def test_a_pending_task_keeps_it_going(self, store, investigation) -> None:
        store.tasks.create(
            ResearchTask(
                id="",
                investigation_id=investigation.id,
                role=ResearchRole.SCOUT,
                operation=Operation.SEARCH_WEB,
                objective="keep going",
            )
        )
        rules = StoppingRules(store, investigation.id, ledger_for(store, investigation))
        assert not rules.evaluate().should_stop

    def test_budget_exhaustion_outranks_everything(self, store, investigation) -> None:
        ledger = ledger_for(store, investigation, BudgetPolicy(max_documents=1))
        ledger.spend(Resource.DOCUMENTS)
        store.tasks.create(
            ResearchTask(
                id="",
                investigation_id=investigation.id,
                role=ResearchRole.SCOUT,
                operation=Operation.SEARCH_WEB,
                objective="more work",
            )
        )
        decision = StoppingRules(store, investigation.id, ledger).evaluate()
        assert decision.reason is StopReason.BUDGET_EXHAUSTED
        assert "documents" in decision.detail

    def test_runtime_exhaustion_is_reported_as_such(self, store, investigation) -> None:
        clock = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0])
        ledger = BudgetLedger(
            store.budget,
            investigation.id,
            BudgetPolicy(max_runtime_minutes=1),
            clock=lambda: next(clock, 10_000.0),
        )
        decision = StoppingRules(store, investigation.id, ledger).evaluate()
        assert decision.reason is StopReason.RUNTIME_EXHAUSTED

    def test_a_corpus_of_copies_is_diminishing_returns(self, store, investigation) -> None:
        acquirer = EvidenceAcquirer(store)
        wire = (
            "WASHINGTON (Reuters) - The utility group said it had agreed to terminate the "
            "flagship reactor project after subscription levels fell short of the amount "
            "needed to proceed with construction at the site."
        )
        for index in range(12):
            acquirer.persist(
                build_document(
                    provider="test",
                    source_type=SourceType.SECONDARY_NEWS_REPORTING,
                    source_family=SourceFamily.NEWS,
                    provenance=Provenance(provider="test"),
                    title="Project terminated",
                    text=wire,
                    url=f"https://outlet{index}.test/story",
                ),
                investigation_id=investigation.id,
            )
        store.tasks.create(
            ResearchTask(
                id="",
                investigation_id=investigation.id,
                role=ResearchRole.NEWS,
                operation=Operation.SEARCH_NEWS,
                objective="more of the same",
            )
        )
        decision = StoppingRules(store, investigation.id, ledger_for(store, investigation)).evaluate()
        assert decision.reason is StopReason.DIMINISHING_RETURNS
        assert "copies or rewrites" in decision.detail


class TestScheduler:
    async def test_a_full_offline_investigation_runs_and_stops(self, offline_environment) -> None:
        store, registry, investigation, config = offline_environment
        ledger = ledger_for(store, investigation)
        scheduler = Scheduler(
            store,
            registry,
            offline_research_model(),
            investigation_id=investigation.id,
            ledger=ledger,
            max_steps_per_task=6,
        )
        run = await scheduler.run()

        assert run.plan is not None and len(run.plan.tasks) >= 3
        assert run.tasks_run >= 3
        assert run.stop_reason in (
            StopReason.NO_OPEN_TASKS,
            StopReason.BUDGET_EXHAUSTED,
            StopReason.DIMINISHING_RETURNS,
            StopReason.EVIDENCE_SUFFICIENT,
        )

        documents = store.documents.list(investigation.id, limit=500)
        claims = store.claims.list(investigation.id, hydrate=False)
        assert documents and claims
        # Mixed corpus: the roles worked different source families.
        families = {document.source_family for document in documents}
        assert SourceFamily.ACADEMIC in families and SourceFamily.NEWS in families

    async def test_the_run_is_recorded_on_the_investigation(self, offline_environment) -> None:
        store, registry, investigation, _ = offline_environment
        scheduler = Scheduler(
            store,
            registry,
            offline_research_model(),
            investigation_id=investigation.id,
            ledger=ledger_for(store, investigation),
            max_steps_per_task=4,
        )
        run = await scheduler.run(max_tasks=2)
        reloaded = store.investigations.get(investigation.id)
        assert reloaded.status is InvestigationStatus.COMPLETED
        assert reloaded.stop_reason is not None
        assert reloaded.stop_detail
        # Nothing is left pending: a stopped investigation is stopped.
        assert store.tasks.list(investigation.id, status=TaskStatus.PENDING) == []
        assert run.summary()["tasks_run"] == 2

    async def test_recursion_is_bounded_by_depth(self, offline_environment) -> None:
        store, registry, investigation, _ = offline_environment
        ledger = ledger_for(store, investigation, BudgetPolicy(max_depth=2, max_tasks=12))
        scheduler = Scheduler(
            store,
            registry,
            offline_research_model(),
            investigation_id=investigation.id,
            ledger=ledger,
            max_steps_per_task=6,
        )
        await scheduler.run()
        depths = [task.depth for task in store.tasks.list(investigation.id, limit=100)]
        assert depths and max(depths) <= 2

    async def test_follow_ups_become_child_tasks(self, offline_environment) -> None:
        store, registry, investigation, _ = offline_environment
        scheduler = Scheduler(
            store,
            registry,
            offline_research_model(),
            investigation_id=investigation.id,
            ledger=ledger_for(store, investigation),
            max_steps_per_task=6,
        )
        await scheduler.run()
        tasks = store.tasks.list(investigation.id, limit=100)
        children = [task for task in tasks if task.parent_task_id]
        assert children, "workers proposed follow-ups and they were planned"
        for child in children:
            parent = store.tasks.get(child.parent_task_id)
            assert child.depth == parent.depth + 1

    async def test_the_task_budget_terminates_recursion(self, offline_environment) -> None:
        store, registry, investigation, _ = offline_environment
        ledger = ledger_for(store, investigation, BudgetPolicy(max_tasks=3))
        scheduler = Scheduler(
            store,
            registry,
            offline_research_model(),
            investigation_id=investigation.id,
            ledger=ledger,
            max_steps_per_task=4,
        )
        run = await scheduler.run()
        assert store.tasks.count(investigation.id) <= 3
        assert run.tasks_run <= 3

    async def test_a_failing_model_stops_the_run_with_a_reason(self, offline_environment) -> None:
        store, registry, investigation, _ = offline_environment
        scheduler = Scheduler(
            store,
            registry,
            ScriptedModel([]),  # exhausted immediately
            investigation_id=investigation.id,
            ledger=ledger_for(store, investigation),
        )
        run = await scheduler.run()
        assert run.stop_reason is StopReason.ERROR
        assert run.error
        assert store.investigations.get(investigation.id).status is InvestigationStatus.FAILED

    async def test_evidence_keeps_its_provenance_through_an_autonomous_run(
        self, offline_environment
    ) -> None:
        store, registry, investigation, _ = offline_environment
        scheduler = Scheduler(
            store,
            registry,
            offline_research_model(),
            investigation_id=investigation.id,
            ledger=ledger_for(store, investigation),
            max_steps_per_task=6,
        )
        await scheduler.run(max_tasks=3)
        documents = store.documents.list(investigation.id, limit=500)
        assert documents
        for document in documents:
            assert document.provenance is not None
            assert document.provenance.task_id or document.provenance.query_id

    async def test_claims_made_autonomously_quote_their_sources(
        self, offline_environment
    ) -> None:
        store, registry, investigation, _ = offline_environment
        scheduler = Scheduler(
            store,
            registry,
            offline_research_model(),
            investigation_id=investigation.id,
            ledger=ledger_for(store, investigation),
            max_steps_per_task=6,
        )
        await scheduler.run(max_tasks=4)
        claims = store.claims.list(investigation.id, hydrate=False)
        assert claims
        quoted = [
            link
            for claim in claims
            for link in store.claims.evidence_links(claim.id)
            if link.excerpt
        ]
        assert quoted, "at least one claim carries a verbatim excerpt"
        for link in quoted:
            document = store.documents.get(link.document_id)
            from research.operations.claims import excerpt_appears_in

            assert excerpt_appears_in(document, link.excerpt)

    async def test_independent_sources_are_still_counted_honestly(
        self, offline_environment
    ) -> None:
        store, registry, investigation, _ = offline_environment
        scheduler = Scheduler(
            store,
            registry,
            offline_research_model(),
            investigation_id=investigation.id,
            ledger=ledger_for(store, investigation),
            max_steps_per_task=6,
        )
        await scheduler.run()
        documents = store.documents.list(investigation.id, limit=500)
        groups = independent_documents(store.documents, [d.id for d in documents])
        assert len(groups) < len(documents)
