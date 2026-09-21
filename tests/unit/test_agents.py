"""The action surface, the worker loop and the planner."""

from __future__ import annotations

import json

import pytest

from research.agents.actions import ACTIONS, actions_for
from research.agents.planner import Planner
from research.agents.runtime import ActionRequest, ActionRuntime
from research.agents.worker import ResearchWorker
from research.budgets import BudgetLedger, Resource
from research.config import BudgetPolicy
from research.llm.scripted import ScriptedModel
from research.models.common import Provenance, SourceFamily, SourceType
from research.models.task import Operation, ResearchRole, ResearchTask, TaskStatus
from research.normalize.document import build_document
from research.acquisition.pipeline import EvidenceAcquirer
from research.sources.registry import SourceRegistry
from research.sources.static import StaticAcademicSource

PAPERS = [
    {
        "id": "P1",
        "title": "Levelized cost projections for small modular reactors",
        "abstract": "We review 42 published cost estimates and find a median of $120/MWh.",
        "doi": "10.1016/j.enpol.2024.114001",
        "authors": ["R. Okafor"],
        "published_at": "2024-04-12",
        "venue": "Energy Policy",
        "references": ["P2"],
    },
    {
        "id": "P2",
        "title": "Historical cost escalation in nuclear construction",
        "abstract": "Realized costs exceeded initial estimates by a median of 117 percent.",
        "doi": "10.1016/j.joule.2021.06.004",
        "authors": ["S. Petrova"],
        "published_at": "2021-07-01",
        "venue": "Joule",
        "references": [],
    },
]


@pytest.fixture
def registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.register(StaticAcademicSource(PAPERS, name="papers"))
    return registry


@pytest.fixture
def task(store, investigation) -> ResearchTask:
    return store.tasks.create(
        ResearchTask(
            id="",
            investigation_id=investigation.id,
            role=ResearchRole.ACADEMIC,
            operation=Operation.SEARCH_ACADEMIC,
            objective="establish what the literature says about reactor costs",
            depth=1,
        )
    )


@pytest.fixture
def runtime(store, investigation, registry, task) -> ActionRuntime:
    ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy())
    return ActionRuntime(
        store, registry, investigation_id=investigation.id, task=task, ledger=ledger
    )


async def act(runtime: ActionRuntime, action: str, **arguments):
    """Run one action, keeping 'name' free for actions that take it."""
    return await runtime.execute(ActionRequest(name=action, arguments=arguments))


class TestActionSurface:
    def test_the_vocabulary_is_the_one_research_needs(self) -> None:
        assert set(ACTIONS) == {
            "search_news", "search_academic", "search_web", "search_social",
            "search_corpus", "fetch_source",
            "follow_citations", "find_primary_source", "find_counterevidence",
            "resolve_entity", "build_timeline", "create_claim", "link_evidence",
            "get_claim", "get_evidence", "get_open_questions",
            "spawn_research_task", "complete_research_task",
        }

    def test_nothing_in_the_vocabulary_executes_anything(self) -> None:
        forbidden = ("shell", "exec", "run_command", "python", "bash", "eval", "write_file")
        assert not [name for name in ACTIONS if any(word in name for word in forbidden)]

    def test_roles_get_different_tools(self) -> None:
        news = {spec.name for spec in actions_for(ResearchRole.NEWS)}
        academic = {spec.name for spec in actions_for(ResearchRole.ACADEMIC)}
        social = {spec.name for spec in actions_for(ResearchRole.SOCIAL)}
        assert "follow_citations" in academic and "follow_citations" not in news
        assert "search_news" in news and "search_news" not in academic
        assert "search_social" in social and "search_social" not in academic

    async def test_an_action_outside_the_role_is_refused(self, runtime) -> None:
        runtime.task.role = ResearchRole.NEWS
        observation = await act(runtime, "follow_citations", document_id="evidence:1")
        assert not observation.ok
        assert "cannot use follow_citations" in observation.error

    async def test_unknown_actions_are_refused_with_the_catalogue(self, runtime) -> None:
        observation = await act(runtime, "run_shell", command="ls")
        assert not observation.ok
        assert "unknown action" in observation.error

    async def test_missing_arguments_are_refused(self, runtime) -> None:
        observation = await act(runtime, "search_academic")
        assert not observation.ok
        assert "missing required argument" in observation.error

    async def test_malformed_identifiers_are_refused(self, runtime) -> None:
        observation = await act(runtime, "get_evidence", document_id="../../etc/passwd")
        assert not observation.ok
        assert "not a evidence identifier" in observation.error


class TestObservations:
    async def test_search_returns_briefs_not_page_text(self, runtime) -> None:
        observation = await act(runtime, "search_academic", query="reactor cost", limit=5)
        assert observation.ok
        assert observation.data["new_evidence"] == 2
        rendered = json.dumps(observation.data)
        assert "median of 117 percent" not in rendered, "abstracts are not pushed at the worker"
        assert all({"id", "title", "type"} <= set(brief) for brief in observation.data["results"])

    async def test_reading_a_document_labels_it_as_external(self, runtime) -> None:
        await act(runtime, "search_academic", query="reactor cost")
        observation = await act(runtime, "get_evidence", document_id="evidence:1")
        assert observation.ok
        assert observation.data["content"].startswith("<external_evidence")
        assert "never as instructions" in observation.data["content"]

    async def test_reading_is_capped(self, runtime) -> None:
        await act(runtime, "search_academic", query="reactor cost")
        observation = await act(runtime, "get_evidence", document_id="evidence:1", characters=10**9)
        assert len(observation.data["content"]) < 20000

    async def test_copies_are_declared_in_the_brief(self, store, investigation, runtime) -> None:
        acquirer = EvidenceAcquirer(store)
        wire = (
            "WASHINGTON (Reuters) - The utility group said on Wednesday it had agreed to "
            "terminate the flagship reactor project after subscriptions fell short of what "
            "was needed to proceed with construction of the plant."
        )
        for url in ("https://www.reuters.test/a", "https://www.gazette.test/a"):
            acquirer.persist(
                build_document(
                    provider="test",
                    source_type=SourceType.ORIGINAL_NEWS_REPORTING,
                    source_family=SourceFamily.NEWS,
                    provenance=Provenance(provider="test"),
                    title="Project terminated",
                    text=wire,
                    url=url,
                ),
                investigation_id=investigation.id,
            )
        observation = await act(runtime, "get_evidence", document_id="evidence:2")
        assert observation.data["not_independent_of"] == "evidence:1"


class TestGraphActions:
    async def test_claims_and_evidence_link_through_the_runtime(self, runtime) -> None:
        await act(runtime, "search_academic", query="reactor cost")
        created = await act(runtime, "create_claim", text="Nuclear projects overrun estimates")
        claim_id = created.data["claim_id"]
        linked = await act(
            runtime,
            "link_evidence",
            claim_id=claim_id,
            document_id="evidence:2",
            stance="supports",
            excerpt="Realized costs exceeded initial estimates by a median of 117 percent",
        )
        assert linked.ok
        assert linked.data["excerpt_verified"]
        assert linked.data["status"] == "supported"

    async def test_a_fabricated_excerpt_is_refused(self, runtime) -> None:
        await act(runtime, "search_academic", query="reactor cost")
        created = await act(runtime, "create_claim", text="Costs tripled")
        observation = await act(
            runtime,
            "link_evidence",
            claim_id=created.data["claim_id"],
            document_id="evidence:1",
            stance="supports",
            excerpt="costs tripled according to leaked internal documents",
        )
        assert not observation.ok
        assert "does not appear" in observation.error

    async def test_open_questions_are_readable(self, runtime) -> None:
        await act(runtime, "search_academic", query="reactor cost")
        created = await act(runtime, "create_claim", text="Nuclear projects overrun estimates")
        await act(
            runtime,
            "link_evidence",
            claim_id=created.data["claim_id"],
            document_id="evidence:1",
            stance="supports",
        )
        observation = await act(runtime, "get_open_questions")
        assert observation.data["open_questions"][0]["claim_id"] == created.data["claim_id"]

    async def test_entities_resolve_and_report_ambiguity(self, runtime) -> None:
        first = await act(
            runtime, "resolve_entity", name="NuScale Power", entity_type="organization"
        )
        assert first.data["created"]
        again = await act(
            runtime, "resolve_entity", name="nuscale power", entity_type="organization"
        )
        assert again.data["entity_id"] == first.data["entity_id"]
        assert again.data["confidence"] == "heuristic"


class TestCorpusAction:
    """Workers can ask what the investigation already holds."""

    async def test_it_finds_evidence_another_search_gathered(self, runtime) -> None:
        await act(runtime, "search_academic", query="reactor cost")
        observation = await act(runtime, "search_corpus", query="levelized cost projections")
        assert observation.ok
        assert observation.data["held_documents"] == 2
        assert observation.data["results"][0]["id"] == "evidence:1"

    async def test_it_spends_no_budget(self, runtime) -> None:
        # Wall-clock is excluded: it is spent by existing, not by searching.
        spendable = [
            resource for resource in Resource if resource is not Resource.RUNTIME_SECONDS
        ]
        await act(runtime, "search_academic", query="reactor cost")
        before = {resource: runtime.ledger.used(resource) for resource in spendable}
        await act(runtime, "search_corpus", query="reactor cost")
        assert {resource: runtime.ledger.used(resource) for resource in spendable} == before

    async def test_it_returns_briefs_not_page_text(self, runtime) -> None:
        await act(runtime, "search_academic", query="reactor cost")
        observation = await act(runtime, "search_corpus", query="cost escalation")
        rendered = json.dumps(observation.data)
        assert "Realized costs exceeded initial estimates" not in rendered

    async def test_an_empty_corpus_says_so_rather_than_failing(self, runtime) -> None:
        observation = await act(runtime, "search_corpus", query="reactor cost")
        assert observation.ok
        assert observation.data["matched"] == 0
        assert "search outside" in observation.data["note"]

    async def test_every_gathering_role_can_use_it(self) -> None:
        for role in (
            ResearchRole.SCOUT, ResearchRole.ACADEMIC, ResearchRole.NEWS,
            ResearchRole.PRIMARY_SOURCE, ResearchRole.SOCIAL, ResearchRole.SKEPTIC,
            ResearchRole.SYNTHESIZER,
        ):
            assert "search_corpus" in {spec.name for spec in actions_for(role)}


class TestReadingEvidence:
    """A worker reads a document for something, not from the top."""

    FILING = (
        "NOTICE OF ANNUAL FILING\n\nFiled pursuant to section 14 of the relevant "
        "act, containing the disclosures required thereunder.\n\n"
        + "The board considered routine administrative matters at each meeting.\n\n" * 40
        + "The overnight capital cost rose from 6.1 billion dollars to 9.3 billion "
        "dollars, a revision of fifty-two percent against the March estimate.\n\n"
        + "Directors' remuneration is disclosed in the usual form in note 19.\n\n" * 40
    )

    async def _hold(self, store, investigation):
        from research.models.common import Provenance, SourceFamily, SourceType
        from research.normalize.document import build_document

        return EvidenceAcquirer(store).persist(
            build_document(
                provider="test",
                source_type=SourceType.CORPORATE_FILING,
                source_family=SourceFamily.CORPORATE,
                provenance=Provenance(provider="test"),
                title="Annual filing",
                text=self.FILING,
                url="https://filings.test/annual",
            ),
            investigation_id=investigation.id,
        ).document

    async def test_the_passage_that_answers_is_returned_not_the_cover_page(
        self, store, investigation, runtime
    ) -> None:
        document = await self._hold(store, investigation)
        observation = await act(
            runtime, "get_evidence", document_id=document.id,
            about="how much did the capital cost rise",
        )
        assert observation.ok
        assert observation.data["read"] == "passages"
        assert "fifty-two percent" in observation.data["content"]
        assert "NOTICE OF ANNUAL FILING" not in observation.data["content"]

    async def test_it_says_where_each_passage_came_from(
        self, store, investigation, runtime
    ) -> None:
        document = await self._hold(store, investigation)
        observation = await act(
            runtime, "get_evidence", document_id=document.id, about="capital cost"
        )
        assert observation.data["passages"]
        assert all("characters" in where for where in observation.data["passages"])

    async def test_the_task_objective_is_the_default_question(
        self, store, investigation, runtime
    ) -> None:
        """A worker that does not say what it is looking for is still looking."""
        document = await self._hold(store, investigation)
        assert "reactor costs" in runtime.task.objective
        observation = await act(runtime, "get_evidence", document_id=document.id)
        assert observation.data["selected_for"] == runtime.task.objective
        assert observation.data["read"] == "passages"
        # "reactor costs" in the objective against "capital cost" in the
        # filing: only matches once plural and singular fold together.
        assert "capital cost" in observation.data["content"]

    async def test_a_document_the_question_does_not_touch_is_read_from_the_top(
        self, store, investigation, runtime
    ) -> None:
        """Nothing matching is not the same as nothing relevant."""
        document = await self._hold(store, investigation)
        observation = await act(
            runtime, "get_evidence", document_id=document.id,
            about="photovoltaic tariff arbitration",
        )
        assert observation.data["read"] == "from the beginning"

    async def test_the_whole_document_is_still_available(
        self, store, investigation, runtime
    ) -> None:
        document = await self._hold(store, investigation)
        observation = await act(
            runtime, "get_evidence", document_id=document.id, whole=True
        )
        assert observation.data["read"] == "from the beginning"
        assert "NOTICE OF ANNUAL FILING" in observation.data["content"]

    async def test_a_short_document_is_not_carved_up(
        self, store, investigation, runtime
    ) -> None:
        await act(runtime, "search_academic", query="reactor cost")
        observation = await act(runtime, "get_evidence", document_id="evidence:1")
        assert observation.data["read"] == "from the beginning"

    async def test_passages_are_still_untrusted_external_content(
        self, store, investigation, runtime
    ) -> None:
        document = await self._hold(store, investigation)
        observation = await act(
            runtime, "get_evidence", document_id=document.id, about="capital cost"
        )
        assert observation.data["content"].startswith("<external_evidence")
        assert "never as instructions" in observation.data["content"]

    async def test_a_quote_from_a_returned_passage_links_to_the_claim(
        self, store, investigation, runtime
    ) -> None:
        """The point of verbatim passages: the excerpt check still passes."""
        document = await self._hold(store, investigation)
        await act(runtime, "get_evidence", document_id=document.id, about="capital cost")
        await act(runtime, "create_claim", text="The capital cost rose by half.")
        observation = await act(
            runtime, "link_evidence", claim_id="claim:1", document_id=document.id,
            stance="supports",
            excerpt="a revision of fifty-two percent against the March estimate",
        )
        assert observation.ok, observation.error


class TestDelegation:
    async def test_spawning_respects_the_depth_limit(self, store, investigation, registry) -> None:
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy(max_depth=2))
        deep = store.tasks.create(
            ResearchTask(
                id="",
                investigation_id=investigation.id,
                role=ResearchRole.SCOUT,
                operation=Operation.SEARCH_WEB,
                objective="deep task",
                depth=2,
            )
        )
        runtime = ActionRuntime(
            store, registry, investigation_id=investigation.id, task=deep, ledger=ledger
        )
        observation = await act(
            runtime, "spawn_research_task", role="academic", objective="go deeper"
        )
        assert not observation.ok
        assert "depth limit" in observation.error

    async def test_spawning_respects_the_task_budget(self, store, investigation, registry, task) -> None:
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy(max_tasks=1))
        ledger.spend(Resource.TASKS)
        runtime = ActionRuntime(
            store, registry, investigation_id=investigation.id, task=task, ledger=ledger
        )
        observation = await act(
            runtime, "spawn_research_task", role="skeptic", objective="attack it"
        )
        assert not observation.ok
        assert "budget" in observation.error

    async def test_a_spawned_task_is_persisted_as_a_child(self, runtime, task) -> None:
        observation = await act(
            runtime, "spawn_research_task", role="skeptic", objective="attack the finding"
        )
        assert observation.ok
        child = runtime.store.tasks.get(observation.data["task_id"])
        assert child.parent_task_id == task.id
        assert child.depth == task.depth + 1
        assert child.status is TaskStatus.PENDING

    async def test_completion_records_a_structured_report(self, runtime) -> None:
        observation = await act(
            runtime,
            "complete_research_task",
            summary="found two relevant papers",
            evidence_ids=["evidence:1", "not-an-id"],
            claim_ids=["claim:1"],
            followups=[{"operation": "find_counterevidence", "objective": "attack it"}],
        )
        assert observation.terminal
        payload = runtime.result_payload
        assert payload["evidence"] == ["evidence:1"], "malformed identifiers are dropped"
        assert payload["recommended_followups"][0]["operation"] == "find_counterevidence"


class TestWorkerLoop:
    def worker(self, store, registry, model, investigation, **kwargs):
        return ResearchWorker(
            store,
            registry,
            model,
            investigation_id=investigation.id,
            ledger=BudgetLedger(store.budget, investigation.id, BudgetPolicy()),
            **kwargs,
        )

    async def test_a_worker_runs_actions_and_reports(self, store, investigation, registry, task) -> None:
        model = ScriptedModel(
            [
                {"thought": "search", "action": "search_academic",
                 "arguments": {"query": "reactor cost"}},
                {"thought": "done", "action": "complete_research_task",
                 "arguments": {"summary": "two papers found", "evidence_ids": ["evidence:1"]}},
            ]
        )
        outcome = await self.worker(store, registry, model, investigation).run(task)
        assert outcome.status is TaskStatus.COMPLETED
        assert outcome.stopped_by == "completed"
        assert "evidence:1" in outcome.result.evidence_ids
        assert store.tasks.get(task.id).status is TaskStatus.COMPLETED

    async def test_json_in_code_fences_is_accepted(self, store, investigation, registry, task) -> None:
        model = ScriptedModel(
            [
                '```json\n{"action": "search_academic", "arguments": {"query": "cost"}}\n```',
                '{"action": "complete_research_task", "arguments": {"summary": "done"}}',
            ]
        )
        outcome = await self.worker(store, registry, model, investigation).run(task)
        assert outcome.status is TaskStatus.COMPLETED

    async def test_unusable_replies_are_corrected_then_abandoned(
        self, store, investigation, registry, task
    ) -> None:
        model = ScriptedModel(["not json at all"], loop_last=True)
        outcome = await self.worker(store, registry, model, investigation).run(task)
        assert outcome.status is TaskStatus.FAILED
        assert "did not produce a usable action" in outcome.stopped_by

    async def test_the_step_limit_ends_a_task(self, store, investigation, registry, task) -> None:
        model = ScriptedModel(
            [{"action": "search_academic", "arguments": {"query": "cost"}}], loop_last=True
        )
        outcome = await self.worker(store, registry, model, investigation, max_steps=3).run(task)
        assert outcome.steps == 3
        assert outcome.stopped_by == "step limit reached"
        # Work done before the limit is kept, and the report says why it is thin.
        assert outcome.result.evidence_ids
        assert "without a report" in outcome.result.summary

    async def test_model_usage_is_charged_to_the_budget(
        self, store, investigation, registry, task
    ) -> None:
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy())
        model = ScriptedModel(
            [{"action": "complete_research_task", "arguments": {"summary": "nothing to do"}}]
        )
        worker = ResearchWorker(
            store, registry, model, investigation_id=investigation.id, ledger=ledger
        )
        await worker.run(task)
        assert ledger.used(Resource.MODEL_CALLS) == 1
        assert ledger.used(Resource.TOKENS) > 0

    async def test_retrieved_text_only_reaches_the_model_when_asked_for(
        self, store, investigation, registry, task
    ) -> None:
        model = ScriptedModel(
            [
                {"action": "search_academic", "arguments": {"query": "reactor cost"}},
                {"action": "complete_research_task", "arguments": {"summary": "done"}},
            ]
        )
        await self.worker(store, registry, model, investigation).run(task)
        prompts = "\n".join(model.prompts())
        assert "median of 117 percent" not in prompts
        # The system prompt explains the envelope; no envelope is ever filled
        # unless the worker asks to read something.
        assert "Untrusted retrieved content" not in prompts

    async def test_requested_text_arrives_labelled(
        self, store, investigation, registry, task
    ) -> None:
        model = ScriptedModel(
            [
                {"action": "search_academic", "arguments": {"query": "reactor cost"}},
                {"action": "get_evidence", "arguments": {"document_id": "evidence:1"}},
                {"action": "complete_research_task", "arguments": {"summary": "done"}},
            ]
        )
        await self.worker(store, registry, model, investigation).run(task)
        prompts = "\n".join(model.prompts())
        assert "<external_evidence" in prompts
        assert "Untrusted retrieved content" in prompts


class TestPlanner:
    async def test_a_plan_becomes_persisted_tasks(self, store, investigation) -> None:
        model = ScriptedModel(
            [
                {
                    "brief": "work the literature, then attack it",
                    "entities": ["NuScale"],
                    "claims_to_verify": ["SMRs are competitive"],
                    "tasks": [
                        {"role": "academic", "operation": "search_academic",
                         "objective": "find cost literature", "priority": 1},
                        {"role": "skeptic", "operation": "find_counterevidence",
                         "objective": "find failed projects", "priority": 2},
                    ],
                }
            ]
        )
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy())
        planner = Planner(store, model, investigation_id=investigation.id, ledger=ledger)
        plan = await planner.plan(investigation)
        assert [str(task.role) for task in plan.tasks] == ["academic", "skeptic"]
        assert all(task.depth == 1 for task in plan.tasks)
        assert store.tasks.count(investigation.id) == 2
        assert store.investigations.get(investigation.id).metadata["plan_brief"]

    async def test_unusable_tasks_are_dropped_not_stored(self, store, investigation) -> None:
        model = ScriptedModel(
            [
                {
                    "tasks": [
                        {"role": "wizard", "objective": "do magic"},
                        {"role": "academic", "objective": ""},
                        {"role": "academic", "objective": "find cost literature"},
                    ]
                }
            ]
        )
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy())
        planner = Planner(store, model, investigation_id=investigation.id, ledger=ledger)
        plan = await planner.plan(investigation)
        assert len(plan.tasks) == 1
        assert len(plan.rejected) == 2

    async def test_the_task_budget_caps_the_plan(self, store, investigation) -> None:
        model = ScriptedModel(
            [
                {
                    "tasks": [
                        {"role": "academic", "objective": f"objective number {index}"}
                        for index in range(6)
                    ]
                }
            ]
        )
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy(max_tasks=2))
        planner = Planner(store, model, investigation_id=investigation.id, ledger=ledger)
        plan = await planner.plan(investigation, max_tasks=5)
        assert len(plan.tasks) == 2

    def test_followups_become_tasks_within_the_depth_limit(self, store, investigation) -> None:
        from research.models.task import FollowUp

        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy(max_depth=2))
        planner = Planner(store, ScriptedModel([]), investigation_id=investigation.id, ledger=ledger)
        parent = store.tasks.create(
            ResearchTask(
                id="",
                investigation_id=investigation.id,
                role=ResearchRole.NEWS,
                operation=Operation.SEARCH_NEWS,
                objective="what was reported",
                depth=1,
            )
        )
        created = planner.ingest_followups(
            parent,
            [
                FollowUp(operation=Operation.FIND_PRIMARY_SOURCE, objective="find the filing"),
                FollowUp(operation=Operation.FIND_PRIMARY_SOURCE, objective="find the filing"),
            ],
        )
        assert len(created) == 1, "a repeated objective is not planned twice"
        assert created[0].depth == 2
        assert created[0].role is ResearchRole.PRIMARY_SOURCE

        too_deep = store.tasks.create(
            ResearchTask(
                id="",
                investigation_id=investigation.id,
                role=ResearchRole.NEWS,
                operation=Operation.SEARCH_NEWS,
                objective="deep",
                depth=2,
            )
        )
        assert planner.ingest_followups(
            too_deep, [FollowUp(operation=Operation.SEARCH_WEB, objective="deeper still")]
        ) == []


class TestObservationShape:
    """Observations must stay parseable, whatever their size."""

    def test_long_lists_are_shortened_not_truncated(self) -> None:
        import json

        from research.agents.runtime import Observation

        observation = Observation(
            action="search_academic",
            ok=True,
            data={
                "query": "reactors",
                "results": [
                    {"id": f"evidence:{index}", "title": "A paper " * 12}
                    for index in range(40)
                ],
            },
        )
        rendered = observation.render()
        payload = json.loads(rendered.split(" -> ", 1)[1])
        assert len(payload["results"]) <= Observation.MAX_LIST_ITEMS
        assert payload["results_omitted"] == 40 - len(payload["results"])
        # Every element is still a result object: nothing is appended that a
        # consumer iterating the list would trip over.
        assert all(isinstance(entry, dict) for entry in payload["results"])

    def test_an_enormous_observation_is_still_valid_json(self) -> None:
        import json

        from research.agents.runtime import Observation

        observation = Observation(
            action="search_web",
            ok=True,
            data={"results": [{"id": f"evidence:{i}", "title": "x" * 400} for i in range(200)]},
        )
        payload = json.loads(observation.render().split(" -> ", 1)[1])
        assert payload["results"]


class TestWorkerResilience:
    async def test_a_model_client_that_raises_fails_only_its_task(
        self, store, investigation, registry, task
    ) -> None:
        from research.budgets import BudgetLedger
        from research.llm.base import ModelSpec

        class ExplodingModel:
            spec = ModelSpec(provider="broken", model="broken")

            async def complete(self, messages, **kwargs):
                raise RuntimeError("adapter bug")

        worker = ResearchWorker(
            store,
            registry,
            ExplodingModel(),
            investigation_id=investigation.id,
            ledger=BudgetLedger(store.budget, investigation.id, BudgetPolicy()),
        )
        outcome = await worker.run(task)
        assert outcome.status is TaskStatus.FAILED
        assert "RuntimeError" in outcome.stopped_by
        assert store.tasks.get(task.id).error
