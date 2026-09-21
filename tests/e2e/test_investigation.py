"""End-to-end runs over a fixed corpus.

These assert the properties the whole system exists to provide: a mixed
evidence corpus, traceable provenance, honest counting of independent
sources, bounded recursion, and an investigation that survives a restart.
"""

from __future__ import annotations

import pytest

from research.graph.independence import independent_documents
from research.config import AcquisitionPolicy, BudgetPolicy, ResearchConfig
from research.graph.citations import CitationGraph
from research.models.common import DuplicateRelation, SourceFamily, SourceType
from research.models.investigation import InvestigationStatus, StopReason
from research.operations.citation_chase import CitationChase
from research.operations.search import SearchOperation
from research.orchestration.budgets import BudgetLedger, Resource
from research.sources.offline import build_offline_registry, load_corpus
from research.storage.store import ResearchStore

PLAN = [
    ("small modular reactor levelized cost estimates", SourceFamily.ACADEMIC),
    ("nuclear construction cost overruns learning rate", SourceFamily.ACADEMIC),
    ("NuScale small modular reactor project data center power", SourceFamily.NEWS),
    ("NRC design approval data center power agreement filing", SourceFamily.WEB),
]


@pytest.fixture
def config() -> ResearchConfig:
    return ResearchConfig(
        acquisition=AcquisitionPolicy(per_host_min_interval_seconds=0.0, retry_backoff_seconds=0.0)
    )


async def run_investigation(
    store: ResearchStore, config: ResearchConfig, *, policy: BudgetPolicy | None = None
):
    corpus = load_corpus()
    registry, _ = build_offline_registry(corpus, config)
    investigation = store.investigations.create(
        corpus["question"], budget=(policy or config.budget).to_dict()
    )
    ledger = BudgetLedger(store.budget, investigation.id, policy or config.budget)
    search = SearchOperation(store, registry, investigation_id=investigation.id, ledger=ledger)
    for text, family in PLAN:
        await search.search(text, family=family, limit=10, objective=f"cover {family}")
    return investigation, registry, ledger


@pytest.fixture
async def investigated(config):
    with ResearchStore.in_memory() as store:
        investigation, registry, ledger = await run_investigation(store, config)
        yield store, investigation, registry, ledger


class TestMixedCorpus:
    async def test_corpus_spans_academic_news_and_primary_material(self, investigated) -> None:
        store, investigation, _, _ = investigated
        types = {
            document.source_type
            for document in store.documents.list(investigation.id, limit=100)
        }
        assert SourceType.ACADEMIC_PEER_REVIEWED in types
        assert SourceType.ACADEMIC_PREPRINT in types
        assert SourceType.REGULATORY_DOCUMENT in types
        assert SourceType.CORPORATE_FILING in types or SourceType.PRESS_RELEASE in types
        assert {SourceType.ORIGINAL_NEWS_REPORTING, SourceType.SECONDARY_NEWS_REPORTING} & types

    async def test_academic_and_news_material_are_both_held(self, investigated) -> None:
        store, investigation, _, _ = investigated
        assert store.documents.count(investigation.id, family=SourceFamily.ACADEMIC) >= 5
        assert store.documents.count(investigation.id, family=SourceFamily.NEWS) >= 3

    async def test_every_document_carries_its_provenance(self, investigated) -> None:
        store, investigation, _, _ = investigated
        for document in store.documents.list(investigation.id, limit=100):
            assert document.provenance is not None
            assert document.provenance.provider
            assert document.content_hash.startswith("sha256:")
            # Every document is reachable back to the call that produced it.
            assert document.provenance.query_id or document.provenance.fetch_id

    async def test_body_text_comes_from_the_publisher_not_the_search_engine(self, investigated) -> None:
        store, investigation, _, _ = investigated
        news = store.documents.list(investigation.id, family=SourceFamily.NEWS)
        assert news
        for document in news:
            assert document.text, "news evidence must have a retrieved body"
            assert document.metadata.get("http_status") == 200


class TestSourceIndependence:
    async def test_syndicated_copies_are_not_independent_corroboration(self, investigated) -> None:
        store, investigation, _, _ = investigated
        documents = store.documents.list(investigation.id, limit=100)
        groups = independent_documents(store.documents, [d.id for d in documents])
        assert len(groups) < len(documents), "some copies must collapse"

        syndicated = [
            document
            for document in documents
            if document.duplicate_relation is DuplicateRelation.SYNDICATED_COPY
        ]
        assert syndicated, "the corpus contains a republished wire story"
        for copy in syndicated:
            original = store.documents.get(copy.duplicate_of)
            assert copy.independence_key == original.independence_key
            assert copy.canonical_host != original.canonical_host

    async def test_a_rewrite_that_credits_its_source_is_marked_derived(self, investigated) -> None:
        store, investigation, _, _ = investigated
        derived = [
            document
            for document in store.documents.list(investigation.id, limit=100)
            if document.duplicate_relation is DuplicateRelation.DERIVED_ARTICLE
        ]
        assert derived
        for document in derived:
            assert document.derived_from
            parent = store.documents.get(document.derived_from)
            assert document.independence_key == parent.independence_key

    async def test_counting_copies_as_sources_would_overstate_agreement(self, investigated) -> None:
        store, investigation, _, _ = investigated
        news = store.documents.list(investigation.id, family=SourceFamily.NEWS)
        groups = independent_documents(store.documents, [d.id for d in news])
        naive = len(news)
        honest = len(groups)
        assert honest < naive


class TestCitationTraversal:
    async def test_traversal_builds_a_resolved_citation_graph(self, investigated) -> None:
        store, investigation, registry, ledger = investigated
        seed = max(
            store.documents.list(investigation.id, family=SourceFamily.ACADEMIC),
            key=lambda document: document.metadata.get("cited_by_count", 0),
        )
        chase = CitationChase(store, registry, investigation_id=investigation.id, ledger=ledger)
        result = await chase.chase(seed.id, direction="both", max_depth=2, max_documents=20)
        graph = CitationGraph(store, investigation.id)
        stats = graph.stats()
        assert result.stopped_by == "traversal complete"
        assert stats["edges"] > 0
        assert stats["resolved_edges"] == stats["edges"]
        assert graph.in_degree(limit=3)

    async def test_traversal_respects_the_citation_budget(self, config) -> None:
        with ResearchStore.in_memory() as store:
            policy = BudgetPolicy(max_citation_documents=2, max_citation_depth=2)
            investigation, registry, ledger = await run_investigation(store, config, policy=policy)
            seed = store.documents.list(investigation.id, family=SourceFamily.ACADEMIC)[0]
            chase = CitationChase(
                store, registry, investigation_id=investigation.id, ledger=ledger
            )
            await chase.chase(seed.id, direction="backward", max_depth=2, max_documents=50)
            assert ledger.used(Resource.CITATION_DOCUMENTS) <= 2


class TestBudgets:
    async def test_budget_terminates_an_investigation_before_the_plan_does(self, config) -> None:
        with ResearchStore.in_memory() as store:
            policy = BudgetPolicy(max_searches=2, max_documents=100)
            investigation, _, ledger = await run_investigation(store, config, policy=policy)
            assert ledger.used(Resource.SEARCHES) == 2
            assert Resource.SEARCHES in ledger.exhausted_resources()

    async def test_stopping_reason_is_recorded_for_later_inspection(self, investigated) -> None:
        store, investigation, _, ledger = investigated
        store.investigations.set_status(
            investigation.id,
            InvestigationStatus.COMPLETED,
            stop_reason=StopReason.EVIDENCE_SUFFICIENT,
            stop_detail="major claims have primary-source backing",
        )
        reloaded = store.investigations.get(investigation.id)
        assert reloaded.stop_reason is StopReason.EVIDENCE_SUFFICIENT
        assert reloaded.stop_detail


class TestResumability:
    async def test_an_investigation_survives_a_restart(self, tmp_path, config) -> None:
        path = tmp_path / "research.sqlite3"
        with ResearchStore.open(path) as store:
            investigation, _, ledger = await run_investigation(store, config)
            investigation_id = investigation.id
            documents_before = store.documents.count(investigation_id)
            searches_before = ledger.used(Resource.SEARCHES)

        # A new process, with no conversation history, picks the work back up.
        with ResearchStore.open(path) as store:
            reloaded = store.investigations.get(investigation_id)
            assert reloaded.question
            assert store.documents.count(investigation_id) == documents_before
            ledger = BudgetLedger(
                store.budget, investigation_id, BudgetPolicy.from_dict(reloaded.budget)
            )
            assert ledger.used(Resource.SEARCHES) == searches_before

            corpus = load_corpus()
            registry, _ = build_offline_registry(corpus, config)
            search = SearchOperation(
                store, registry, investigation_id=investigation_id, ledger=ledger
            )
            outcome = await search.search(
                "NuScale small modular reactor project data center power",
                family=SourceFamily.NEWS,
                limit=10,
            )
            # Everything this query returns is already held, which is what a
            # resumed run should discover rather than re-import.
            assert outcome.new_document_ids == []
            assert store.documents.count(investigation_id) == documents_before

    async def test_activity_log_explains_what_happened(self, tmp_path, config) -> None:
        path = tmp_path / "research.sqlite3"
        with ResearchStore.open(path) as store:
            investigation, _, _ = await run_investigation(store, config)
            investigation_id = investigation.id
        with ResearchStore.open(path) as store:
            queries = store.queries.list(investigation_id)
            fetches = store.fetches.list(investigation_id)
            assert len(queries) == len(PLAN)
            assert all(query["objective"] for query in queries)
            assert fetches and all(fetch["document_id"] for fetch in fetches if fetch["ok"])
