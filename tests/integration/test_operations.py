"""Operations: search fan-out, fetching, citation traversal, resilience."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from research.config import AcquisitionPolicy, BudgetPolicy, ResearchConfig
from research.errors import SourceUnavailable
from research.graph.citations import CitationGraph
from research.models.common import SourceFamily
from research.models.query import ResearchQuery, SearchHit
from research.operations.citation_chase import CitationChase
from research.operations.search import SearchOperation
from research.budgets import BudgetLedger, Resource
from research.sources.base import SourceAdapter, SourceCapabilities
from research.sources.fetcher import DirectFetchSource
from research.sources.http import SafeHttpClient
from research.sources.registry import SourceRegistry
from research.sources.static import StaticAcademicSource, StaticWebSource
from research.storage.store import ResearchStore

FIXTURES = Path(__file__).parent.parent / "fixtures"

PAPERS = [
    {
        "id": "P1",
        "title": "Levelized cost projections for small modular reactors",
        "abstract": "We review 42 published cost estimates for reactors.",
        "doi": "10.1016/j.enpol.2024.114001",
        "authors": ["R. Okafor"],
        "published_at": "2024-04-12",
        "venue": "Energy Policy",
        "references": ["P2", "P3"],
    },
    {
        "id": "P2",
        "title": "Historical cost escalation in nuclear construction",
        "abstract": "Realized costs exceeded estimates by a median of 117 percent.",
        "doi": "10.1016/j.joule.2021.06.004",
        "authors": ["S. Petrova"],
        "published_at": "2021-07-01",
        "venue": "Joule",
        "references": ["P4"],
    },
    {
        "id": "P3",
        "title": "Learning rates and modular construction",
        "abstract": "Factory fabrication produced learning rates of 8 to 12 percent; transferring them to reactor manufacturing needs far larger order books to affect cost.",
        "doi": "10.1016/j.apenergy.2022.119455",
        "authors": ["K. Tanaka"],
        "published_at": "2022-09-15",
        "venue": "Applied Energy",
        "references": ["P4"],
    },
    {
        "id": "P4",
        "title": "Negative learning in the French nuclear programme",
        "abstract": "Costs rose rather than fell with cumulative experience.",
        "doi": "10.1016/j.enpol.2010.03.029",
        "authors": ["A. Dubois"],
        "published_at": "2010-05-01",
        "venue": "Energy Policy",
        "references": [],
    },
]

PAGE = """<html lang="en"><head><title>{title} | {site}</title>
<link rel="canonical" href="{url}"/><meta property="og:site_name" content="{site}"/>
<meta property="article:published_time" content="2025-02-19"/></head>
<body><article><h1>{title}</h1><p>{body}</p></article></body></html>"""

WEB = [
    {
        "id": "W1",
        "url": "https://www.example-news.test/reactor-agreement",
        "title": "Reactor developer signs data centre agreement",
        "snippet": "The developer signed an agreement to supply power to a campus.",
        "published_at": "2025-02-19",
        "family": "news",
    },
    {
        "id": "W2",
        "url": "https://www.nrc.gov/reactors/design-approval.html",
        "title": "NRC issues standard design approval",
        "snippet": "The Commission issued a standard design approval for the design.",
        "published_at": "2025-05-29",
        "family": "web",
    },
]


def page_transport() -> httpx.MockTransport:
    pages = {
        entry["url"]: PAGE.format(
            title=entry["title"],
            site=entry["url"].split("//")[1].split("/")[0],
            url=entry["url"],
            body=entry["snippet"] + " Further detail follows in the body of the document.",
        )
        for entry in WEB
    }

    def handler(request: httpx.Request) -> httpx.Response:
        html = pages.get(str(request.url))
        if html is None:
            return httpx.Response(404, text="missing")
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    return httpx.MockTransport(handler)


class FailingSource(SourceAdapter):
    """A provider that is down."""

    name = "broken"
    family = SourceFamily.ACADEMIC

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name, families=(SourceFamily.ACADEMIC,), returns_inline_documents=True
        )

    async def search(self, query: ResearchQuery):
        raise SourceUnavailable("provider is down", provider=self.name)

    async def fetch(self, hit: SearchHit):  # pragma: no cover - never reached
        raise SourceUnavailable("provider is down", provider=self.name)


@pytest.fixture
def environment():
    config = ResearchConfig(
        acquisition=AcquisitionPolicy(per_host_min_interval_seconds=0.0, retry_backoff_seconds=0.0)
    )
    client = SafeHttpClient(config.acquisition, transport=page_transport())
    registry = SourceRegistry()
    registry.register(StaticAcademicSource(PAPERS, name="papers"))
    registry.register(StaticWebSource(WEB, name="pages"))
    fetcher = DirectFetchSource(client)
    registry.register(fetcher)
    registry.fetcher = fetcher
    with ResearchStore.in_memory() as store:
        investigation = store.investigations.create("are SMRs competitive?")
        yield store, registry, investigation, config


def operation(environment, *, policy: BudgetPolicy | None = None, task_id: str | None = None):
    store, registry, investigation, config = environment
    ledger = BudgetLedger(store.budget, investigation.id, policy or config.budget)
    return (
        SearchOperation(
            store,
            registry,
            investigation_id=investigation.id,
            ledger=ledger,
            task_id=task_id,
        ),
        ledger,
    )


class TestSearchOperation:
    async def test_academic_search_persists_records_without_fetching(self, environment) -> None:
        store, _, investigation, _ = environment
        search, _ = operation(environment)
        outcome = await search.search("reactor cost", family=SourceFamily.ACADEMIC, limit=10)
        assert len(outcome.new_document_ids) == 4
        assert store.documents.count(investigation.id) == 4
        assert store.fetches.list(investigation.id) == [], "academic records need no page fetch"

    async def test_web_search_fetches_bodies_and_records_the_call(self, environment) -> None:
        store, _, investigation, _ = environment
        search, _ = operation(environment)
        outcome = await search.search("design approval", family=SourceFamily.WEB, limit=5)
        document = store.documents.get(outcome.document_ids[0])
        assert "Further detail follows" in (document.text or "")
        fetches = store.fetches.list(investigation.id)
        assert fetches and all(fetch["ok"] for fetch in fetches)
        assert fetches[0]["document_id"] == document.id

    async def test_every_search_is_logged_with_its_objective(self, environment) -> None:
        store, _, investigation, _ = environment
        search, _ = operation(environment)
        await search.search(
            "reactor cost",
            family=SourceFamily.ACADEMIC,
            objective="establish what the literature says",
        )
        logged = store.queries.list(investigation.id)[0]
        assert logged["objective"] == "establish what the literature says"
        assert logged["status"] == "ok"
        assert logged["result_count"] == 4

    async def test_documents_link_back_to_the_query_that_found_them(self, environment) -> None:
        store, _, investigation, _ = environment
        search, _ = operation(environment)
        outcome = await search.search("reactor cost", family=SourceFamily.ACADEMIC)
        document = store.documents.get(outcome.document_ids[0])
        assert document.provenance.query_id is not None
        logged = store.queries.list(investigation.id)[0]
        assert document.provenance.query_id == logged["id"]

    async def test_one_failing_provider_does_not_stop_the_search(self, environment) -> None:
        store, registry, investigation, _ = environment
        registry.register(FailingSource())
        search, _ = operation(environment)
        outcome = await search.search("reactor cost", family=SourceFamily.ACADEMIC)
        assert "broken" in outcome.provider_errors
        assert outcome.new_document_ids, "the working provider still contributed evidence"
        statuses = {row["provider"]: row["status"] for row in store.queries.list(investigation.id)}
        assert statuses["broken"] == "failed"
        assert statuses["papers"] == "ok"

    async def test_total_provider_outage_is_reported_not_hidden(self, environment) -> None:
        store, registry, investigation, _ = environment
        registry.sources.pop("papers")
        registry.register(FailingSource())
        search, ledger = operation(environment)
        outcome = await search.search("reactor cost", family=SourceFamily.ACADEMIC)
        assert outcome.results == []
        assert outcome.provider_errors
        assert ledger.used(Resource.FAILED_SOURCE_CALLS) == 1

    async def test_unfetchable_results_are_not_stored_as_evidence(self, environment) -> None:
        store, registry, investigation, _ = environment
        registry.sources["pages"] = StaticWebSource(
            [{**WEB[0], "url": "https://www.example-news.test/missing"}], name="pages"
        )
        search, _ = operation(environment)
        outcome = await search.search("reactor", family=SourceFamily.NEWS)
        assert outcome.results == []
        assert len(outcome.unretrieved) == 1
        failures = store.fetches.list(investigation.id)
        assert failures[0]["ok"] == 0
        assert store.documents.count(investigation.id) == 0

    async def test_search_budget_stops_further_searches(self, environment) -> None:
        search, ledger = operation(environment, policy=BudgetPolicy(max_searches=1))
        first = await search.search("reactor cost", family=SourceFamily.ACADEMIC)
        second = await search.search("something else", family=SourceFamily.ACADEMIC)
        assert first.results
        assert second.results == []
        assert second.stopped_by == "search budget exhausted"

    async def test_document_budget_refuses_further_evidence(self, environment) -> None:
        store, _, investigation, _ = environment
        search, _ = operation(environment, policy=BudgetPolicy(max_documents=2, max_academic_documents=2))
        outcome = await search.search("reactor cost", family=SourceFamily.ACADEMIC, limit=10)
        assert store.documents.count(investigation.id) == 2
        assert outcome.stopped_by and "budget" in outcome.stopped_by

    async def test_direct_fetch_persists_one_url(self, environment) -> None:
        store, _, investigation, _ = environment
        search, _ = operation(environment)
        result = await search.fetch_source(
            "https://www.nrc.gov/reactors/design-approval.html", family=SourceFamily.WEB
        )
        assert result is not None and result.created
        assert result.document.source_type.value == "regulatory_document"

    async def test_direct_fetch_failure_is_recorded_and_returns_nothing(self, environment) -> None:
        store, _, investigation, _ = environment
        search, _ = operation(environment)
        assert await search.fetch_source("https://www.example-news.test/nope") is None
        assert store.fetches.failure_count(investigation.id) == 1


class TestCitationChase:
    async def test_backward_traversal_adds_references_and_edges(self, environment) -> None:
        store, registry, investigation, config = environment
        search, ledger = operation(environment)
        outcome = await search.search(
            "levelized cost projections", family=SourceFamily.ACADEMIC, limit=1
        )
        root = store.documents.get(outcome.document_ids[0])

        chase = CitationChase(store, registry, investigation_id=investigation.id, ledger=ledger)
        result = await chase.chase(root.id, direction="backward", max_depth=1, max_documents=10)
        assert len(result.added_document_ids) == 2
        assert result.edges_recorded == 2
        graph = CitationGraph(store, investigation.id)
        assert len(graph.references(root.id)) == 2

    async def test_depth_limit_stops_expansion(self, environment) -> None:
        store, registry, investigation, _ = environment
        search, ledger = operation(environment)
        outcome = await search.search(
            "levelized cost projections", family=SourceFamily.ACADEMIC, limit=1
        )
        root = outcome.document_ids[0]
        chase = CitationChase(store, registry, investigation_id=investigation.id, ledger=ledger)

        shallow = await chase.chase(root, direction="backward", max_depth=1, max_documents=50)
        assert shallow.depth_reached == 1
        assert len(shallow.added_document_ids) == 2  # P2 and P3 only, not P4

    async def test_deeper_traversal_reaches_foundational_work(self, environment) -> None:
        store, registry, investigation, _ = environment
        search, ledger = operation(environment)
        root = (
            await search.search("levelized cost projections", family=SourceFamily.ACADEMIC, limit=1)
        ).document_ids[0]
        chase = CitationChase(store, registry, investigation_id=investigation.id, ledger=ledger)
        result = await chase.chase(root, direction="backward", max_depth=2, max_documents=50)
        titles = [store.documents.get(doc_id).title for doc_id in result.added_document_ids]
        assert any("French nuclear" in (title or "") for title in titles)
        assert result.depth_reached == 2

    async def test_document_limit_terminates_traversal(self, environment) -> None:
        store, registry, investigation, _ = environment
        search, ledger = operation(environment)
        root = (
            await search.search("levelized cost projections", family=SourceFamily.ACADEMIC, limit=1)
        ).document_ids[0]
        chase = CitationChase(store, registry, investigation_id=investigation.id, ledger=ledger)
        result = await chase.chase(root, direction="backward", max_depth=3, max_documents=1)
        assert len(result.added_document_ids) <= 1
        assert "document limit" in result.stopped_by

    async def test_citation_depth_budget_overrides_a_larger_request(self, environment) -> None:
        store, registry, investigation, _ = environment
        search, ledger = operation(environment, policy=BudgetPolicy(max_citation_depth=1))
        root = (
            await search.search("levelized cost projections", family=SourceFamily.ACADEMIC, limit=1)
        ).document_ids[0]
        chase = CitationChase(store, registry, investigation_id=investigation.id, ledger=ledger)
        result = await chase.chase(root, direction="backward", max_depth=5, max_documents=50)
        assert result.max_depth == 1
        assert result.depth_reached <= 1

    async def test_traversal_terminates_on_a_cyclic_graph(self, environment) -> None:
        store, _, investigation, config = environment
        cyclic = [
            {"id": "C1", "title": "First paper on cycles", "abstract": "a", "doi": "10.1000/c1", "references": ["C2"]},
            {"id": "C2", "title": "Second paper on cycles", "abstract": "b", "doi": "10.1000/c2", "references": ["C1"]},
        ]
        registry = SourceRegistry()
        registry.register(StaticAcademicSource(cyclic, name="papers"))
        ledger = BudgetLedger(store.budget, investigation.id, BudgetPolicy())
        search = SearchOperation(store, registry, investigation_id=investigation.id, ledger=ledger)
        root = (await search.search("cycles", family=SourceFamily.ACADEMIC, limit=1)).document_ids[0]
        chase = CitationChase(store, registry, investigation_id=investigation.id, ledger=ledger)
        result = await chase.chase(root, direction="both", max_depth=4, max_documents=20)
        assert result.stopped_by == "traversal complete"
        assert store.documents.count(investigation.id) <= 2

    async def test_already_held_papers_are_not_counted_twice(self, environment) -> None:
        store, registry, investigation, _ = environment
        search, ledger = operation(environment)
        # Search brings in every paper first; traversal should find them held.
        first = await search.search("cost", family=SourceFamily.ACADEMIC, limit=10)
        assert len(first.new_document_ids) == 4
        root = [
            document
            for document in store.documents.list(investigation.id)
            if document.external_id == "P1"
        ][0]
        chase = CitationChase(store, registry, investigation_id=investigation.id, ledger=ledger)
        result = await chase.chase(root.id, direction="backward", max_depth=1, max_documents=10)
        assert result.added_document_ids == []
        assert result.already_held == 2
        assert store.documents.count(investigation.id) == 4
