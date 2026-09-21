"""Retrieval over held evidence: lexical, graph, fusion and vectors.

The question these tests answer is whether an investigation can find what it
already has. That matters twice over: a worker that cannot search its own
store re-queries a provider for material sitting in SQLite, and a reader
cannot get at a document without already knowing its identifier.
"""

from __future__ import annotations

import pytest

from research.models.claim import ClaimEvidenceLink, EvidenceStance
from research.models.common import DuplicateRelation, Provenance, SourceFamily, SourceType, utcnow
from research.models.evidence import EvidenceDocument
from research.config import RetrievalSettings
from research.retrieval.embeddings import (
    EmbeddingIndex,
    HashingEmbedder,
    LocalEmbedder,
    cosine,
    pack,
    unpack,
)
from research.retrieval.fusion import reciprocal_rank_fusion
from research.retrieval.graph import GraphExpansion
from research.retrieval.lexical import LexicalIndex, prepare_query
from research.retrieval import search as search_module
from research.retrieval.search import CorpusSearch
from research.storage.store import ResearchStore


def add_document(store: ResearchStore, investigation_id: str, **overrides) -> EvidenceDocument:
    number = store.documents.count(investigation_id) + 1
    defaults = dict(
        id=store.documents.new_id(),
        investigation_id=investigation_id,
        source_type=SourceType.ACADEMIC_PEER_REVIEWED,
        source_family=SourceFamily.ACADEMIC,
        provider="openalex",
        external_id=f"W{number}",
        canonical_url=f"https://example.org/{number}",
        published_at=utcnow(),
        content_hash=f"sha256:{number}",
        provenance=Provenance(provider="openalex"),
    )
    defaults.update(overrides)
    return store.documents.add(EvidenceDocument(**defaults))


@pytest.fixture
def corpus_store(store: ResearchStore, investigation):
    """A small corpus with the structure retrieval is supposed to exploit."""
    costs = add_document(
        store,
        investigation.id,
        title="Construction cost overruns in small modular reactor programmes",
        abstract="Evidence on capital cost escalation across reactor builds.",
        text="Overnight capital cost estimates rose through construction.",
    )
    # No cost vocabulary at all: reachable only through the graph.
    siting = add_document(
        store,
        investigation.id,
        title="Grid interconnection queues and hyperscale siting decisions",
        abstract="Where operators place new load, and how long they wait.",
        text="Queue times shape where a large load can be connected.",
    )
    unrelated = add_document(
        store,
        investigation.id,
        title="Benthic invertebrate communities of the Baltic Sea",
        abstract="A survey of seafloor fauna.",
        text="Sampling was carried out across three summers.",
    )
    store.citations.add(
        investigation.id,
        citing_document_id=costs.id,
        cited_document_id=siting.id,
        provider="openalex",
    )
    return store, investigation, {"costs": costs, "siting": siting, "unrelated": unrelated}


class TestQueryPreparation:
    def test_prose_becomes_a_valid_fts_query(self) -> None:
        assert prepare_query("reactor cost") == '"reactor" OR "cost"'
        assert prepare_query("reactor cost", mode="all") == '"reactor" AND "cost"'

    def test_fts_syntax_in_a_query_cannot_reach_the_engine(self) -> None:
        """A research query is prose, not a query language."""
        prepared = prepare_query('reactor" OR documents_fts MATCH "x')
        assert "MATCH" not in prepared.replace('"MATCH"', "")
        for character in "*():^":
            assert character not in prepared

    def test_an_empty_query_stays_empty(self) -> None:
        assert prepare_query("   ") == ""
        assert prepare_query("a !") == ""


class TestLexicalIndex:
    def test_documents_are_indexed_as_they_are_stored(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        index = LexicalIndex(store.db)
        assert index.count(investigation.id) == 3

        hits = index.search("construction cost overruns", investigation_id=investigation.id)
        assert hits[0].document_id == documents["costs"].id

    def test_the_index_follows_deletions_and_rebuilds(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        index = LexicalIndex(store.db)
        index.remove(documents["costs"].id)
        assert index.count(investigation.id) == 2

        assert index.rebuild(investigation.id) == 3
        hits = index.search("capital cost escalation", investigation_id=investigation.id)
        assert [hit.document_id for hit in hits][:1] == [documents["costs"].id]

    def test_title_outranks_a_passing_mention_in_the_body(
        self, store: ResearchStore, investigation
    ) -> None:
        titled = add_document(
            store, investigation.id, title="Reactor licensing", text="Unrelated body."
        )
        mentioning = add_document(
            store,
            investigation.id,
            title="Something else entirely",
            text="A reactor is mentioned once here, in passing.",
        )
        hits = LexicalIndex(store.db).search("reactor", investigation_id=investigation.id)
        assert [hit.document_id for hit in hits] == [titled.id, mentioning.id]

    def test_one_investigation_cannot_read_another(
        self, store: ResearchStore, investigation
    ) -> None:
        other = store.investigations.create("a different question")
        add_document(store, other.id, title="Reactor cost overruns elsewhere")
        hits = LexicalIndex(store.db).search("reactor", investigation_id=investigation.id)
        assert hits == []

    def test_results_carry_a_snippet_of_what_matched(self, corpus_store) -> None:
        store, investigation, _ = corpus_store
        hits = LexicalIndex(store.db).search(
            "overnight capital", investigation_id=investigation.id
        )
        assert "capital" in hits[0].snippet.lower()


class TestGraphExpansion:
    def test_citations_reach_documents_the_words_do_not(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        hits = GraphExpansion(store, investigation.id).expand([documents["costs"].id])
        found = {hit.document_id: hit for hit in hits}
        assert documents["siting"].id in found
        assert documents["unrelated"].id not in found

    def test_every_expansion_says_why(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        hits = GraphExpansion(store, investigation.id).expand([documents["costs"].id])
        assert hits[0].reasons == [f"cited by {documents['costs'].id}"]

    def test_shared_claims_connect_evidence(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        claim = store.claims.create(investigation.id, "SMR costs have overrun.")
        for document in (documents["costs"], documents["unrelated"]):
            store.claims.link_evidence(
                ClaimEvidenceLink(
                    claim_id=claim.id,
                    document_id=document.id,
                    stance=EvidenceStance.SUPPORTS,
                )
            )
        hits = GraphExpansion(store, investigation.id).expand([documents["costs"].id])
        reasons = {hit.document_id: hit.reasons for hit in hits}
        assert claim.id in " ".join(reasons[documents["unrelated"].id])

    def test_a_seed_does_not_return_itself(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        hits = GraphExpansion(store, investigation.id).expand([documents["costs"].id])
        assert documents["costs"].id not in {hit.document_id for hit in hits}

    def test_relations_can_be_restricted(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        hits = GraphExpansion(store, investigation.id).expand(
            [documents["costs"].id], relations=("same_entity",)
        )
        assert hits == []

    def test_no_seeds_is_not_an_error(self, corpus_store) -> None:
        store, investigation, _ = corpus_store
        assert GraphExpansion(store, investigation.id).expand([]) == []


class TestFusion:
    def test_agreement_between_retrievers_outranks_one_strong_opinion(self) -> None:
        fused = reciprocal_rank_fusion(
            {"lexical": ["evidence:1", "evidence:2"], "graph": ["evidence:2", "evidence:3"]}
        )
        assert fused[0].document_id == "evidence:2"
        assert fused[0].retrievers == ["graph", "lexical"]

    def test_weights_shift_the_ranking(self) -> None:
        rankings = {"lexical": ["evidence:1"], "vector": ["evidence:2"]}
        assert reciprocal_rank_fusion(rankings, weights={"lexical": 2.0})[0].document_id == (
            "evidence:1"
        )
        assert reciprocal_rank_fusion(rankings, weights={"vector": 2.0})[0].document_id == (
            "evidence:2"
        )

    def test_ranks_are_recorded_per_retriever(self) -> None:
        fused = reciprocal_rank_fusion(
            {"lexical": ["evidence:9", "evidence:1"], "graph": ["evidence:1"]}
        )
        by_id = {hit.document_id: hit for hit in fused}
        assert by_id["evidence:1"].ranks == {"lexical": 2, "graph": 1}

    def test_reasons_attach_to_the_documents_they_explain(self) -> None:
        fused = reciprocal_rank_fusion(
            {"graph": ["evidence:1"]},
            reasons={"evidence:1": ["cited by evidence:4"], "evidence:99": ["not in the ranking"]},
        )
        assert fused[0].reasons == ["cited by evidence:4"]
        assert len(fused) == 1

    def test_ties_break_deterministically(self) -> None:
        first = reciprocal_rank_fusion({"a": ["evidence:2"], "b": ["evidence:1"]})
        second = reciprocal_rank_fusion({"b": ["evidence:1"], "a": ["evidence:2"]})
        assert [hit.document_id for hit in first] == [hit.document_id for hit in second]


class TestVectors:
    def test_packing_a_vector_round_trips(self) -> None:
        vector = [0.5, -0.25, 0.125]
        assert unpack(pack(vector)) == pytest.approx(vector)

    def test_cosine_handles_degenerate_input(self) -> None:
        assert cosine([], [1.0]) == 0.0
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_indexing_is_idempotent(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        index = EmbeddingIndex(store.db, HashingEmbedder())
        held = list(documents.values())
        assert await index.index(held) == 3
        assert await index.index(held) == 0
        assert index.count(investigation.id) == 3

    @pytest.mark.asyncio
    async def test_search_is_restricted_to_the_candidates_offered(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        index = EmbeddingIndex(store.db, HashingEmbedder())
        await index.index(list(documents.values()))
        hits = await index.search(
            "reactor construction cost",
            investigation_id=investigation.id,
            candidates=[documents["unrelated"].id],
        )
        assert [hit.document_id for hit in hits] in ([], [documents["unrelated"].id])
        wide = await index.search("reactor construction cost", investigation_id=investigation.id)
        assert wide[0].document_id == documents["costs"].id


class TestCorpusSearch:
    @pytest.mark.asyncio
    async def test_it_finds_held_evidence_by_its_words(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        hits = await CorpusSearch(store, investigation.id).search("cost overruns")
        assert hits[0].document.id == documents["costs"].id
        assert "lexical" in hits[0].ranks

    @pytest.mark.asyncio
    async def test_graph_expansion_surfaces_what_the_words_miss(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        search = CorpusSearch(store, investigation.id)

        with_graph = await search.search("cost overruns")
        assert documents["siting"].id in {hit.document.id for hit in with_graph}

        text_only = await search.search("cost overruns", expand=False)
        assert documents["siting"].id not in {hit.document.id for hit in text_only}

    @pytest.mark.asyncio
    async def test_a_result_explains_itself(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        hits = await CorpusSearch(store, investigation.id).search("cost overruns")
        siting = next(hit for hit in hits if hit.document.id == documents["siting"].id)
        assert siting.reasons == [f"cited by {documents['costs'].id}"]

    @pytest.mark.asyncio
    async def test_copies_of_one_story_take_one_place(
        self, store: ResearchStore, investigation
    ) -> None:
        original = add_document(
            store,
            investigation.id,
            source_family=SourceFamily.NEWS,
            source_type=SourceType.ORIGINAL_NEWS_REPORTING,
            title="Utility signs reactor deal",
            text="The utility signed an agreement for reactor capacity.",
            independence_key="wire:reuters:reactor-deal",
        )
        copy = add_document(
            store,
            investigation.id,
            source_family=SourceFamily.NEWS,
            source_type=SourceType.ORIGINAL_NEWS_REPORTING,
            title="Utility signs reactor deal",
            text="The utility signed an agreement for reactor capacity.",
            independence_key="wire:reuters:reactor-deal",
            duplicate_of=original.id,
            duplicate_relation=DuplicateRelation.SYNDICATED_COPY,
        )
        search = CorpusSearch(store, investigation.id)

        collapsed = await search.search("reactor deal")
        assert [hit.document.id for hit in collapsed] == [original.id]
        assert collapsed[0].copies == [copy.id]
        assert collapsed[0].independent

        everything = await search.search("reactor deal", collapse_copies=False)
        assert {hit.document.id for hit in everything} == {original.id, copy.id}

    @pytest.mark.asyncio
    async def test_the_original_represents_the_group_even_when_the_copy_ranks_higher(
        self, store: ResearchStore, investigation
    ) -> None:
        # The copy is stored first and carries the query's words in its title,
        # so lexical ranking prefers it. The group should still be reported
        # under the document that did the reporting.
        copy = add_document(
            store,
            investigation.id,
            title="Reactor deal reactor deal",
            text="Syndicated text.",
            independence_key="wire:ap:deal",
        )
        original = add_document(
            store,
            investigation.id,
            title="Reactor deal",
            text="Original reporting.",
            independence_key="wire:ap:deal",
        )
        store.documents.mark_duplicate(
            copy.id,
            duplicate_of=original.id,
            relation=DuplicateRelation.SYNDICATED_COPY,
        )

        hits = await CorpusSearch(store, investigation.id).search("reactor deal")
        assert [hit.document.id for hit in hits] == [original.id]
        assert hits[0].copies == [copy.id]

    @pytest.mark.asyncio
    async def test_nothing_held_is_an_empty_result_not_an_error(self, corpus_store) -> None:
        store, investigation, _ = corpus_store
        assert await CorpusSearch(store, investigation.id).search("photovoltaic tariffs") == []
        assert await CorpusSearch(store, investigation.id).search("   ") == []

    @pytest.mark.asyncio
    async def test_vectors_join_the_ranking_when_they_are_enabled(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        index = EmbeddingIndex(store.db, HashingEmbedder())
        await index.index(list(documents.values()))

        search = CorpusSearch(store, investigation.id, embeddings=index)
        hits = await search.search("cost overruns")
        assert any("vector" in hit.ranks for hit in hits)
        assert search.stats() == {"documents": 3, "indexed": 3, "embedded": 3}

    @pytest.mark.asyncio
    async def test_a_result_serialises_to_what_a_reader_needs(self, corpus_store) -> None:
        store, investigation, documents = corpus_store
        hits = await CorpusSearch(store, investigation.id).search("cost overruns")
        payload = hits[0].to_dict()
        assert payload["id"] == documents["costs"].id
        assert payload["found_by"] == sorted(hits[0].ranks)
        assert payload["independent"] is True
        assert isinstance(payload["score"], float)


class TestIndexMigration:
    def test_a_store_written_before_the_index_existed_is_backfilled(self, tmp_path) -> None:
        """An empty index is worse than none: it answers every search with nothing."""
        import sqlite3

        path = tmp_path / "old.sqlite3"
        with ResearchStore.open(path) as store:
            investigation = store.investigations.create("q")
            add_document(
                store,
                investigation.id,
                title="Construction cost overruns",
                text="Capital costs exceeded the estimates.",
            )
            investigation_id = investigation.id

        # What a database written before schema v2 looks like.
        connection = sqlite3.connect(path)
        connection.execute("DROP TABLE documents_fts")
        connection.execute("UPDATE schema_meta SET value = '1' WHERE key = 'schema_version'")
        connection.commit()
        connection.close()

        with ResearchStore.open(path) as store:
            index = LexicalIndex(store.db)
            assert index.count(investigation_id) == 1
            hits = index.search("cost overruns", investigation_id=investigation_id)
            assert len(hits) == 1


class TestVectorsByDefault:
    """Vectors are part of the ranking, not a switch - and never a blocker."""

    def test_the_default_embedder_needs_no_credential(self) -> None:
        settings = RetrievalSettings()
        assert settings.embeddings_enabled is True
        assert settings.embedding_provider == "local"

    @pytest.mark.asyncio
    async def test_the_shortlist_embeds_itself(self, corpus_store) -> None:
        """No separate indexing step: a search embeds what it is about to rank."""
        store, investigation, _ = corpus_store
        index = EmbeddingIndex(store.db, HashingEmbedder())
        assert index.count(investigation.id) == 0

        search = CorpusSearch(store, investigation.id, embeddings=index)
        hits = await search.search("cost overruns")

        assert any("vector" in hit.ranks for hit in hits)
        assert index.count(investigation.id) == len(
            {document_id for hit in hits for document_id in [hit.document.id]}
        ) or index.count(investigation.id) > 0
        # Second search costs nothing: the vectors are in SQLite now.
        before = index.count(investigation.id)
        await search.search("cost overruns")
        assert index.count(investigation.id) == before

    @pytest.mark.asyncio
    async def test_vectors_reach_a_document_that_shares_no_words(self, corpus_store) -> None:
        """The one thing lexical retrieval cannot do, and the reason for vectors."""
        store, investigation, documents = corpus_store
        # An embedder that places one held document near this query and
        # everything else far from it. What is under test is whether a
        # vector hit can enter the ranking on its own, not model quality.
        class Pointed:
            model = "pointed"
            dimensions = 2

            async def embed(self, texts):
                near = ("substation", "interconnection")
                return [
                    [1.0, 0.0] if any(word in text.lower() for word in near) else [0.0, 1.0]
                    for text in texts
                ]

        # Not one of these words appears anywhere in the corpus, so lexical
        # retrieval returns nothing and graph expansion has no seeds.
        query = "substation headroom bottlenecks"
        assert LexicalIndex(store.db).search(query, investigation_id=investigation.id) == []

        search = CorpusSearch(
            store, investigation.id, embeddings=EmbeddingIndex(store.db, Pointed())
        )
        hits = await search.search(query)

        assert [hit.document.id for hit in hits] == [documents["siting"].id]
        assert hits[0].ranks == {"vector": 1}

    @pytest.mark.asyncio
    async def test_embedding_catches_up_across_searches(self, corpus_store, monkeypatch) -> None:
        """A large corpus must not stall the first search that touches it."""
        store, investigation, _ = corpus_store
        for number in range(9):
            add_document(
                store,
                investigation.id,
                title=f"Marine sediment survey {number}",
                text="Sampling was carried out across three summers.",
            )
        monkeypatch.setattr(search_module, "EMBED_PER_SEARCH", 4)
        index = EmbeddingIndex(store.db, HashingEmbedder())
        search = CorpusSearch(store, investigation.id, embeddings=index)

        await search.search("cost overruns")
        assert index.count(investigation.id) == 4
        assert search.stats()["partly_embedded"] is True

        for _ in range(3):
            await search.search("cost overruns")
        assert index.count(investigation.id) == store.documents.count(investigation.id)

    @pytest.mark.asyncio
    async def test_a_model_that_cannot_load_costs_an_opinion_not_the_search(
        self, corpus_store
    ) -> None:
        store, investigation, documents = corpus_store
        index = EmbeddingIndex(store.db, LocalEmbedder())  # stubbed unloadable

        search = CorpusSearch(store, investigation.id, embeddings=index)
        hits = await search.search("cost overruns")

        assert hits[0].document.id == documents["costs"].id
        assert not any("vector" in hit.ranks for hit in hits)
        assert search.degraded
        assert "degraded" in search.stats()

    @pytest.mark.asyncio
    async def test_an_embedder_that_raises_does_not_lose_the_search(
        self, corpus_store
    ) -> None:
        class Broken:
            model = "broken"
            dimensions = 8

            async def embed(self, texts):
                raise RuntimeError("the model host is down")

        store, investigation, documents = corpus_store
        search = CorpusSearch(
            store, investigation.id, embeddings=EmbeddingIndex(store.db, Broken())
        )
        hits = await search.search("cost overruns")
        assert hits[0].document.id == documents["costs"].id


class TestLocalEmbedder:
    def test_it_reports_why_it_cannot_run_rather_than_raising(self) -> None:
        embedder = LocalEmbedder()
        assert embedder._load() is None
        assert embedder.unavailable

    @pytest.mark.asyncio
    async def test_it_returns_nothing_rather_than_failing(self) -> None:
        assert await LocalEmbedder().embed(["anything at all"]) == []

    @pytest.mark.asyncio
    async def test_no_texts_is_not_a_model_load(self) -> None:
        embedder = LocalEmbedder()
        assert await embedder.embed([]) == []
        assert embedder.unavailable is None, "nothing was asked for, nothing was loaded"


class TestOfflineMode:
    """--offline promises no network, and that has to include model weights."""

    def _context(self, store, *, offline: bool, model_path: str | None = None):
        from research.cli.context import CliContext
        from research.config import ResearchConfig, RetrievalSettings
        from research.sources.registry import SourceRegistry

        config = ResearchConfig(
            retrieval=RetrievalSettings(embedding_model_path=model_path)
        )
        return CliContext(store, config, SourceRegistry(), None, offline=offline)

    def test_offline_sits_vectors_out_rather_than_fetching_weights(self, store) -> None:
        assert self._context(store, offline=True).embedding_index() is None

    def test_weights_already_on_disk_are_used_offline(self, store) -> None:
        context = self._context(store, offline=True, model_path="/opt/models/minilm")
        assert context.embedding_index() is not None

    def test_a_normal_run_has_vectors(self, store) -> None:
        assert self._context(store, offline=False).embedding_index() is not None
