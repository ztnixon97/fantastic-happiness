"""The harness that measures the rest of the system.

A harness has to be more trustworthy than the thing it measures, so the
metrics are checked against worked examples rather than against themselves,
and the fixtures are checked for the properties the judgements assume.
"""

from __future__ import annotations

import pytest

from research.evaluation.corpus import build_corpus, load_dataset
from research.evaluation.independence import evaluate_independence
from research.evaluation.metrics import (
    Classification,
    RankingScore,
    average,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from research.evaluation.retrieval import ABLATIONS, evaluate_retrieval
from research.evaluation.run import evaluate_run
from research.storage.store import ResearchStore


class TestMetrics:
    def test_recall_counts_what_was_found(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "d"], 3) == 0.5
        assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
        assert recall_at_k(["x"], ["a"], 5) == 0.0

    def test_recall_only_looks_at_the_top_k(self) -> None:
        assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0
        assert recall_at_k(["x", "y", "a"], ["a"], 3) == 1.0

    def test_precision_is_over_what_was_returned(self) -> None:
        assert precision_at_k(["a", "x", "y", "z"], ["a"], 4) == 0.25
        assert precision_at_k([], ["a"], 5) == 0.0

    def test_reciprocal_rank_is_the_first_hit(self) -> None:
        assert reciprocal_rank(["a"], ["a"]) == 1.0
        assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_ndcg_rewards_putting_them_at_the_top(self) -> None:
        """Two rankings with the same recall are not equally good."""
        top = ndcg_at_k(["a", "b", "x", "y"], ["a", "b"], 4)
        bottom = ndcg_at_k(["x", "y", "a", "b"], ["a", "b"], 4)
        assert top == 1.0
        assert bottom < top

    def test_ndcg_of_a_perfect_ranking_is_one(self) -> None:
        assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == pytest.approx(1.0)

    def test_nothing_relevant_scores_zero_rather_than_dividing_by_zero(self) -> None:
        assert recall_at_k(["a"], [], 5) == 0.0
        assert ndcg_at_k(["a"], [], 5) == 0.0

    def test_a_score_records_what_it_missed(self) -> None:
        score = RankingScore.of("q", ["a", "b"], ["a", "z"])
        assert score.missed == ("z",)
        assert score.relevant == 2

    def test_averaging_an_empty_set_is_empty(self) -> None:
        assert average([]) == {}

    def test_the_confusion_matrix_is_the_textbook_one(self) -> None:
        matrix = Classification(
            true_positive=3, false_positive=1, false_negative=2, true_negative=4
        )
        assert matrix.precision == 0.75
        assert matrix.recall == 0.6
        assert matrix.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)
        assert matrix.accuracy == 0.7

    def test_an_empty_matrix_does_not_divide_by_zero(self) -> None:
        assert Classification().f1 == 0.0
        assert Classification().accuracy == 0.0


class TestEvaluationCorpus:
    def test_it_builds_the_same_store_every_time(self) -> None:
        """A number is only comparable if the corpus behind it is."""
        sizes, keys = [], []
        for _ in range(2):
            with ResearchStore.in_memory() as store:
                corpus = build_corpus(store)
                sizes.append(corpus.size)
                keys.append(sorted(corpus.ids))
        assert sizes[0] == sizes[1] == 15
        assert keys[0] == keys[1]

    def test_labels_survive_into_the_store(self) -> None:
        with ResearchStore.in_memory() as store:
            corpus = build_corpus(store)
            assert corpus.labels[corpus.ids["A1"]] == "A1"
            assert corpus.resolve(["A1", "nonexistent"]) == [corpus.ids["A1"]]

    def test_deduplication_runs_rather_than_being_set_up(self) -> None:
        """The independence evaluation must measure the system, not itself."""
        with ResearchStore.in_memory() as store:
            corpus = build_corpus(store)
            copy = store.documents.get(corpus.ids["N3"])
            assert copy.duplicate_of == corpus.ids["N1"]

    def test_the_citation_graph_is_loaded(self) -> None:
        with ResearchStore.in_memory() as store:
            corpus = build_corpus(store)
            assert store.citations.count(corpus.investigation_id) > 0

    def test_an_unknown_dataset_says_so(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset("no-such-dataset")


class TestJudgements:
    """The labelled data has to be valid, or every number is."""

    def test_every_judged_document_exists(self) -> None:
        with ResearchStore.in_memory() as store:
            corpus = build_corpus(store)
            for entry in load_dataset("retrieval")["queries"]:
                for key in entry["relevant"]:
                    assert key in corpus.ids, f"{key} is judged but not in the corpus"
            for pair in load_dataset("independence")["pairs"]:
                assert pair["left"] in corpus.ids
                assert pair["right"] in corpus.ids

    def test_every_query_has_something_to_find(self) -> None:
        for entry in load_dataset("retrieval")["queries"]:
            assert entry["relevant"], f"{entry['query']!r} judges nothing relevant"

    def test_the_independence_set_has_both_classes(self) -> None:
        pairs = load_dataset("independence")["pairs"]
        judged = {bool(pair.get("independent", True)) for pair in pairs}
        assert judged == {True, False}, "a one-class set measures nothing"


class TestRetrievalEvaluation:
    @pytest.mark.asyncio
    async def test_it_scores_every_ablation_it_can_run(self) -> None:
        report = await evaluate_retrieval()
        names = {row["retrievers"] for row in report["ablations"]}
        assert names == {"lexical", "lexical+graph"}

    @pytest.mark.asyncio
    async def test_vector_rows_are_omitted_without_an_embedder(self) -> None:
        """Zeroes from an absent retriever would read as a finding."""
        report = await evaluate_retrieval()
        assert not any("vector" in row["retrievers"] for row in report["ablations"])

    @pytest.mark.asyncio
    async def test_the_ranking_is_good_enough_to_be_worth_measuring(self) -> None:
        """A regression guard, not a target: these are the numbers today."""
        report = await evaluate_retrieval()
        rows = {row["retrievers"]: row for row in report["ablations"]}
        assert rows["lexical"]["recall_at_10"] == 1.0
        assert rows["lexical"]["mrr"] >= 0.9

    @pytest.mark.asyncio
    async def test_graph_expansion_does_not_make_the_ranking_worse(self) -> None:
        """It used to. Reciprocal rank fusion over an unfiltered lexical tail
        let a document two hops from a marginal match outrank a direct one,
        and this is the guard that would catch it coming back."""
        report = await evaluate_retrieval()
        rows = {row["retrievers"]: row for row in report["ablations"]}
        assert rows["lexical+graph"]["mrr"] >= rows["lexical"]["mrr"]
        assert rows["lexical+graph"]["ndcg_at_10"] >= rows["lexical"]["ndcg_at_10"]
        assert rows["lexical+graph"]["recall_at_5"] >= rows["lexical"]["recall_at_5"]

    @pytest.mark.asyncio
    async def test_misses_are_reported_in_labelled_terms(self) -> None:
        report = await evaluate_retrieval()
        for row in report["per_query"]:
            for missed in row["missed"]:
                assert not missed.startswith("evidence:"), "report the label, not the id"

    def test_no_ablation_is_graph_alone(self) -> None:
        """Graph expansion needs lexical seeds; a graph-only row measures nothing."""
        assert all(
            "lexical" in retrievers or retrievers == ("vector",)
            for retrievers in ABLATIONS.values()
        )


class TestIndependenceEvaluation:
    @pytest.mark.asyncio
    async def test_it_scores_the_judged_pairs(self) -> None:
        report = await evaluate_independence()
        assert report["pairs"] == 12
        total = (
            report["true_positive"] + report["false_positive"]
            + report["false_negative"] + report["true_negative"]
        )
        assert total == report["pairs"]

    @pytest.mark.asyncio
    async def test_it_never_merges_two_real_sources(self) -> None:
        """The failure that understates evidence. It should stay at zero."""
        report = await evaluate_independence()
        assert report["false_positive"] == 0
        assert report["precision"] == 1.0

    @pytest.mark.asyncio
    async def test_the_misses_are_named_so_they_can_be_chased(self) -> None:
        report = await evaluate_independence()
        assert len(report["mistakes"]) == report["false_negative"] + report["false_positive"]
        for mistake in report["mistakes"]:
            assert "/" in mistake["pair"]

    @pytest.mark.asyncio
    async def test_recall_has_not_regressed(self) -> None:
        """Derived-article detection is the weak spot; this is where it is."""
        report = await evaluate_independence()
        assert report["recall"] >= 0.6


class TestRunEvaluation:
    @pytest.mark.asyncio
    async def test_a_whole_run_is_measured(self) -> None:
        from research.agents.offline_model import offline_research_model

        report = await evaluate_run(model=offline_research_model(), max_tasks=4)
        assert report["tasks"] > 0
        assert report["documents"] > 0
        assert report["stop_reason"]

    @pytest.mark.asyncio
    async def test_no_excerpt_is_unverifiable(self) -> None:
        """Zero is the only acceptable value: a quotation that does not appear
        in its document is refused at write time, so a non-zero count here
        means a guarantee is broken rather than the research being poor."""
        from research.agents.offline_model import offline_research_model

        report = await evaluate_run(model=offline_research_model(), max_tasks=4)
        assert report["unverifiable_excerpts"] == []

    @pytest.mark.asyncio
    async def test_copies_do_not_inflate_the_source_count(self) -> None:
        from research.agents.offline_model import offline_research_model

        report = await evaluate_run(model=offline_research_model(), max_tasks=8)
        assert report["independent_sources"] <= report["documents"]
