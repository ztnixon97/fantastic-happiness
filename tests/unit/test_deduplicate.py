"""Deduplication and source independence."""

from __future__ import annotations

import pytest

from research.acquisition.deduplicate import DuplicateDetector, independent_documents
from research.normalize.document import build_document, detect_wire_service
from research.acquisition.pipeline import EvidenceAcquirer, summarise
from research.models.common import (
    DuplicateRelation,
    Provenance,
    SourceFamily,
    SourceType,
)

WIRE_STORY = (
    "WASHINGTON (Reuters) - NuScale Power and the utility group said on Wednesday they had "
    "agreed to terminate the flagship small modular reactor project, citing insufficient "
    "subscription levels from participating utilities. The target price had risen to about "
    "$89 per megawatt hour from $58 per megawatt hour in earlier estimates. Analysts called "
    "the cancellation a setback for a technology promoted as a cheaper route to new capacity."
)

OTHER_STORY = (
    "State regulators opened a hearing on Tuesday into whether ratepayers should finance new "
    "generation capacity for data centre growth, hearing testimony from consumer advocates and "
    "two utilities about cost allocation and rate design over the next decade."
)


def news(title: str, text: str, url: str, **overrides):
    return build_document(
        provider="fetch",
        source_type=overrides.pop("source_type", SourceType.SECONDARY_NEWS_REPORTING),
        source_family=SourceFamily.NEWS,
        provenance=Provenance(provider="fetch"),
        title=title,
        text=text,
        url=url,
        **overrides,
    )


def paper(title: str, *, doi: str | None = None, provider: str = "openalex", external_id: str = "W1", abstract: str = "abstract text about reactors"):
    return build_document(
        provider=provider,
        source_type=SourceType.ACADEMIC_PEER_REVIEWED,
        source_family=SourceFamily.ACADEMIC,
        provenance=Provenance(provider=provider),
        title=title,
        abstract=abstract,
        doi=doi,
        external_id=external_id,
        url=f"https://doi.org/{doi}" if doi else None,
    )


@pytest.fixture
def acquirer(store, investigation):
    return EvidenceAcquirer(store), investigation.id


class TestWireDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("WASHINGTON (Reuters) - the company said", "Reuters"),
            ("By Associated Press - March 4", "Associated Press"),
            ("PARIS (AFP) - officials confirmed", "Agence France-Presse"),
            ("Our reporter found documents showing", None),
        ],
    )
    def test_detects_credit_lines(self, text: str, expected: str | None) -> None:
        assert detect_wire_service(text) == expected

    def test_publisher_name_alone_is_enough(self) -> None:
        assert detect_wire_service(None, "Reuters") == "Reuters"


class TestDuplicateDetection:
    def test_first_document_is_independent(self, acquirer) -> None:
        acq, investigation_id = acquirer
        result = acq.persist(
            news("Project terminated", WIRE_STORY, "https://reuters.com/a"),
            investigation_id=investigation_id,
        )
        assert result.is_new_evidence
        assert result.verdict.independent

    def test_same_doi_from_two_providers_is_one_record(self, acquirer) -> None:
        acq, investigation_id = acquirer
        first = acq.persist(
            paper("A review of reactor costs", doi="10.1016/j.enpol.2024.114001", provider="openalex"),
            investigation_id=investigation_id,
        )
        second = acq.persist(
            paper(
                "A Review of Reactor Costs",
                doi="10.1016/J.ENPOL.2024.114001",
                provider="crossref",
                external_id="10.1016/j.enpol.2024.114001",
                abstract="a fuller abstract from the other provider",
            ),
            investigation_id=investigation_id,
        )
        assert second.verdict.relation is DuplicateRelation.SAME_RECORD
        assert not second.created
        assert second.document.id == first.document.id
        # The second provider contributes what the first lacked, and is credited.
        assert second.enriched
        assert "crossref" in second.document.metadata["also_provided_by"]

    def test_syndicated_copy_on_another_site_is_not_independent(self, acquirer) -> None:
        acq, investigation_id = acquirer
        original = acq.persist(
            news("Project terminated", WIRE_STORY, "https://www.reuters.com/business/a"),
            investigation_id=investigation_id,
        )
        copy = acq.persist(
            news(
                "Reactor project ends",
                WIRE_STORY + " Sign up for our newsletter.",
                "https://www.localpaper.example/wire/a",
            ),
            investigation_id=investigation_id,
        )
        assert copy.verdict.relation is DuplicateRelation.SYNDICATED_COPY
        assert copy.created  # the copy is kept: that it was syndicated is evidence
        assert not copy.is_new_evidence
        assert copy.document.duplicate_of == original.document.id
        assert copy.document.independence_key == original.document.independence_key

    def test_ten_copies_of_one_story_are_one_source(self, acquirer) -> None:
        acq, investigation_id = acquirer
        documents = [news("Project terminated", WIRE_STORY, "https://www.reuters.com/business/a")]
        documents += [
            news(f"Reactor project ends {index}", WIRE_STORY, f"https://outlet{index}.example/wire")
            for index in range(9)
        ]
        results = acq.persist_all(documents, investigation_id=investigation_id)
        ids = [result.document.id for result in results]
        groups = independent_documents(acq.store.documents, ids)
        assert len(ids) == 10
        assert len(groups) == 1, "ten republications of one wire story are one source"
        assert summarise(results)["new_evidence"] == 1

    def test_unrelated_story_stays_independent(self, acquirer) -> None:
        acq, investigation_id = acquirer
        acq.persist(
            news("Project terminated", WIRE_STORY, "https://www.reuters.com/business/a"),
            investigation_id=investigation_id,
        )
        other = acq.persist(
            news("Regulators open hearing", OTHER_STORY, "https://www.othernews.example/hearing"),
            investigation_id=investigation_id,
        )
        assert other.is_new_evidence

    def test_same_url_is_the_same_artifact(self, acquirer) -> None:
        acq, investigation_id = acquirer
        first = acq.persist(
            news("Story", WIRE_STORY, "https://www.site.example/a"),
            investigation_id=investigation_id,
        )
        again = acq.persist(
            news("Story", WIRE_STORY, "https://www.site.example/a?utm_source=x"),
            investigation_id=investigation_id,
        )
        assert again.document.id == first.document.id
        assert not again.created

    def test_article_crediting_another_is_marked_derived(self, acquirer) -> None:
        acq, investigation_id = acquirer
        original = acq.persist(
            news("Original reporting", OTHER_STORY, "https://apnews.example/article/x"),
            investigation_id=investigation_id,
        )
        derived = news(
            "Rewrite of the story",
            "Regulators are looking at who pays for data centre power, according to a report. "
            "The hearing will consider cost allocation.",
            "https://aggregator.example/posts/y",
        )
        derived.metadata["outbound_links"] = [
            ["https://apnews.example/article/x", "AP News"],
            ["https://aggregator.example/about", "About us"],
        ]
        result = acq.persist(derived, investigation_id=investigation_id)
        assert result.verdict.relation is DuplicateRelation.DERIVED_ARTICLE
        assert result.document.derived_from == original.document.id
        assert not result.is_new_evidence

    def test_detection_is_scoped_to_one_investigation(self, store) -> None:
        first = store.investigations.create("one")
        second = store.investigations.create("two")
        acq = EvidenceAcquirer(store)
        acq.persist(news("Story", WIRE_STORY, "https://reuters.com/a"), investigation_id=first.id)
        result = acq.persist(
            news("Story", WIRE_STORY, "https://reuters.com/a"), investigation_id=second.id
        )
        assert result.is_new_evidence


class TestExplanations:
    def test_verdicts_explain_themselves(self, acquirer) -> None:
        acq, investigation_id = acquirer
        acq.persist(
            news("Project terminated", WIRE_STORY, "https://www.reuters.com/business/a"),
            investigation_id=investigation_id,
        )
        copy = acq.persist(
            news("Project ends", WIRE_STORY, "https://www.localpaper.example/wire/a"),
            investigation_id=investigation_id,
        )
        assert "syndicated" in copy.verdict.explanation
        assert copy.document.metadata["duplicate_reason"] == copy.verdict.explanation
        assert copy.document.metadata["duplicate_similarity"] >= 0.7

    def test_detector_reports_closest_match_even_when_independent(self, store, investigation) -> None:
        acq = EvidenceAcquirer(store)
        acq.persist(
            news("Project terminated", WIRE_STORY, "https://www.reuters.com/business/a"),
            investigation_id=investigation.id,
        )
        detector = DuplicateDetector(store.documents)
        verdict = detector.check(
            news("Hearing opens", OTHER_STORY, "https://other.example/x"), investigation.id
        )
        assert verdict.independent
