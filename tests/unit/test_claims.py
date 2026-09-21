"""Claim operations and the assessment that replaces a confidence score."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from research.errors import IntegrityError
from research.graph.claims import derive_status, explain
from research.models.claim import ClaimStatus, EvidenceStance
from research.models.common import (
    DuplicateRelation,
    Provenance,
    SourceFamily,
    SourceType,
)
from research.normalize.document import build_document
from research.operations.claims import ClaimOperations, excerpt_appears_in
from research.acquisition.pipeline import EvidenceAcquirer


def dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


@pytest.fixture
def workspace(store, investigation):
    acquirer = EvidenceAcquirer(store)
    operations = ClaimOperations(store, investigation_id=investigation.id, task_id="task:1")

    def add(
        title: str,
        text: str,
        *,
        source_type: SourceType = SourceType.ACADEMIC_PEER_REVIEWED,
        family: SourceFamily = SourceFamily.ACADEMIC,
        url: str | None = None,
        published: datetime | None = None,
        metadata: dict | None = None,
        publisher: str | None = None,
    ):
        document = build_document(
            provider="test",
            source_type=source_type,
            source_family=family,
            provenance=Provenance(provider="test"),
            title=title,
            text=text,
            url=url or f"https://example{abs(hash(title)) % 997}.test/{abs(hash(text)) % 997}",
            published_at=published,
            publisher=publisher,
            metadata=metadata,
        )
        return acquirer.persist(document, investigation_id=investigation.id).document

    return store, investigation, operations, add


class TestClaimCreation:
    def test_a_new_claim_is_unverified(self, workspace) -> None:
        _, _, operations, _ = workspace
        claim = operations.create_claim("SMR costs exceed early projections")
        assert claim.status is ClaimStatus.UNVERIFIED
        assert operations.get_claim(claim.id).explanation.startswith("no evidence")

    def test_empty_claims_are_refused(self, workspace) -> None:
        _, _, operations, _ = workspace
        with pytest.raises(IntegrityError):
            operations.create_claim("   ")

    def test_claims_can_refine_one_another(self, workspace) -> None:
        store, _, operations, _ = workspace
        broad = operations.create_claim("SMRs are uneconomic")
        narrow = operations.create_claim(
            "First-of-a-kind SMR projects have exceeded their initial cost estimates"
        )
        operations.relate_claims(narrow.id, broad.id, "REFINES")
        assert store.claims.get(narrow.id).related_claim_ids == [broad.id]

    def test_unknown_relations_are_refused(self, workspace) -> None:
        _, _, operations, _ = workspace
        first = operations.create_claim("a")
        second = operations.create_claim("b")
        with pytest.raises(IntegrityError):
            operations.relate_claims(first.id, second.id, "VIBES_WITH")
        with pytest.raises(IntegrityError):
            operations.relate_claims(first.id, first.id, "REFINES")


class TestExcerptIntegrity:
    def test_a_quotation_must_appear_in_the_document(self, workspace) -> None:
        _, _, operations, add = workspace
        document = add("Cost review", "Realized costs exceeded estimates by 117 percent.")
        claim = operations.create_claim("Nuclear projects overrun their estimates")
        result = operations.link_evidence(
            claim.id,
            document.id,
            EvidenceStance.SUPPORTS,
            excerpt="costs exceeded estimates by 117 percent",
        )
        assert result.excerpt_verified

    def test_a_fabricated_quotation_is_refused(self, workspace) -> None:
        _, _, operations, add = workspace
        document = add("Cost review", "Realized costs exceeded estimates by 117 percent.")
        claim = operations.create_claim("Costs tripled")
        with pytest.raises(IntegrityError) as exc:
            operations.link_evidence(
                claim.id,
                document.id,
                EvidenceStance.SUPPORTS,
                excerpt="Costs have tripled, according to internal documents",
            )
        assert "does not appear" in str(exc.value)
        assert operations.get_claim(claim.id).support.count == 0

    def test_typographic_differences_do_not_make_a_quote_false(self, workspace) -> None:
        _, _, operations, add = workspace
        document = add(
            "Statement",
            "The company said “we remain confident” — despite the delay.",
        )
        assert excerpt_appears_in(document, 'we remain confident" - despite the delay')

    def test_analysis_is_stored_apart_from_evidence(self, workspace) -> None:
        store, _, operations, add = workspace
        document = add("Paper", "We model three deployment scenarios.")
        claim = operations.create_claim("Deployment scenarios are contested")
        operations.link_evidence(
            claim.id,
            document.id,
            EvidenceStance.QUALIFIES,
            analysis="the modelling assumes an order book no announced pipeline supports",
        )
        link = store.claims.evidence_links(claim.id)[0]
        assert link.excerpt is None
        assert "order book" in link.analysis

    def test_evidence_from_another_investigation_is_refused(self, store, investigation) -> None:
        other = store.investigations.create("another question")
        acquirer = EvidenceAcquirer(store)
        document = acquirer.persist(
            build_document(
                provider="test",
                source_type=SourceType.WEB_PAGE,
                source_family=SourceFamily.WEB,
                provenance=Provenance(provider="test"),
                title="Elsewhere",
                text="body text that belongs to another investigation entirely",
                url="https://elsewhere.test/a",
            ),
            investigation_id=other.id,
        ).document
        operations = ClaimOperations(store, investigation_id=investigation.id)
        claim = operations.create_claim("a claim")
        with pytest.raises(IntegrityError):
            operations.link_evidence(claim.id, document.id, EvidenceStance.SUPPORTS)


class TestAssessment:
    def test_independent_sources_are_counted_not_documents(self, workspace) -> None:
        store, investigation, operations, add = workspace
        wire = (
            "WASHINGTON (Reuters) - The utility group said on Wednesday it had agreed to "
            "terminate the flagship reactor project after subscriptions fell short of the "
            "level needed to proceed with construction."
        )
        original = add(
            "Utility group terminates reactor project",
            wire,
            source_type=SourceType.ORIGINAL_NEWS_REPORTING,
            family=SourceFamily.NEWS,
            url="https://www.reuters.test/business/a",
            published=dt(2023, 11, 8),
        )
        copy = add(
            "Utility group terminates reactor project",
            wire,
            source_type=SourceType.SECONDARY_NEWS_REPORTING,
            family=SourceFamily.NEWS,
            url="https://www.gazette.test/wire/a",
            published=dt(2023, 11, 8),
        )
        assert copy.duplicate_relation is DuplicateRelation.SYNDICATED_COPY

        claim = operations.create_claim("The flagship project was terminated")
        operations.link_evidence(claim.id, original.id, EvidenceStance.SUPPORTS)
        result = operations.link_evidence(claim.id, copy.id, EvidenceStance.SUPPORTS)

        assessment = operations.get_claim(claim.id)
        assert result.not_independent_of == original.id
        assert assessment.support.count == 2
        assert assessment.support.independent_count == 1
        assert "1 independent" in assessment.explanation
        assert "counted once" in assessment.explanation

    def test_a_company_announcement_about_itself_is_not_support(self, workspace) -> None:
        _, _, operations, add = workspace
        release = add(
            "Company announces agreement",
            "The company today announced a landmark agreement to deliver 900 megawatts.",
            source_type=SourceType.PRESS_RELEASE,
            family=SourceFamily.WEB,
            published=dt(2025, 2, 18),
        )
        claim = operations.create_claim("The plant will deliver 900 megawatts")
        operations.link_evidence(claim.id, release.id, EvidenceStance.SUPPORTS)
        assessment = operations.get_claim(claim.id)
        assert assessment.status is ClaimStatus.INSUFFICIENT_EVIDENCE
        assert "announcing party" in assessment.explanation
        assert any("announcing party" in gap for gap in assessment.gaps)

    def test_a_primary_record_is_recognised(self, workspace) -> None:
        _, _, operations, add = workspace
        filing = add(
            "Annual report",
            "Our estimated cost of electricity for a first-of-a-kind plant is subject to "
            "material uncertainty and prior estimates have been revised upward.",
            source_type=SourceType.CORPORATE_FILING,
            family=SourceFamily.CORPORATE,
            published=dt(2025, 2, 19),
        )
        claim = operations.create_claim("The company has revised its cost estimates upward")
        operations.link_evidence(
            claim.id,
            filing.id,
            EvidenceStance.SUPPORTS,
            excerpt="prior estimates have been revised upward",
        )
        assessment = operations.get_claim(claim.id)
        assert assessment.status is ClaimStatus.SUPPORTED
        assert assessment.support.has_primary_source
        assert "primary" in assessment.explanation

    def test_mixed_evidence_says_so(self, workspace) -> None:
        _, _, operations, add = workspace
        supporting = add(
            "Learning rates paper",
            "Factory fabrication produced learning rates of 8 to 12 percent per doubling.",
            published=dt(2022, 9, 15),
        )
        against = add(
            "Critique",
            "We find no empirical basis for the claim that modularity reverses escalation.",
            published=dt(2023, 8, 2),
        )
        claim = operations.create_claim("Modular construction will lower reactor costs")
        operations.link_evidence(claim.id, supporting.id, EvidenceStance.SUPPORTS)
        operations.link_evidence(claim.id, against.id, EvidenceStance.CONTRADICTS)
        assessment = operations.get_claim(claim.id)
        assert assessment.status is ClaimStatus.MIXED
        assert assessment.possibly_superseded, "the rebuttal is newer than the support"
        assert "contradicted by" in assessment.explanation

    def test_retracted_support_does_not_count(self, workspace) -> None:
        _, _, operations, add = workspace
        retracted = add(
            "Rapid cost declines",
            "We report cost declines of 40 percent per doubling in factory-built components.",
            metadata={"is_retracted": True},
            published=dt(2022, 2, 11),
        )
        claim = operations.create_claim("Factory production has halved component costs")
        operations.link_evidence(claim.id, retracted.id, EvidenceStance.SUPPORTS)
        assessment = operations.get_claim(claim.id)
        assert assessment.status is ClaimStatus.INSUFFICIENT_EVIDENCE
        assert "retracted" in assessment.explanation

    def test_status_is_persisted_with_its_explanation(self, workspace) -> None:
        store, _, operations, add = workspace
        document = add("Paper", "Realized costs exceeded estimates by 117 percent.")
        claim = operations.create_claim("Projects overrun")
        operations.link_evidence(claim.id, document.id, EvidenceStance.SUPPORTS)
        stored = store.claims.get(claim.id)
        assert stored.status is ClaimStatus.SUPPORTED
        assert stored.notes and "independent" in stored.notes

    def test_preprint_only_support_is_flagged(self, workspace) -> None:
        _, _, operations, add = workspace
        preprint = add(
            "Preprint",
            "We model hyperscale load growth under three AI deployment scenarios.",
            source_type=SourceType.ACADEMIC_PREPRINT,
            published=dt(2025, 1, 30),
        )
        claim = operations.create_claim("Firm capacity is met most cheaply without reactors")
        operations.link_evidence(claim.id, preprint.id, EvidenceStance.SUPPORTS)
        assessment = operations.get_claim(claim.id)
        assert "not peer reviewed" in assessment.explanation
        assert any("peer-reviewed" in gap for gap in assessment.gaps)


class TestOpenQuestions:
    def test_thin_claims_surface_with_what_would_fix_them(self, workspace) -> None:
        _, _, operations, add = workspace
        document = add(
            "One report",
            "A single outlet reported the agreement was signed last week in the capital.",
            source_type=SourceType.SECONDARY_NEWS_REPORTING,
            family=SourceFamily.NEWS,
        )
        claim = operations.create_claim("The agreement was signed")
        operations.link_evidence(claim.id, document.id, EvidenceStance.SUPPORTS)
        questions = operations.open_questions()
        assert questions and questions[0]["claim_id"] == claim.id
        assert any("counterevidence" in gap for gap in questions[0]["gaps"])
        assert any("primary record" in gap for gap in questions[0]["gaps"])


class TestStatusRules:
    def test_no_evidence_is_unverified(self) -> None:
        from research.graph.claims import EvidenceSummary

        assert derive_status(EvidenceSummary(), EvidenceSummary()) is ClaimStatus.UNVERIFIED

    def test_explanation_without_evidence_says_so(self) -> None:
        from research.graph.claims import EvidenceSummary

        assert "no evidence" in explain(EvidenceSummary(), EvidenceSummary())
