"""The report: assembled from the record, and traceable back to it."""

from __future__ import annotations

import re

import pytest

from research.acquisition.pipeline import EvidenceAcquirer
from research.llm.scripted import ScriptedModel
from research.models.claim import EvidenceStance
from research.models.common import Provenance, SourceFamily, SourceType
from research.models.investigation import InvestigationStatus, StopReason
from research.normalize.document import build_document
from research.operations.claims import ClaimOperations
from research.synthesis.report import collect, render_markdown
from research.synthesis.synthesizer import Synthesizer

WIRE = (
    "WASHINGTON (Reuters) - The utility group said on Wednesday it had agreed to "
    "terminate the flagship small modular reactor project, citing subscription levels "
    "insufficient to proceed with construction at the Idaho site."
)
REVIEW = (
    "We review 42 published cost estimates for small modular reactors and find a median "
    "projected levelized cost of $120 per megawatt hour, with vendor estimates clustering "
    "well below those of independent analysts."
)
CRITIQUE = (
    "We find no empirical basis for the claim that modularity will reverse the historical "
    "pattern of cost escalation in nuclear construction."
)


@pytest.fixture
def investigated(store):
    """An investigation with claims, copies, a contradiction and a stop reason."""
    investigation = store.investigations.create(
        "Are small modular reactors economically competitive for AI data centres?"
    )
    acquirer = EvidenceAcquirer(store)

    def add(title, text, *, source_type, family, url, publisher=None, metadata=None):
        return acquirer.persist(
            build_document(
                provider="test",
                source_type=source_type,
                source_family=family,
                provenance=Provenance(provider="test"),
                title=title,
                text=text,
                url=url,
                publisher=publisher,
                metadata=metadata,
            ),
            investigation_id=investigation.id,
        ).document

    review = add(
        "Levelized cost projections for small modular reactors",
        REVIEW,
        source_type=SourceType.ACADEMIC_PEER_REVIEWED,
        family=SourceFamily.ACADEMIC,
        url="https://doi.org/10.1016/j.enpol.2024.114001",
        publisher="Energy Policy",
    )
    critique = add(
        "Why small modular reactor cost estimates are systematically optimistic",
        CRITIQUE,
        source_type=SourceType.ACADEMIC_PEER_REVIEWED,
        family=SourceFamily.ACADEMIC,
        url="https://doi.org/10.1016/j.erss.2023.103211",
        publisher="Energy Research & Social Science",
    )
    original = add(
        "Utility group terminates flagship reactor project",
        WIRE,
        source_type=SourceType.ORIGINAL_NEWS_REPORTING,
        family=SourceFamily.NEWS,
        url="https://www.reuters.test/business/a",
    )
    copy = add(
        "Utility group terminates flagship reactor project",
        WIRE,
        source_type=SourceType.SECONDARY_NEWS_REPORTING,
        family=SourceFamily.NEWS,
        url="https://www.gazette.test/wire/a",
    )

    operations = ClaimOperations(store, investigation_id=investigation.id)
    supported = operations.create_claim("The flagship reactor project was terminated")
    operations.link_evidence(
        supported.id,
        original.id,
        EvidenceStance.SUPPORTS,
        excerpt="agreed to terminate the flagship small modular reactor project",
    )
    operations.link_evidence(supported.id, copy.id, EvidenceStance.SUPPORTS)

    contested = operations.create_claim("Modularity will reverse historical cost escalation")
    operations.link_evidence(
        contested.id, review.id, EvidenceStance.SUPPORTS, analysis="reports a median estimate"
    )
    operations.link_evidence(
        contested.id,
        critique.id,
        EvidenceStance.CONTRADICTS,
        excerpt="no empirical basis for the claim that modularity will reverse",
    )

    store.investigations.set_status(
        investigation.id,
        InvestigationStatus.COMPLETED,
        stop_reason=StopReason.DIMINISHING_RETURNS,
        stop_detail="new searches returned material already held",
    )
    return store, investigation, {"supported": supported, "contested": contested,
                                  "original": original, "copy": copy}


class TestCollection:
    def test_sources_are_counted_once_with_their_copies_listed(self, investigated) -> None:
        store, investigation, fixtures = investigated
        data = collect(store, investigation.id)
        assert data.counts["documents"] == 4
        assert data.counts["independent_sources"] == 3
        wire_source = next(
            source for source in data.sources if source.document.id == fixtures["original"].id
        )
        assert [copy.id for copy in wire_source.copies] == [fixtures["copy"].id]

    def test_findings_are_ordered_by_how_well_evidenced_they_are(self, investigated) -> None:
        store, investigation, _ = investigated
        findings = collect(store, investigation.id).findings()
        assert [str(finding.status) for finding in findings] == ["supported", "mixed"]

    def test_contested_claims_are_identified(self, investigated) -> None:
        store, investigation, fixtures = investigated
        contested = collect(store, investigation.id).contested()
        assert [assessment.claim_id for assessment in contested] == [fixtures["contested"].id]


class TestReport:
    @pytest.fixture
    def markdown(self, investigated) -> str:
        store, investigation, _ = investigated
        return render_markdown(collect(store, investigation.id), store)

    def test_the_report_has_the_sections_a_reader_needs(self, markdown: str) -> None:
        for heading in (
            "## Executive summary",
            "## Key findings",
            "## Contradictory and qualifying evidence",
            "## Areas of uncertainty",
            "## How this investigation ended",
            "## Sources",
        ):
            assert heading in markdown

    def test_every_finding_carries_evidence_identifiers(self, investigated) -> None:
        store, investigation, _ = investigated
        data = collect(store, investigation.id)
        markdown = render_markdown(data, store)
        for assessment in data.findings():
            assert assessment.text in markdown
            for document_id in assessment.support.document_ids:
                assert document_id in markdown

    def test_quotations_appear_with_their_document(self, markdown: str, investigated) -> None:
        store, _, fixtures = investigated
        assert "agreed to terminate the flagship small modular reactor project" in markdown
        assert f"`{fixtures['original'].id}`" in markdown

    def test_analysis_is_labelled_as_analysis(self, markdown: str) -> None:
        assert "*analysis:*" in markdown

    def test_copies_are_declared_rather_than_counted(self, markdown: str, investigated) -> None:
        _, _, fixtures = investigated
        assert "not independent of" in markdown
        assert "counted once" in markdown
        assert fixtures["copy"].id in markdown

    def test_the_stopping_reason_is_reported(self, markdown: str) -> None:
        assert "diminishing_returns" in markdown
        assert "material already held" in markdown

    def test_absent_counterevidence_is_not_read_as_confirmation(self, store) -> None:
        investigation = store.investigations.create("a question with no research behind it")
        markdown = render_markdown(collect(store, investigation.id), store)
        assert "That is a gap in the research, not a confirmation" in markdown

    def test_every_evidence_reference_resolves(self, investigated) -> None:
        store, investigation, _ = investigated
        markdown = render_markdown(collect(store, investigation.id), store)
        referenced = set(re.findall(r"evidence:\d+", markdown))
        held = {
            document.id for document in store.documents.list(investigation.id, limit=100)
        }
        assert referenced
        assert referenced <= held, "a report never cites a document that is not held"


class TestSynthesizer:
    async def test_model_prose_is_used_when_it_checks_out(self, investigated) -> None:
        store, investigation, fixtures = investigated
        summary = (
            f"The project was terminated ({fixtures['supported'].id}), and the claim that "
            f"modularity reverses cost escalation is contested ({fixtures['contested'].id})."
        )
        synthesis = await Synthesizer(
            store, investigation_id=investigation.id, model=ScriptedModel([summary])
        ).run()
        assert synthesis.summary_was_written
        assert summary in synthesis.markdown
        assert not synthesis.invalid_references

    async def test_a_summary_citing_things_that_do_not_exist_is_discarded(
        self, investigated
    ) -> None:
        store, investigation, _ = investigated
        synthesis = await Synthesizer(
            store,
            investigation_id=investigation.id,
            model=ScriptedModel(["Everything is settled, see claim:99 and evidence:88."]),
        ).run()
        assert synthesis.summary is None
        assert synthesis.invalid_references == ["claim:99", "evidence:88"]
        # The report is still produced, from the record.
        assert "## Key findings" in synthesis.markdown

    async def test_a_failing_model_costs_only_the_prose(self, investigated) -> None:
        store, investigation, _ = investigated
        synthesis = await Synthesizer(
            store, investigation_id=investigation.id, model=ScriptedModel([])
        ).run()
        assert synthesis.summary is None
        assert "## Sources" in synthesis.markdown

    async def test_synthesis_does_no_new_research(self, investigated) -> None:
        store, investigation, _ = investigated
        searches_before = store.queries.count(investigation.id)
        documents_before = store.documents.count(investigation.id)
        await Synthesizer(
            store,
            investigation_id=investigation.id,
            model=ScriptedModel(["A summary drawn from the record."]),
        ).run()
        assert store.queries.count(investigation.id) == searches_before
        assert store.documents.count(investigation.id) == documents_before

    async def test_the_model_sees_the_record_not_the_corpus(self, investigated) -> None:
        store, investigation, _ = investigated
        model = ScriptedModel(["A summary."])
        await Synthesizer(store, investigation_id=investigation.id, model=model).run()
        prompts = "\n".join(model.prompts())
        assert "claim:" in prompts
        assert WIRE not in prompts, "the synthesizer reasons over claims, not page text"
