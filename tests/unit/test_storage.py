"""Persistence: identifiers, documents, claims, entities, events, logs."""

from __future__ import annotations

import pytest

from research.errors import NotFound
from research.ids import is_id, parse_id
from research.models.claim import ClaimEvidenceLink, ClaimStatus, EvidenceStance
from research.models.common import (
    DuplicateRelation,
    Provenance,
    SourceFamily,
    SourceType,
    utcnow,
)
from research.models.entity import EntityIdentifier, EntityType
from research.models.event import DatePrecision
from research.models.evidence import EvidenceDocument
from research.models.investigation import InvestigationStatus, StopReason
from research.models.task import (
    FollowUp,
    Operation,
    ResearchRole,
    ResearchTask,
    TaskResult,
    TaskStatus,
)
from research.storage.store import ResearchStore


def make_document(store: ResearchStore, investigation_id: str, **overrides) -> EvidenceDocument:
    defaults = dict(
        id=store.documents.new_id(),
        investigation_id=investigation_id,
        source_type=SourceType.ACADEMIC_PEER_REVIEWED,
        source_family=SourceFamily.ACADEMIC,
        provider="openalex",
        external_id="W1",
        canonical_url="https://doi.org/10.1/a",
        title="A paper about reactors and their costs",
        authors=["Jane Roe"],
        published_at=utcnow(),
        content_hash="sha256:abc",
        doi="10.1/a",
        title_key="paper-about-reactors",
        simhash=0xFFFFFFFFFFFFFFFF,
        provenance=Provenance(provider="openalex"),
    )
    defaults.update(overrides)
    return store.documents.add(EvidenceDocument(**defaults))


class TestIdentifiers:
    def test_ids_are_sequential_per_prefix(self, store: ResearchStore) -> None:
        assert [store.db.next_id("evidence") for _ in range(3)] == [
            "evidence:1",
            "evidence:2",
            "evidence:3",
        ]
        assert store.db.next_id("claim") == "claim:1"

    def test_parsing_rejects_malformed_references(self) -> None:
        assert parse_id("evidence:42") == ("evidence", 42)
        assert not is_id("evidence:abc")
        assert not is_id("; DROP TABLE documents")
        with pytest.raises(ValueError):
            parse_id("nonsense")

    def test_allocation_survives_reopening(self, tmp_path) -> None:
        path = tmp_path / "r.sqlite3"
        with ResearchStore.open(path) as first:
            assert first.db.next_id("evidence") == "evidence:1"
        with ResearchStore.open(path) as second:
            assert second.db.next_id("evidence") == "evidence:2"


class TestDocuments:
    def test_round_trip_preserves_every_field(self, store, investigation) -> None:
        original = make_document(store, investigation.id)
        loaded = store.documents.get(original.id)
        assert loaded.title == original.title
        assert loaded.doi == "10.1/a"
        assert loaded.authors == ["Jane Roe"]
        assert loaded.simhash == 0xFFFFFFFFFFFFFFFF  # 64-bit value survives SQLite
        assert loaded.provenance is not None
        assert loaded.provenance.provider == "openalex"

    def test_missing_document_raises(self, store) -> None:
        with pytest.raises(NotFound):
            store.documents.get("evidence:999")

    def test_identity_index_supports_lookup_by_any_scheme(self, store, investigation) -> None:
        document = make_document(store, investigation.id)
        assert store.documents.find_by_identity(investigation.id, "doi", "10.1/a") == document.id
        assert (
            store.documents.find_by_identity(investigation.id, "provider_id", "openalex:W1")
            == document.id
        )
        assert store.documents.find_by_identity(investigation.id, "doi", "10.9/z") is None

    def test_identity_index_is_scoped_to_one_investigation(self, store) -> None:
        first = store.investigations.create("one")
        second = store.investigations.create("two")
        make_document(store, first.id)
        assert store.documents.find_by_identity(second.id, "doi", "10.1/a") is None

    def test_marking_a_duplicate_groups_it_with_the_original(self, store, investigation) -> None:
        original = make_document(store, investigation.id)
        copy = make_document(
            store, investigation.id, external_id="W2", doi=None, canonical_url="https://b.example/x"
        )
        store.documents.mark_duplicate(
            copy.id,
            duplicate_of=original.id,
            relation=DuplicateRelation.SYNDICATED_COPY,
            independence_key=original.independence_key,
        )
        groups = store.documents.independence_groups([original.id, copy.id])
        assert list(groups.values()) == [[original.id, copy.id]]
        assert [d.id for d in store.documents.duplicates_of(original.id)] == [copy.id]

    def test_counts_can_exclude_copies(self, store, investigation) -> None:
        original = make_document(store, investigation.id)
        copy = make_document(store, investigation.id, external_id="W2", doi=None, canonical_url=None)
        store.documents.mark_duplicate(
            copy.id, duplicate_of=original.id, relation=DuplicateRelation.EXACT_DUPLICATE
        )
        assert store.documents.count(investigation.id) == 2
        assert store.documents.count(investigation.id, originals_only=True) == 1
        assert store.documents.count(investigation.id, family=SourceFamily.NEWS) == 0

    def test_update_persists_enrichment(self, store, investigation) -> None:
        document = make_document(store, investigation.id, text=None)
        document.text = "full text arrived later"
        store.documents.update(document)
        assert store.documents.get(document.id).text == "full text arrived later"


class TestTasks:
    def test_task_result_round_trips_through_storage(self, store, investigation) -> None:
        task = store.tasks.create(
            ResearchTask(
                id="",
                investigation_id=investigation.id,
                role=ResearchRole.ACADEMIC,
                operation=Operation.SEARCH_ACADEMIC,
                objective="find cost literature",
                parameters={"query": "smr cost"},
            )
        )
        result = TaskResult(
            summary="found three relevant reviews",
            claim_ids=["claim:1"],
            evidence_ids=["evidence:1", "evidence:2"],
            open_questions=["what do vendors assume about order volume?"],
            recommended_followups=[
                FollowUp(
                    operation=Operation.FIND_COUNTEREVIDENCE,
                    objective="look for failed SMR projects",
                    priority=2,
                )
            ],
        )
        store.tasks.finish(task.id, TaskStatus.COMPLETED, result=result)

        loaded = store.tasks.get(task.id)
        assert loaded.status is TaskStatus.COMPLETED
        assert loaded.result is not None
        assert loaded.result.evidence_ids == ["evidence:1", "evidence:2"]
        assert loaded.result.recommended_followups[0].operation is Operation.FIND_COUNTEREVIDENCE
        assert loaded.parameters == {"query": "smr cost"}

    def test_pending_tasks_come_back_in_priority_order(self, store, investigation) -> None:
        for priority in (5, 1, 3):
            store.tasks.create(
                ResearchTask(
                    id="",
                    investigation_id=investigation.id,
                    role=ResearchRole.SCOUT,
                    operation=Operation.SEARCH_WEB,
                    objective=f"p{priority}",
                    priority=priority,
                )
            )
        assert store.tasks.next_pending(investigation.id).objective == "p1"


class TestClaims:
    def test_claim_hydrates_its_evidence_by_stance(self, store, investigation) -> None:
        support = make_document(store, investigation.id)
        against = make_document(store, investigation.id, external_id="W9", doi="10.1/b")
        claim = store.claims.create(investigation.id, "SMR costs exceed projections")
        store.claims.link_evidence(
            ClaimEvidenceLink(
                claim_id=claim.id,
                document_id=support.id,
                stance=EvidenceStance.SUPPORTS,
                excerpt="costs exceeded estimates by 117 percent",
            )
        )
        store.claims.link_evidence(
            ClaimEvidenceLink(
                claim_id=claim.id, document_id=against.id, stance=EvidenceStance.CONTRADICTS
            )
        )
        loaded = store.claims.get(claim.id)
        assert loaded.supporting_evidence_ids == [support.id]
        assert loaded.contradicting_evidence_ids == [against.id]

    def test_evidence_links_are_not_duplicated(self, store, investigation) -> None:
        document = make_document(store, investigation.id)
        claim = store.claims.create(investigation.id, "a claim")
        link = ClaimEvidenceLink(
            claim_id=claim.id, document_id=document.id, stance=EvidenceStance.SUPPORTS
        )
        store.claims.link_evidence(link)
        store.claims.link_evidence(link)
        assert len(store.claims.evidence_links(claim.id)) == 1

    def test_status_and_notes_replace_a_confidence_score(self, store, investigation) -> None:
        claim = store.claims.create(investigation.id, "a claim")
        store.claims.set_status(
            claim.id,
            ClaimStatus.MIXED,
            notes="supported by one preprint and contradicted by two later studies",
        )
        loaded = store.claims.get(claim.id)
        assert loaded.status is ClaimStatus.MIXED
        assert "preprint" in (loaded.notes or "")

    def test_documents_report_the_claims_they_bear_on(self, store, investigation) -> None:
        document = make_document(store, investigation.id)
        claim = store.claims.create(investigation.id, "a claim")
        store.claims.link_evidence(
            ClaimEvidenceLink(
                claim_id=claim.id, document_id=document.id, stance=EvidenceStance.SUPPORTS
            )
        )
        assert [link.claim_id for link in store.claims.claims_for_document(document.id)] == (
            [claim.id]
        )


class TestEntitiesAndEvents:
    def test_entities_resolve_by_authoritative_identifier(self, store, investigation) -> None:
        entity = store.entities.create(
            investigation.id,
            EntityType.ORGANIZATION,
            "NuScale Power",
            identifiers=[EntityIdentifier("sec_cik", "0001822966")],
            aliases=["NuScale"],
        )
        found = store.entities.find_by_identifier(investigation.id, "sec_cik", "0001822966")
        assert found is not None and found.id == entity.id
        assert found.identifier("sec_cik") == "0001822966"

    def test_entities_are_findable_by_alias(self, store, investigation) -> None:
        entity = store.entities.create(
            investigation.id, EntityType.ORGANIZATION, "NuScale Power", aliases=["NuScale"]
        )
        assert [e.id for e in store.entities.find_by_name(
            investigation.id, EntityType.ORGANIZATION, "nuscale"
        )] == [entity.id]

    def test_timeline_orders_events_and_keeps_undated_ones_last(self, store, investigation) -> None:
        document = make_document(store, investigation.id)
        later = store.events.create(
            investigation.id,
            "project cancelled",
            date_start=utcnow(),
            date_precision=DatePrecision.DAY,
            evidence_ids=[document.id],
        )
        undated = store.events.create(investigation.id, "cost estimate revised")
        timeline = store.events.timeline(investigation.id)
        assert [event.id for event in timeline] == [later.id, undated.id]
        assert timeline[0].evidence_ids == [document.id]


class TestLogsAndInvestigations:
    def test_fetch_log_records_failures(self, store, investigation) -> None:
        fetch_id = store.fetches.begin(
            provider="fetch", url="https://e.com/x", investigation_id=investigation.id
        )
        store.fetches.complete(fetch_id, ok=False, error="timeout", status_code=None)
        assert store.fetches.failure_count(investigation.id) == 1
        assert store.fetches.list(investigation.id)[0]["error"] == "timeout"

    def test_stopping_records_the_reason(self, store, investigation) -> None:
        store.investigations.set_status(
            investigation.id,
            InvestigationStatus.COMPLETED,
            stop_reason=StopReason.DIMINISHING_RETURNS,
            stop_detail="new searches returned only material already held",
        )
        loaded = store.investigations.get(investigation.id)
        assert loaded.status is InvestigationStatus.COMPLETED
        assert loaded.stop_reason is StopReason.DIMINISHING_RETURNS
        assert loaded.completed_at is not None
        assert not loaded.is_open
