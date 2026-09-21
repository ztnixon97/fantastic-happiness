"""Snapshots, and what changed since the last one.

The behaviour under test is the one nothing else in the surveyed field can
offer: ask the same question again and be told what is different about the
answer, rather than being handed a fresh report to re-read.
"""

from __future__ import annotations

import pytest

from research.errors import NotFound
from research.graph import snapshot as snapshots
from research.graph.snapshot import ClaimState, Snapshot, describe, diff
from research.models.claim import ClaimEvidenceLink, EvidenceStance
from research.models.common import (
    Provenance,
    SourceFamily,
    SourceType,
    utcnow,
)
from research.models.evidence import EvidenceDocument
from research.storage.store import ResearchStore


def add_document(store: ResearchStore, investigation_id: str, **overrides):
    number = store.documents.count(investigation_id) + 1
    defaults = dict(
        id=store.documents.new_id(),
        investigation_id=investigation_id,
        source_type=SourceType.ACADEMIC_PEER_REVIEWED,
        source_family=SourceFamily.ACADEMIC,
        provider="test",
        external_id=f"W{number}",
        canonical_url=f"https://example.org/{number}",
        title=f"Paper {number}",
        published_at=utcnow(),
        content_hash=f"sha256:{number}",
        provenance=Provenance(provider="test"),
    )
    defaults.update(overrides)
    return store.documents.add(EvidenceDocument(**defaults))


def state(claim_id: str = "claim:1", **overrides) -> ClaimState:
    defaults = dict(
        claim_id=claim_id,
        text="A proposition",
        status="supported",
        independent_support=2,
        independent_contradiction=0,
        has_primary_source=False,
    )
    defaults.update(overrides)
    return ClaimState(**defaults)


def snapshot(**overrides) -> Snapshot:
    defaults = dict(investigation_id="investigation:1", documents=10, independent_sources=8)
    defaults.update(overrides)
    return Snapshot(**defaults)


class TestTakingASnapshot:
    def test_it_records_the_claims_and_their_evidence(
        self, store: ResearchStore, investigation
    ) -> None:
        document = add_document(store, investigation.id)
        claim = store.claims.create(investigation.id, "Costs rose")
        store.claims.link_evidence(
            ClaimEvidenceLink(
                claim_id=claim.id, document_id=document.id, stance=EvidenceStance.SUPPORTS
            )
        )
        taken = snapshots.take(store, investigation.id)

        assert taken.documents == 1
        assert taken.independent_sources == 1
        assert taken.claims[claim.id].text == "Costs rose"
        assert taken.claims[claim.id].independent_support == 1

    def test_support_is_counted_in_sources_not_documents(
        self, store: ResearchStore, investigation
    ) -> None:
        """A fourth copy of a story already cited is not new support."""
        original = add_document(store, investigation.id, independence_key="wire:one")
        copy = add_document(store, investigation.id, independence_key="wire:one")
        claim = store.claims.create(investigation.id, "It happened")
        for document in (original, copy):
            store.claims.link_evidence(
                ClaimEvidenceLink(
                    claim_id=claim.id,
                    document_id=document.id,
                    stance=EvidenceStance.SUPPORTS,
                )
            )
        taken = snapshots.take(store, investigation.id)
        assert taken.claims[claim.id].independent_support == 1
        assert len(taken.claims[claim.id].support_sources) == 1

    def test_it_round_trips_through_the_store(
        self, store: ResearchStore, investigation
    ) -> None:
        add_document(store, investigation.id)
        store.claims.create(investigation.id, "Something")
        snapshot_id, taken = snapshots.save(store, investigation.id, label="before")

        loaded = snapshots.load(store, snapshot_id)
        assert loaded.to_dict() == taken.to_dict()
        assert loaded.label == "before"
        assert loaded.taken_at

    def test_the_latest_is_the_most_recent(
        self, store: ResearchStore, investigation
    ) -> None:
        first, _ = snapshots.save(store, investigation.id, label="one")
        second, _ = snapshots.save(store, investigation.id, label="two")
        assert first != second
        assert snapshots.latest(store, investigation.id).label == "two"

    def test_no_snapshot_is_none_rather_than_an_error(
        self, store: ResearchStore, investigation
    ) -> None:
        assert snapshots.latest(store, investigation.id) is None

    def test_an_unknown_snapshot_is_not_found(self, store: ResearchStore) -> None:
        with pytest.raises(NotFound):
            snapshots.load(store, "snapshot:404")


class TestWhatChanged:
    def test_nothing_changing_is_reported_as_nothing(self) -> None:
        before = snapshot(claims={"claim:1": state()})
        after = snapshot(claims={"claim:1": state()})
        result = diff(before, after)
        assert not result.anything_changed
        assert "Nothing changed" in describe(result)

    def test_a_retraction_is_material_and_comes_first(self) -> None:
        """The change nothing about the original run could ever have noticed."""
        before = snapshot(claims={"claim:1": state()})
        after = snapshot(claims={"claim:1": state(retracted_ids=["evidence:4"])})

        change = diff(before, after).changed_claims[0]
        assert change.material
        assert "retracted" in change.changes[0]
        assert "evidence:4" in change.changes[0]

    def test_a_status_change_is_material(self) -> None:
        before = snapshot(claims={"claim:1": state(status="supported")})
        after = snapshot(claims={"claim:1": state(status="contradicted")})
        change = diff(before, after).changed_claims[0]
        assert change.material
        assert "supported -> contradicted" in " ".join(change.changes)

    def test_new_counterevidence_is_material(self) -> None:
        before = snapshot(claims={"claim:1": state(independent_contradiction=0)})
        after = snapshot(claims={"claim:1": state(independent_contradiction=2)})
        change = diff(before, after).changed_claims[0]
        assert change.material
        assert "counterevidence 0 -> 2" in " ".join(change.changes)

    def test_being_superseded_is_material(self) -> None:
        before = snapshot(claims={"claim:1": state()})
        after = snapshot(claims={"claim:1": state(possibly_superseded=True)})
        change = diff(before, after).changed_claims[0]
        assert change.material
        assert "newer than anything supporting" in " ".join(change.changes)

    def test_reaching_two_independent_sources_is_material(self) -> None:
        """One to two crosses the threshold the assessment itself cares about."""
        before = snapshot(claims={"claim:1": state(
            independent_support=1, support_sources=["evidence:1"]
        )})
        after = snapshot(claims={"claim:1": state(
            independent_support=2, support_sources=["evidence:1", "evidence:9"]
        )})
        assert diff(before, after).changed_claims[0].material

    def test_a_fifth_supporting_source_is_not_material(self) -> None:
        """More of what you already had does not change anybody's mind."""
        before = snapshot(claims={"claim:1": state(
            independent_support=4, support_sources=[f"evidence:{n}" for n in range(4)]
        )})
        after = snapshot(claims={"claim:1": state(
            independent_support=5, support_sources=[f"evidence:{n}" for n in range(5)]
        )})
        change = diff(before, after).changed_claims[0]
        assert not change.material
        assert "independent support 4 -> 5" in " ".join(change.changes)

    def test_losing_support_is_material(self) -> None:
        before = snapshot(claims={"claim:1": state(
            independent_support=2, support_sources=["evidence:1", "evidence:2"]
        )})
        after = snapshot(claims={"claim:1": state(
            independent_support=1, support_sources=["evidence:1"]
        )})
        change = diff(before, after).changed_claims[0]
        assert change.material
        assert "lost 1 supporting source" in " ".join(change.changes)

    def test_a_new_claim_is_reported_separately(self) -> None:
        before = snapshot(claims={"claim:1": state()})
        after = snapshot(claims={"claim:1": state(), "claim:2": state("claim:2")})
        result = diff(before, after)
        assert [change.claim_id for change in result.new_claims] == ["claim:2"]
        assert result.changed_claims == []

    def test_a_claim_that_disappeared_is_named(self) -> None:
        before = snapshot(claims={"claim:1": state(), "claim:2": state("claim:2")})
        after = snapshot(claims={"claim:1": state()})
        assert diff(before, after).dropped_claims == ["claim:2"]

    def test_evidence_counts_never_go_negative(self) -> None:
        """A smaller corpus is not minus-five new documents."""
        result = diff(snapshot(documents=20), snapshot(documents=3))
        assert result.new_documents == 0

    def test_the_stopping_reason_is_reported_when_it_changes(self) -> None:
        before = snapshot(stop_reason="")
        after = snapshot(stop_reason="evidence_sufficient")
        result = diff(before, after)
        assert result.stop_reason_changed == ("", "evidence_sufficient")
        assert "evidence_sufficient" in describe(result)

    def test_material_changes_lead_the_description(self) -> None:
        before = snapshot(claims={
            "claim:1": state(),
            "claim:2": state("claim:2", independent_support=4,
                             support_sources=[f"evidence:{n}" for n in range(4)]),
        })
        after = snapshot(claims={
            "claim:1": state(status="contradicted"),
            "claim:2": state("claim:2", independent_support=5,
                             support_sources=[f"evidence:{n}" for n in range(5)]),
        })
        rendered = describe(diff(before, after))
        assert rendered.index("bear on the conclusions") < rendered.index("Other changes")


class TestAcrossARun:
    """The whole point, end to end: the world moves and the claim changes."""

    async def test_a_retraction_between_runs_is_surfaced(
        self, store: ResearchStore, investigation
    ) -> None:
        supporting = add_document(store, investigation.id, title="The original finding")
        claim = store.claims.create(investigation.id, "The finding holds")
        store.claims.link_evidence(
            ClaimEvidenceLink(
                claim_id=claim.id,
                document_id=supporting.id,
                stance=EvidenceStance.SUPPORTS,
            )
        )
        _, before = snapshots.save(store, investigation.id, label="first run")

        # Time passes; the paper is retracted.
        supporting.metadata["is_retracted"] = True
        store.documents.update(supporting)

        result = diff(before, snapshots.take(store, investigation.id))
        assert result.material_changes
        assert supporting.id in " ".join(result.material_changes[0].changes)

    async def test_counterevidence_arriving_between_runs_is_surfaced(
        self, store: ResearchStore, investigation
    ) -> None:
        supporting = add_document(store, investigation.id)
        claim = store.claims.create(investigation.id, "The finding holds")
        store.claims.link_evidence(
            ClaimEvidenceLink(
                claim_id=claim.id, document_id=supporting.id, stance=EvidenceStance.SUPPORTS
            )
        )
        _, before = snapshots.save(store, investigation.id)

        against = add_document(store, investigation.id, title="A later result")
        store.claims.link_evidence(
            ClaimEvidenceLink(
                claim_id=claim.id, document_id=against.id, stance=EvidenceStance.CONTRADICTS
            )
        )

        result = diff(before, snapshots.take(store, investigation.id))
        assert result.material_changes
        assert result.new_documents == 1

    async def test_a_second_run_finding_only_copies_changes_no_claim(
        self, store: ResearchStore, investigation
    ) -> None:
        """New documents are activity; new sources are news."""
        original = add_document(store, investigation.id, independence_key="wire:one")
        claim = store.claims.create(investigation.id, "It happened")
        store.claims.link_evidence(
            ClaimEvidenceLink(
                claim_id=claim.id, document_id=original.id, stance=EvidenceStance.SUPPORTS
            )
        )
        _, before = snapshots.save(store, investigation.id)

        copy = add_document(store, investigation.id, independence_key="wire:one")
        store.claims.link_evidence(
            ClaimEvidenceLink(
                claim_id=claim.id, document_id=copy.id, stance=EvidenceStance.SUPPORTS
            )
        )

        result = diff(before, snapshots.take(store, investigation.id))
        assert result.new_documents == 1
        assert result.material_changes == []
