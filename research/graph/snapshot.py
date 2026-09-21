"""What changed since last time.

Every tool in the surveyed field answers a question once. Most of them keep
nothing between runs, so "ask this again and tell me what is different" is
not a feature they have withheld - it is one they cannot build without first
building a store. This system has the store, so it can.

A snapshot is what an investigation's claims and evidence amounted to at a
moment: derived, never authoritative, and stored whole so that a later
comparison is with what was actually recorded rather than with what the
current code would compute from the same rows.

The diff is deliberately about *claims*, not about documents. "Forty-one new
documents" is activity; "claim:3 was supported by two independent sources
and is now contradicted by a newer one, and the paper behind it has been
retracted" is the answer to the question somebody asked. Those are the
changes worth waking somebody for:

* a claim's status changed,
* a claim gained or lost independent support,
* a claim acquired counterevidence,
* evidence behind a claim was retracted,
* newer contradicting evidence appeared than anything supporting it.

The last two are why this is worth having at all. A retraction happens after
a run is finished and nothing about the original run will ever notice it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.graph.claims import ClaimGraph
from research.storage.store import ResearchStore

SNAPSHOT_VERSION = 1


@dataclass(slots=True)
class ClaimState:
    """One claim's evidentiary situation, reduced to what a diff needs."""

    claim_id: str
    text: str
    status: str
    independent_support: int
    independent_contradiction: int
    has_primary_source: bool
    retracted_ids: list[str] = field(default_factory=list)
    possibly_superseded: bool = False
    gaps: int = 0
    #: Independence keys, not document ids: gaining a fourth copy of a story
    #: already cited is not new support, and a diff should not say it is.
    support_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "status": self.status,
            "independent_support": self.independent_support,
            "independent_contradiction": self.independent_contradiction,
            "has_primary_source": self.has_primary_source,
            "retracted_ids": list(self.retracted_ids),
            "possibly_superseded": self.possibly_superseded,
            "gaps": self.gaps,
            "support_sources": list(self.support_sources),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimState":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass(slots=True)
class Snapshot:
    investigation_id: str
    question: str = ""
    stop_reason: str = ""
    documents: int = 0
    independent_sources: int = 0
    claims: dict[str, ClaimState] = field(default_factory=dict)
    version: int = SNAPSHOT_VERSION
    #: Set when read back from the store.
    snapshot_id: str | None = None
    label: str | None = None
    taken_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "investigation_id": self.investigation_id,
            "question": self.question,
            "stop_reason": self.stop_reason,
            "documents": self.documents,
            "independent_sources": self.independent_sources,
            "claims": {key: state.to_dict() for key, state in self.claims.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Snapshot":
        return cls(
            investigation_id=str(data.get("investigation_id") or ""),
            question=str(data.get("question") or ""),
            stop_reason=str(data.get("stop_reason") or ""),
            documents=int(data.get("documents") or 0),
            independent_sources=int(data.get("independent_sources") or 0),
            claims={
                key: ClaimState.from_dict(value)
                for key, value in (data.get("claims") or {}).items()
            },
            version=int(data.get("version") or SNAPSHOT_VERSION),
            snapshot_id=data.get("snapshot_id"),
            label=data.get("label"),
            taken_at=data.get("taken_at"),
        )


def take(store: ResearchStore, investigation_id: str) -> Snapshot:
    """Read the investigation's current state into a snapshot."""
    investigation = store.investigations.get(investigation_id)
    documents = store.documents.list(investigation_id, limit=5000)
    graph = ClaimGraph(store, investigation_id)

    claims: dict[str, ClaimState] = {}
    for assessment in graph.assess_all():
        support = assessment.support
        claims[assessment.claim_id] = ClaimState(
            claim_id=assessment.claim_id,
            text=assessment.text,
            status=str(assessment.status),
            independent_support=support.independent_count,
            independent_contradiction=assessment.contradiction.independent_count,
            has_primary_source=support.has_primary_source,
            retracted_ids=sorted(
                set(support.retracted_ids) | set(assessment.contradiction.retracted_ids)
            ),
            possibly_superseded=assessment.possibly_superseded,
            gaps=len(assessment.gaps),
            support_sources=sorted(
                group[0] for group in support.independent_groups if group
            ),
        )

    return Snapshot(
        investigation_id=investigation_id,
        question=investigation.question,
        stop_reason=str(investigation.stop_reason or ""),
        documents=len(documents),
        independent_sources=len(
            {document.independence_key or document.id for document in documents}
        ),
        claims=claims,
    )


def save(
    store: ResearchStore, investigation_id: str, *, label: str | None = None
) -> tuple[str, Snapshot]:
    """Take a snapshot and store it."""
    snapshot = take(store, investigation_id)
    snapshot_id = store.snapshots.save(investigation_id, snapshot.to_dict(), label=label)
    snapshot.snapshot_id = snapshot_id
    return snapshot_id, snapshot


def load(store: ResearchStore, snapshot_id: str) -> Snapshot:
    return Snapshot.from_dict(store.snapshots.get(snapshot_id))


def latest(store: ResearchStore, investigation_id: str) -> Snapshot | None:
    payload = store.snapshots.latest(investigation_id)
    return Snapshot.from_dict(payload) if payload else None


# -- the diff ------------------------------------------------------------
@dataclass(slots=True)
class ClaimChange:
    claim_id: str
    text: str
    #: Short phrases, each one thing that changed about this claim.
    changes: list[str] = field(default_factory=list)
    #: True when this change ought to change somebody's mind about the claim,
    #: rather than merely adding to it.
    material: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "changes": list(self.changes),
            "material": self.material,
        }


@dataclass(slots=True)
class InvestigationDiff:
    investigation_id: str
    before_taken_at: str | None = None
    after_taken_at: str | None = None
    new_claims: list[ClaimChange] = field(default_factory=list)
    changed_claims: list[ClaimChange] = field(default_factory=list)
    dropped_claims: list[str] = field(default_factory=list)
    new_documents: int = 0
    new_independent_sources: int = 0
    stop_reason_changed: tuple[str, str] | None = None

    @property
    def material_changes(self) -> list[ClaimChange]:
        return [change for change in self.changed_claims if change.material]

    @property
    def anything_changed(self) -> bool:
        return bool(
            self.new_claims
            or self.changed_claims
            or self.dropped_claims
            or self.new_documents
            or self.stop_reason_changed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "investigation": self.investigation_id,
            "before": self.before_taken_at,
            "after": self.after_taken_at,
            "new_claims": [change.to_dict() for change in self.new_claims],
            "changed_claims": [change.to_dict() for change in self.changed_claims],
            "dropped_claims": list(self.dropped_claims),
            "new_documents": self.new_documents,
            "new_independent_sources": self.new_independent_sources,
            **(
                {"stop_reason": {"before": self.stop_reason_changed[0],
                                 "after": self.stop_reason_changed[1]}}
                if self.stop_reason_changed
                else {}
            ),
        }


def diff(before: Snapshot, after: Snapshot) -> InvestigationDiff:
    """What changed between two snapshots of one investigation."""
    result = InvestigationDiff(
        investigation_id=after.investigation_id or before.investigation_id,
        before_taken_at=before.taken_at,
        after_taken_at=after.taken_at,
        new_documents=max(0, after.documents - before.documents),
        new_independent_sources=max(
            0, after.independent_sources - before.independent_sources
        ),
    )
    if before.stop_reason != after.stop_reason:
        result.stop_reason_changed = (before.stop_reason, after.stop_reason)

    for claim_id, state in after.claims.items():
        previous = before.claims.get(claim_id)
        if previous is None:
            result.new_claims.append(
                ClaimChange(
                    claim_id=claim_id,
                    text=state.text,
                    changes=[_describe_new(state)],
                    material=state.independent_support > 0
                    or state.independent_contradiction > 0,
                )
            )
            continue
        change = _compare(previous, state)
        if change is not None:
            result.changed_claims.append(change)

    result.dropped_claims = sorted(set(before.claims) - set(after.claims))
    return result


def _describe_new(state: ClaimState) -> str:
    if state.independent_support or state.independent_contradiction:
        return (
            f"new claim, {state.status}: {state.independent_support} independent "
            f"supporting, {state.independent_contradiction} contradicting"
        )
    return f"new claim, {state.status}, no evidence attached yet"


def _compare(before: ClaimState, after: ClaimState) -> ClaimChange | None:
    """Everything that changed about one claim, worst news first."""
    changes: list[str] = []
    material = False

    # A retraction is the change nothing about the original run could have
    # noticed, and the one most likely to invalidate a conclusion.
    newly_retracted = sorted(set(after.retracted_ids) - set(before.retracted_ids))
    if newly_retracted:
        changes.append(
            "evidence retracted since the last run: " + ", ".join(newly_retracted)
        )
        material = True

    if before.status != after.status:
        changes.append(f"status {before.status} -> {after.status}")
        material = True

    if after.possibly_superseded and not before.possibly_superseded:
        changes.append(
            "contradicting evidence is now newer than anything supporting it"
        )
        material = True

    gained = sorted(set(after.support_sources) - set(before.support_sources))
    lost = sorted(set(before.support_sources) - set(after.support_sources))
    if gained:
        changes.append(
            f"independent support {before.independent_support} -> "
            f"{after.independent_support} (+{len(gained)} sources)"
        )
        # Reaching two independent sources is the threshold the assessment
        # itself cares about; going from one to two is a different kind of
        # event from going from four to five.
        material = material or (before.independent_support < 2 <= after.independent_support)
    if lost:
        changes.append(f"lost {len(lost)} supporting source(s): {', '.join(lost)}")
        material = True

    if after.independent_contradiction > before.independent_contradiction:
        changes.append(
            f"counterevidence {before.independent_contradiction} -> "
            f"{after.independent_contradiction} independent sources"
        )
        material = True

    if after.has_primary_source and not before.has_primary_source:
        changes.append("now reaches a primary record")

    if after.gaps < before.gaps:
        changes.append(f"{before.gaps - after.gaps} open question(s) closed")
    elif after.gaps > before.gaps:
        changes.append(f"{after.gaps - before.gaps} new open question(s)")

    if not changes:
        return None
    return ClaimChange(
        claim_id=after.claim_id, text=after.text, changes=changes, material=material
    )


def describe(result: InvestigationDiff) -> str:
    """The diff as something a person reads."""
    if not result.anything_changed:
        return "Nothing changed: no new evidence, and every claim stands where it did."

    lines: list[str] = []
    material = result.material_changes
    if material:
        lines.append("Changes that bear on the conclusions")
        for change in material:
            lines.append(f"  {change.claim_id}  {change.text}")
            for detail in change.changes:
                lines.append(f"      {detail}")
        lines.append("")

    other = [change for change in result.changed_claims if not change.material]
    if other:
        lines.append("Other changes")
        for change in other:
            lines.append(f"  {change.claim_id}  {'; '.join(change.changes)}")
        lines.append("")

    if result.new_claims:
        lines.append("New claims")
        for change in result.new_claims:
            lines.append(f"  {change.claim_id}  {change.text}")
            lines.append(f"      {change.changes[0]}")
        lines.append("")

    if result.dropped_claims:
        lines.append("No longer present: " + ", ".join(result.dropped_claims))

    tail = (
        f"{result.new_documents} new documents, "
        f"{result.new_independent_sources} new independent sources"
    )
    if result.stop_reason_changed:
        tail += (
            f"; stopped {result.stop_reason_changed[1]} "
            f"(was {result.stop_reason_changed[0] or 'still running'})"
        )
    lines.append(tail)
    return "\n".join(lines)
