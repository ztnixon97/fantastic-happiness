"""Assembling the final report from stored state.

The report is built from the record, not from a conversation: claims and
their assessments, evidence and its provenance, the timeline, the open
questions, and the reason the investigation stopped. A model may write the
prose; it cannot introduce a fact, because every section is assembled here
from identifiers that already exist.

That is what makes the result inspectable. A reader who doubts a sentence can
follow its claim id to the evidence, and each evidence id to the provider call
that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.graph.claims import ClaimAssessment, ClaimGraph
from research.graph.independence import independent_documents
from research.graph.timelines import Timeline, TimelineEntry
from research.models.claim import ClaimStatus, EvidenceStance
from research.models.evidence import EvidenceDocument
from research.models.investigation import Investigation
from research.operations.claims import ClaimOperations
from research.storage.store import ResearchStore

#: Statuses worth a heading of their own in the findings section.
_ORDER = {
    ClaimStatus.SUPPORTED: 0,
    ClaimStatus.MIXED: 1,
    ClaimStatus.CONTRADICTED: 2,
    ClaimStatus.INSUFFICIENT_EVIDENCE: 3,
    ClaimStatus.UNVERIFIED: 4,
}


@dataclass(slots=True)
class ReportSource:
    """One independent source, with the copies it stood in for."""

    document: EvidenceDocument
    copies: list[EvidenceDocument] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return self.document.citation_label()


@dataclass(slots=True)
class ReportData:
    """Everything a report is made of, before any prose is written."""

    investigation: Investigation
    assessments: list[ClaimAssessment] = field(default_factory=list)
    sources: list[ReportSource] = field(default_factory=list)
    timeline: list[TimelineEntry] = field(default_factory=list)
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def findings(self) -> list[ClaimAssessment]:
        return sorted(
            self.assessments,
            key=lambda assessment: (
                _ORDER.get(assessment.status, 9),
                -assessment.support.independent_count,
                assessment.claim_id,
            ),
        )

    def contested(self) -> list[ClaimAssessment]:
        return [
            assessment
            for assessment in self.assessments
            if assessment.contradiction.count or assessment.possibly_superseded
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "investigation": self.investigation.id,
            "question": self.investigation.question,
            "status": str(self.investigation.status),
            "stop_reason": str(self.investigation.stop_reason)
            if self.investigation.stop_reason
            else None,
            "stop_detail": self.investigation.stop_detail,
            "counts": dict(self.counts),
            "findings": [assessment.to_dict() for assessment in self.findings()],
            "timeline": [entry.to_dict() for entry in self.timeline],
            "open_questions": list(self.open_questions),
            "sources": [
                {
                    "id": source.document.id,
                    "citation": source.citation,
                    "type": str(source.document.source_type),
                    "url": source.document.canonical_url,
                    "doi": source.document.doi,
                    "published": source.document.published_at.date().isoformat()
                    if source.document.published_at
                    else None,
                    "provenance": source.document.provenance.describe()
                    if source.document.provenance
                    else None,
                    "copies": [copy.id for copy in source.copies],
                }
                for source in self.sources
            ],
        }


def collect(store: ResearchStore, investigation_id: str) -> ReportData:
    """Gather the state a report is written from."""
    investigation = store.investigations.get(investigation_id)
    graph = ClaimGraph(store, investigation_id)
    operations = ClaimOperations(store, investigation_id=investigation_id)
    documents = store.documents.list(investigation_id, limit=2000)
    by_id = {document.id: document for document in documents}

    sources: list[ReportSource] = []
    for group in independent_documents(store.documents, [d.id for d in documents]):
        members = [by_id[document_id] for document_id in group if document_id in by_id]
        if not members:
            continue
        # The original is the one the others are copies of.
        head = next((member for member in members if not member.duplicate_of), members[0])
        sources.append(
            ReportSource(
                document=head,
                copies=[member for member in members if member.id != head.id],
            )
        )

    data = ReportData(
        investigation=investigation,
        assessments=graph.assess_all(limit=200),
        sources=sources,
        timeline=Timeline(store, investigation_id).build(include_publications=True, limit=60),
        open_questions=operations.open_questions(limit=50),
        counts={
            "documents": len(documents),
            "independent_sources": len(sources),
            "claims": len(store.claims.list(investigation_id, hydrate=False)),
            "tasks": store.tasks.count(investigation_id),
            "searches": store.queries.count(investigation_id),
            "failed_fetches": store.fetches.failure_count(investigation_id),
        },
    )
    return data


def render_markdown(data: ReportData, store: ResearchStore, *, summary: str | None = None) -> str:
    """Render the report.

    ``summary`` is the one place model prose may appear, and it is labelled as
    such. Everything else is assembled from stored records.
    """
    investigation = data.investigation
    lines: list[str] = [
        f"# {investigation.question}",
        "",
        f"*{investigation.id} | {data.counts['documents']} documents from "
        f"{data.counts['independent_sources']} independent sources | "
        f"{data.counts['claims']} claims | {data.counts['tasks']} research tasks*",
        "",
        "## Executive summary",
        "",
    ]
    if summary:
        lines.extend([summary.strip(), ""])
    else:
        lines.extend([_default_summary(data), ""])

    lines.extend(["## Key findings", ""])
    findings = data.findings()
    if not findings:
        lines.append("No claims were recorded, so there are no findings to report.")
        lines.append("")
    for assessment in findings:
        lines.append(f"### {assessment.text}")
        lines.append("")
        lines.append(f"**{assessment.status}** — {assessment.explanation}")
        lines.append("")
        lines.extend(_evidence_lines(assessment, store))
        lines.append("")

    contested = data.contested()
    lines.extend(["## Contradictory and qualifying evidence", ""])
    if not contested:
        lines.extend(
            [
                "No contradicting evidence was linked to any claim. That is a gap in the "
                "research, not a confirmation: absence of recorded counterevidence means "
                "nobody has yet found any, and the skeptic pass may not have run.",
                "",
            ]
        )
    for assessment in contested:
        lines.append(f"- **{assessment.text}** — {assessment.explanation}")
        for document_id in assessment.contradiction.document_ids:
            document = store.documents.get(document_id)
            lines.append(f"  - {document_id}: {document.citation_label()}")
        if assessment.possibly_superseded:
            lines.append(
                "  - the contradicting evidence is more recent than any supporting evidence"
            )
    lines.append("")

    dated = [entry for entry in data.timeline if entry.date]
    if dated:
        lines.extend(["## Timeline", ""])
        for entry in dated:
            date = entry.date.date().isoformat()
            kind = "" if entry.kind == "event" else " *(publication)*"
            evidence = ", ".join(entry.evidence_ids[:4])
            lines.append(f"- **{date}**{kind} {entry.description} [{evidence}]")
        lines.append("")

    lines.extend(["## Areas of uncertainty", ""])
    if data.open_questions:
        for question in data.open_questions:
            lines.append(f"- **{question['claim_id']}** {question['text']}")
            for gap in question["gaps"]:
                lines.append(f"  - {gap}")
    else:
        lines.append("Every recorded claim has evidence with no outstanding gaps noted.")
    lines.append("")

    lines.extend(["## How this investigation ended", ""])
    lines.append(
        f"- stopped because: **{investigation.stop_reason or 'not recorded'}**"
        + (f" — {investigation.stop_detail}" if investigation.stop_detail else "")
    )
    lines.append(
        f"- work done: {data.counts['tasks']} tasks, {data.counts['searches']} searches, "
        f"{data.counts['failed_fetches']} failed retrievals"
    )
    lines.append("")

    lines.extend(["## Sources", "", _sources_note(data), ""])
    for index, source in enumerate(data.sources, start=1):
        document = source.document
        bits = [f"{index}. **{document.id}** {document.citation_label()}"]
        if document.title:
            bits.append(f" — {document.title}")
        lines.append("".join(bits))
        locator = document.doi and f"doi:{document.doi}" or document.canonical_url
        if locator:
            lines.append(f"   - {locator}")
        lines.append(f"   - type: {document.source_type}")
        if document.provenance:
            lines.append(f"   - retrieved: {document.provenance.describe()}")
        if source.copies:
            lines.append(
                f"   - also seen as {len(source.copies)} cop"
                f"{'y' if len(source.copies) == 1 else 'ies'}: "
                + ", ".join(copy.id for copy in source.copies)
                + " (counted once)"
            )
    lines.append("")
    return "\n".join(lines)


def _evidence_lines(assessment: ClaimAssessment, store: ResearchStore) -> list[str]:
    lines: list[str] = []
    links = store.claims.evidence_links(assessment.claim_id)
    for stance, label in (
        (EvidenceStance.SUPPORTS, "Supporting"),
        (EvidenceStance.CONTRADICTS, "Contradicting"),
        (EvidenceStance.QUALIFIES, "Qualifying"),
    ):
        relevant = [link for link in links if link.stance is stance]
        if not relevant:
            continue
        lines.append(f"{label} evidence:")
        for link in relevant:
            document = store.documents.get(link.document_id)
            marker = ""
            if document.duplicate_of or document.derived_from:
                parent = document.duplicate_of or document.derived_from
                marker = f" *(not independent of {parent})*"
            lines.append(f"- `{document.id}` {document.citation_label()}{marker}")
            if link.excerpt:
                lines.append(f'  > "{link.excerpt}"')
            if link.analysis:
                lines.append(f"  - *analysis:* {link.analysis}")
        lines.append("")
    if assessment.gaps:
        lines.append("What would strengthen this:")
        for gap in assessment.gaps:
            lines.append(f"- {gap}")
    return lines


def _default_summary(data: ReportData) -> str:
    """A factual summary when no prose has been supplied.

    Deliberately dull: it states what was found and what was not, and draws no
    conclusion the evidence has not been recorded as supporting.
    """
    counts: dict[ClaimStatus, int] = {}
    for assessment in data.assessments:
        counts[assessment.status] = counts.get(assessment.status, 0) + 1
    parts = [
        f"{data.counts['claims']} claims were recorded across "
        f"{data.counts['independent_sources']} independent sources "
        f"({data.counts['documents']} documents once copies are included)."
    ]
    if counts:
        parts.append(
            "By status: "
            + ", ".join(
                f"{count} {status}"
                for status, count in sorted(counts.items(), key=lambda item: str(item[0]))
            )
            + "."
        )
    if data.open_questions:
        parts.append(
            f"{len(data.open_questions)} claims still have gaps that further work could close."
        )
    parts.append(
        "This summary is generated from the record; no conclusion is drawn here that "
        "the evidence below does not carry."
    )
    return " ".join(parts)


def _sources_note(data: ReportData) -> str:
    copies = sum(len(source.copies) for source in data.sources)
    if not copies:
        return (
            f"{len(data.sources)} sources, each counted once. Every entry carries the "
            "provider call that produced it."
        )
    return (
        f"{len(data.sources)} independent sources. A further {copies} document"
        f"{'s' if copies != 1 else ''} were retrieved but are copies, syndicated versions "
        "or rewrites of these, and are not counted as separate corroboration."
    )
