"""Plain-text rendering for the CLI.

Every view answers the same question in a different shape: where did this
come from, and is it independent of everything else here.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from research.models.evidence import EvidenceDocument
from research.normalize.text import truncate


def as_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str, ensure_ascii=False)


def table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "(nothing to show)"
    widths = {
        column: max(len(column), *(len(_cell(row.get(column))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.upper().ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    body = [
        "  ".join(_cell(row.get(column)).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "-"
    return str(value)


def document_row(document: EvidenceDocument) -> dict[str, Any]:
    relation = ""
    if document.duplicate_relation:
        parent = document.duplicate_of or document.derived_from or "?"
        relation = f"{document.duplicate_relation} of {parent}"
    return {
        "id": document.id,
        "type": str(document.source_type),
        "published": document.published_at.date().isoformat() if document.published_at else "-",
        "title": truncate(document.title or "(untitled)", 62),
        "source": document.publisher or document.canonical_host or document.provider,
        "independence": relation or "independent",
    }


def document_detail(
    document: EvidenceDocument,
    *,
    duplicates: Iterable[EvidenceDocument] = (),
    references: Sequence[dict[str, Any]] = (),
    citing: Sequence[dict[str, Any]] = (),
    claims: Sequence[dict[str, Any]] = (),
    excerpt_characters: int = 1200,
) -> str:
    lines = [
        f"{document.id}  {document.title or '(untitled)'}",
        "=" * 78,
        f"type            {document.source_type} "
        f"(distance to primary record: {document.primary_source_distance})",
        f"family          {document.source_family}",
        f"provider        {document.provider}"
        + (f"  external id {document.external_id}" if document.external_id else ""),
        f"url             {document.canonical_url or '-'}",
        f"doi             {document.doi or '-'}",
        f"authors         {', '.join(document.authors) if document.authors else '-'}",
        f"published       {document.published_at.isoformat() if document.published_at else '-'}",
        f"retrieved       {document.fetched_at.isoformat() if document.fetched_at else '-'}",
        f"publisher       {document.publisher or '-'}",
        f"content hash    {document.content_hash}",
    ]
    if document.provenance:
        lines.append(f"provenance      {document.provenance.describe()}")
        if document.provenance.query_id:
            lines.append(f"  from query    {document.provenance.query_id}")
        if document.provenance.fetch_id:
            lines.append(f"  from fetch    {document.provenance.fetch_id}")

    if document.duplicate_of or document.derived_from:
        parent = document.duplicate_of or document.derived_from
        lines.append(
            f"independence    NOT independent: {document.duplicate_relation} of {parent}"
        )
        reason = document.metadata.get("duplicate_reason")
        if reason:
            lines.append(f"  reason        {reason}")
    else:
        lines.append(f"independence    independent (group {document.independence_key})")

    copies = list(duplicates)
    if copies:
        lines.append(f"copies held     {len(copies)}")
        for copy in copies[:10]:
            lines.append(
                f"  {copy.id:<12} {copy.duplicate_relation} "
                f"via {copy.canonical_host or copy.provider}"
            )
    if document.metadata.get("wire_service"):
        lines.append(f"wire service    {document.metadata['wire_service']}")
    if document.metadata.get("is_retracted"):
        lines.append("RETRACTED       provider reports this work as retracted")
    if document.metadata.get("truncated"):
        lines.append("truncated       body was cut at the response size limit")

    if references:
        lines.append(f"references      {len(references)} outbound citation edges")
    if citing:
        lines.append(f"cited by        {len(citing)} documents in this investigation")
    if claims:
        lines.append("claims")
        for link in claims:
            lines.append(f"  {link['claim_id']:<10} {link['stance']}")

    body = document.text or document.abstract or ""
    if body:
        lines.extend(
            [
                "",
                "-- content (external, untrusted) " + "-" * 45,
                truncate(body, excerpt_characters),
            ]
        )
    return "\n".join(lines)


def budget_report(snapshot: dict[str, dict[str, float]]) -> str:
    rows = [
        {
            "resource": resource,
            "used": _number(values["used"]),
            "limit": _number(values["limit"]),
            "remaining": _number(values["remaining"]),
            "spent": (
                f"{(values['used'] / values['limit'] * 100):.0f}%"
                if values["limit"] and values["limit"] != float("inf")
                else "-"
            ),
        }
        for resource, values in snapshot.items()
    ]
    return table(rows, ["resource", "used", "limit", "remaining", "spent"])


def _number(value: float) -> str:
    value = float(value)
    if value == float("inf"):
        return "-"
    if value and abs(value) < 0.01:
        # Money: a tenth of a cent is still a number, and rounding it to
        # 0.00 would read as free.
        return f"{value:.4f}"
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


def claim_detail(assessment: Any, *, links: Sequence[Any] = ()) -> str:
    """Render a claim with the evidence behind it, stance by stance."""
    lines = [
        f"{assessment.claim_id}  {assessment.text}",
        "=" * 78,
        f"status          {assessment.status}",
        f"assessment      {assessment.explanation}",
        f"independent     {assessment.support.independent_count} supporting, "
        f"{assessment.contradiction.independent_count} contradicting",
    ]
    if assessment.possibly_superseded:
        lines.append(
            "superseded?     contradicting evidence is more recent than the support"
        )
    for label, summary in (
        ("supporting", assessment.support),
        ("contradicting", assessment.contradiction),
        ("qualifying", assessment.qualification),
    ):
        if not summary.count:
            continue
        lines.append("")
        lines.append(f"-- {label} evidence " + "-" * (58 - len(label)))
        for group in summary.independent_groups:
            head, *copies = group
            suffix = f"  (+{len(copies)} non-independent cop{'y' if len(copies) == 1 else 'ies'}: "
            suffix += ", ".join(copies) + ")" if copies else ""
            lines.append(f"  {head}{suffix if copies else ''}")
    if links:
        quoted = [link for link in links if link.excerpt]
        if quoted:
            lines.append("")
            lines.append("-- verbatim excerpts " + "-" * 57)
            for link in quoted[:10]:
                quote = truncate(link.excerpt, 200)
                lines.append(f'  [{link.document_id} {link.stance}] "{quote}"')
        analysed = [link for link in links if link.analysis]
        if analysed:
            lines.append("")
            lines.append("-- analysis (model reasoning, not evidence) " + "-" * 35)
            for link in analysed[:10]:
                lines.append(f"  [{link.document_id}] {truncate(link.analysis, 200)}")
    if assessment.gaps:
        lines.append("")
        lines.append("-- what would strengthen this " + "-" * 48)
        for gap in assessment.gaps:
            lines.append(f"  - {gap}")
    return "\n".join(lines)


def timeline_report(entries: Sequence[Any]) -> str:
    rows = []
    for entry in entries:
        rows.append(
            {
                "date": entry.date.date().isoformat() if entry.date else "undated",
                "precision": str(entry.precision),
                "kind": entry.kind,
                "sources": entry.independent_sources,
                "primary": entry.has_primary_source,
                "what": truncate(entry.description, 64),
                "evidence": ", ".join(entry.evidence_ids[:3]),
            }
        )
    return table(rows, ["date", "precision", "kind", "sources", "primary", "what", "evidence"])
