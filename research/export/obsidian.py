"""Exporting an investigation into an Obsidian vault.

The mapping is close to exact, which is why this integration is worth having:
an investigation is already a graph of claims, evidence, entities and events
joined by stable identifiers, and a vault is a graph of notes joined by
links. Each record becomes a note, each relationship becomes a wikilink, and
Obsidian's own graph view, backlinks and search then work on the research
without knowing anything about it.

Three properties are preserved in the translation:

*Independence.* A syndicated copy does not become a second note in the graph.
It is listed on the note for the document it copies, so counting notes and
counting sources give the same answer.

*Provenance.* Every evidence note records the provider call that produced it,
its content hash and its retrieval time, so a note can be checked against the
store it came from.

*Evidence is not analysis.* Retrieved text is quoted in a callout marked as
external, model reasoning is labelled as analysis, and the two never share a
block.

The vault belongs to the user, so this writes only inside its own folder and
refuses to overwrite a note it did not generate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research.export.markdown import (
    blockquote,
    callout,
    escape_external,
    frontmatter,
    safe_filename,
    table,
    wikilink,
)
from research.graph.claims import ClaimAssessment, ClaimGraph
from research.models.claim import EvidenceStance
from research.models.evidence import EvidenceDocument
from research.normalize.text import truncate
from research.storage.store import ResearchStore
from research.synthesis.report import ReportData, collect, render_markdown

#: Written into every generated note. A note without it was not written by
#: this exporter, and is never overwritten without being asked twice.
MARKER = "research-export"

#: Identifiers appearing in report prose, turned into links on the way out.
_ID_RE = re.compile(r"\b((?:claim|evidence|entity|event|task):\d+)\b")

_STANCE_COLOUR = {
    EvidenceStance.SUPPORTS: "4",  # green
    EvidenceStance.CONTRADICTS: "1",  # red
    EvidenceStance.QUALIFIES: "3",  # yellow
    EvidenceStance.MENTIONS: "6",  # purple
}


@dataclass(slots=True)
class ExportResult:
    folder: Path
    written: list[Path] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    canvas: Path | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "folder": str(self.folder),
            "notes_written": len(self.written),
            "skipped": [{"path": str(path), "reason": reason} for path, reason in self.skipped],
            "canvas": str(self.canvas) if self.canvas else None,
        }


class ObsidianExporter:
    def __init__(self, store: ResearchStore, investigation_id: str) -> None:
        self.store = store
        self.investigation_id = investigation_id
        self.claims = ClaimGraph(store, investigation_id)
        self._names: dict[str, str] = {}
        self._paths: dict[str, str] = {}

    # -- entry point ----------------------------------------------------
    def export(
        self,
        vault: str | Path,
        *,
        folder: str = "Research",
        force: bool = False,
        text_limit: int = 20000,
        canvas: bool = True,
    ) -> ExportResult:
        data = collect(self.store, self.investigation_id)
        self._build_registry(data)
        vault_path = Path(vault).expanduser().resolve()
        slug = self._names[self.investigation_id]
        target = (vault_path / folder / slug).resolve()
        if vault_path not in target.parents and target != vault_path:
            raise ValueError(f"refusing to write outside the vault: {target}")

        result = ExportResult(folder=target)

        for relative, body in self._notes(data, text_limit=text_limit):
            self._write(target / relative, body, result, force=force)

        if canvas:
            canvas_path = target / f"{slug}.canvas"
            self._write(
                canvas_path,
                json.dumps(self._canvas(data, folder=folder, slug=slug), indent=1),
                result,
                force=force,
            )
            if canvas_path in result.written:
                result.canvas = canvas_path
        return result

    # -- note registry --------------------------------------------------
    def _build_registry(self, data: ReportData) -> None:
        """Decide every note's name before any note is written.

        Names are needed on both sides of every link, so they are assigned in
        one pass and used consistently. The identifier leads the name, which
        keeps names stable when a title is later enriched and makes a note
        findable by the id a report cites.
        """
        self._names.clear()
        self._paths.clear()

        def register(identifier: str, label: str, subfolder: str | None) -> None:
            name = safe_filename(f"{identifier.replace(':', '-')} {label}")
            self._names[identifier] = name
            self._paths[identifier] = f"{subfolder}/{name}.md" if subfolder else f"{name}.md"

        # The index note is a link target like any other: every note ends with
        # a link back to it.
        register(
            self.investigation_id,
            truncate(data.investigation.question, 60),
            None,
        )

        for document in self.store.documents.list(self.investigation_id, limit=2000):
            register(document.id, truncate(document.title or "untitled", 70), "Evidence")
        for assessment in data.assessments:
            register(assessment.claim_id, truncate(assessment.text, 70), "Claims")
        for entity in self.store.entities.list(self.investigation_id, limit=500):
            register(entity.id, truncate(entity.name, 60), "Entities")
        for event in self.store.events.timeline(self.investigation_id, limit=500):
            register(event.id, truncate(event.description, 60), "Events")
        for task in self.store.tasks.list(self.investigation_id, limit=500):
            register(task.id, f"{task.role} {truncate(task.objective, 55)}", "Tasks")

    def _link(self, identifier: str, *, alias: str | None = None) -> str:
        name = self._names.get(identifier)
        if not name:
            return f"`{identifier}`"
        return wikilink(name, alias)

    def _linkify(self, text: str) -> str:
        """Turn identifiers in generated prose into links."""
        return _ID_RE.sub(lambda match: self._link(match.group(1), alias=match.group(1)), text)

    # -- notes ----------------------------------------------------------
    def _notes(self, data: ReportData, *, text_limit: int) -> list[tuple[str, str]]:
        notes: list[tuple[str, str]] = [
            (self._paths[self.investigation_id], self._investigation_note(data))
        ]
        for assessment in data.assessments:
            notes.append((self._paths[assessment.claim_id], self._claim_note(assessment)))
        for document in self.store.documents.list(self.investigation_id, limit=2000):
            notes.append(
                (self._paths[document.id], self._evidence_note(document, text_limit=text_limit))
            )
        for entity in self.store.entities.list(self.investigation_id, limit=500):
            notes.append((self._paths[entity.id], self._entity_note(entity)))
        for event in self.store.events.timeline(self.investigation_id, limit=500):
            notes.append((self._paths[event.id], self._event_note(event)))
        for task in self.store.tasks.list(self.investigation_id, limit=500):
            notes.append((self._paths[task.id], self._task_note(task)))
        return notes

    def _investigation_note(self, data: ReportData) -> str:
        investigation = data.investigation
        properties = {
            "generated_by": MARKER,
            "type": "investigation",
            "id": investigation.id,
            "aliases": [investigation.id],
            "status": str(investigation.status),
            "stop_reason": str(investigation.stop_reason) if investigation.stop_reason else None,
            "documents": data.counts["documents"],
            "independent_sources": data.counts["independent_sources"],
            "claims": data.counts["claims"],
            "tags": ["research/investigation"],
        }
        report = self._linkify(render_markdown(data, self.store))
        sections = [
            frontmatter(properties),
            report,
            "",
            "## Notes in this investigation",
            "",
        ]
        tasks = self.store.tasks.list(self.investigation_id, limit=500)
        entities = self.store.entities.list(self.investigation_id, limit=500)
        for label, identifiers in (
            ("Claims", [assessment.claim_id for assessment in data.assessments]),
            ("Tasks", [task.id for task in tasks]),
            ("Entities", [entity.id for entity in entities]),
        ):
            if not identifiers:
                continue
            sections.append(f"**{label}**: " + ", ".join(self._link(item) for item in identifiers))
            sections.append("")
        sections.append(
            callout(
                "info",
                "Sources are counted once",
                f"{data.counts['documents']} documents collapse to "
                f"{data.counts['independent_sources']} independent sources. Copies are "
                "listed on the note for the document they copy, so counting notes in the "
                "graph gives the same answer as counting sources.",
            )
        )
        return "\n".join(sections)

    def _claim_note(self, assessment: ClaimAssessment) -> str:
        links = self.store.claims.evidence_links(assessment.claim_id)
        properties = {
            "generated_by": MARKER,
            "type": "claim",
            "id": assessment.claim_id,
            "aliases": [assessment.claim_id],
            "investigation": self.investigation_id,
            "status": str(assessment.status),
            "independent_support": assessment.support.independent_count,
            "independent_against": assessment.contradiction.independent_count,
            "has_primary_source": assessment.support.has_primary_source,
            "possibly_superseded": assessment.possibly_superseded,
            "tags": ["research/claim", f"claim-status/{assessment.status}"],
        }
        body = [
            frontmatter(properties),
            f"# {escape_external(assessment.text)}",
            "",
            callout(
                "abstract",
                f"Assessment — {assessment.status}",
                escape_external(assessment.explanation),
            ),
            "",
        ]
        if assessment.possibly_superseded:
            body.extend(
                [
                    callout(
                        "warning",
                        "Possibly superseded",
                        "The contradicting evidence is more recent than any supporting "
                        "evidence.",
                    ),
                    "",
                ]
            )

        for stance, heading in (
            (EvidenceStance.SUPPORTS, "Supporting evidence"),
            (EvidenceStance.CONTRADICTS, "Contradicting evidence"),
            (EvidenceStance.QUALIFIES, "Qualifying evidence"),
            (EvidenceStance.MENTIONS, "Mentions"),
        ):
            relevant = [link for link in links if link.stance is stance]
            if not relevant:
                continue
            body.extend([f"## {heading}", ""])
            for link in relevant:
                document = self.store.documents.get(link.document_id)
                marker = ""
                if document.duplicate_of or document.derived_from:
                    parent = document.duplicate_of or document.derived_from
                    marker = f" — *not independent of {self._link(parent, alias=parent)}*"
                body.append(f"- {self._link(document.id)}{marker}")
                if link.excerpt:
                    # Indented so the quote stays inside the list item rather
                    # than ending the list.
                    quoted = blockquote(escape_external(link.excerpt))
                    body.append("\n".join(f"  {line}" for line in quoted.split("\n")))
                if link.analysis:
                    body.append(
                        "  - *analysis (model reasoning, not evidence):* "
                        f"{escape_external(link.analysis)}"
                    )
            body.append("")

        if assessment.gaps:
            body.extend([
                callout(
                    "question",
                    "What would strengthen this",
                    "\n".join(f"- {escape_external(gap)}" for gap in assessment.gaps),
                ),
                "",
            ])
        body.append(f"Part of {self._link(self.investigation_id)}")
        return "\n".join(body)

    def _evidence_note(self, document: EvidenceDocument, *, text_limit: int) -> str:
        copies = self.store.documents.duplicates_of(document.id)
        claims = self.store.claims.claims_for_document(document.id)
        independence = "independent"
        if document.duplicate_of or document.derived_from:
            independence = f"{document.duplicate_relation}"

        properties = {
            "generated_by": MARKER,
            "type": "evidence",
            "id": document.id,
            "aliases": [document.id],
            "investigation": self.investigation_id,
            "source_type": str(document.source_type),
            "source_family": str(document.source_family),
            "provider": document.provider,
            "publisher": document.publisher,
            "authors": document.authors or None,
            "published": document.published_at.date().isoformat()
            if document.published_at
            else None,
            "retrieved": document.fetched_at.isoformat() if document.fetched_at else None,
            "url": document.canonical_url,
            "doi": document.doi,
            "content_hash": document.content_hash,
            "independence": independence,
            "retracted": bool(document.metadata.get("is_retracted")) or None,
            "tags": [
                "research/evidence",
                f"source/{document.source_type}",
                *(["evidence/not-independent"] if independence != "independent" else []),
                *(["evidence/retracted"] if document.metadata.get("is_retracted") else []),
            ],
        }
        body = [
            frontmatter(properties),
            f"# {escape_external(document.title or document.id)}",
            "",
            f"*{document.source_type} · {escape_external(document.citation_label())}*",
            "",
        ]
        if document.metadata.get("is_retracted"):
            body.extend([
                callout("danger", "Retracted", "The provider reports this work as retracted."),
                "",
            ])
        if document.duplicate_of or document.derived_from:
            parent = document.duplicate_of or document.derived_from
            body.extend([
                callout(
                    "warning",
                    f"Not independent — {document.duplicate_relation}",
                    f"{escape_external(str(document.metadata.get('duplicate_reason', '')))}\n\n"
                    f"Counted as one source with {self._link(parent, alias=parent)}.",
                ),
                "",
            ])
        if copies:
            body.extend([
                callout(
                    "info",
                    f"Also retrieved as {len(copies)} cop{'y' if len(copies) == 1 else 'ies'}",
                    "\n".join(
                        f"- {self._link(copy.id)} — {copy.duplicate_relation} via "
                        f"{escape_external(copy.canonical_host or copy.provider)}"
                        for copy in copies
                    ),
                ),
                "",
            ])
        if claims:
            body.extend(["## Bears on", ""])
            for link in claims:
                body.append(f"- {self._link(link.claim_id)} — **{link.stance}**")
            body.append("")

        body.extend(["## Provenance", ""])
        provenance_rows = [
            ["provider", document.provider],
            ["url", document.canonical_url or "—"],
            ["doi", document.doi or "—"],
            ["retrieved", document.fetched_at.isoformat() if document.fetched_at else "—"],
            ["content hash", document.content_hash],
        ]
        if document.provenance:
            provenance_rows.insert(1, ["retrieval", document.provenance.describe()])
        body.extend([table(["field", "value"], provenance_rows), ""])

        text = document.best_text
        if text:
            body.extend([
                callout(
                    "quote",
                    "Retrieved content (external, untrusted)",
                    escape_external(text, limit=text_limit),
                    fold="-",
                ),
                "",
            ])
        body.append(f"Part of {self._link(self.investigation_id)}")
        return "\n".join(body)

    def _entity_note(self, entity: Any) -> str:
        from research.graph.entities import EntityRegistry

        registry = EntityRegistry(self.store, self.investigation_id)
        documents = registry.documents_for(entity.id)
        claims = registry.claims_for(entity.id)
        properties = {
            "generated_by": MARKER,
            "type": "entity",
            "id": entity.id,
            "aliases": [entity.id, *entity.aliases],
            "investigation": self.investigation_id,
            "entity_type": str(entity.entity_type),
            "identifiers": [identifier.key() for identifier in entity.identifiers] or None,
            "review_needed": bool(entity.metadata.get("review_needed")) or None,
            "tags": ["research/entity", f"entity/{entity.entity_type}"],
        }
        body = [
            frontmatter(properties),
            f"# {escape_external(entity.name)}",
            "",
        ]
        if entity.metadata.get("review_needed"):
            body.extend([
                callout(
                    "warning",
                    "Matched on a name alone",
                    "This entity was matched by name without an authoritative identifier, "
                    "so it may conflate different people or organisations.",
                ),
                "",
            ])
        if entity.identifiers:
            body.extend([
                "**Identifiers**: "
                + ", ".join(f"`{identifier.key()}`" for identifier in entity.identifiers),
                "",
            ])
        if documents:
            body.extend(["## Appears in", ""])
            body.extend(f"- {self._link(document_id)}" for document_id in documents)
            body.append("")
        if claims:
            body.extend(["## Claims about this", ""])
            body.extend(f"- {self._link(claim_id)}" for claim_id in claims)
            body.append("")
        body.append(f"Part of {self._link(self.investigation_id)}")
        return "\n".join(body)

    def _event_note(self, event: Any) -> str:
        properties = {
            "generated_by": MARKER,
            "type": "event",
            "id": event.id,
            "aliases": [event.id],
            "investigation": self.investigation_id,
            "date": event.date_start.date().isoformat() if event.date_start else None,
            "date_precision": str(event.date_precision),
            "tags": ["research/event"],
        }
        body = [
            frontmatter(properties),
            f"# {escape_external(event.description)}",
            "",
            f"*{event.date_start.date().isoformat() if event.date_start else 'undated'} "
            f"({event.date_precision})*",
            "",
        ]
        if event.evidence_ids:
            body.extend(["## Evidence", ""])
            body.extend(f"- {self._link(document_id)}" for document_id in event.evidence_ids)
            body.append("")
        if event.entity_ids:
            body.extend(["## Involved", ""])
            body.extend(f"- {self._link(entity_id)}" for entity_id in event.entity_ids)
            body.append("")
        body.append(f"Part of {self._link(self.investigation_id)}")
        return "\n".join(body)

    def _task_note(self, task: Any) -> str:
        properties = {
            "generated_by": MARKER,
            "type": "task",
            "id": task.id,
            "aliases": [task.id],
            "investigation": self.investigation_id,
            "role": str(task.role),
            "operation": str(task.operation),
            "status": str(task.status),
            "depth": task.depth,
            "parent": task.parent_task_id,
            "tags": ["research/task", f"role/{task.role}"],
        }
        body = [
            frontmatter(properties),
            f"# {task.id} — {str(task.role)}",
            "",
            f"**Objective**: {escape_external(task.objective)}",
            "",
        ]
        if task.parent_task_id:
            body.extend([f"Spawned by {self._link(task.parent_task_id)}", ""])
        children = self.store.tasks.children(task.id)
        if children:
            body.extend(["## Spawned", ""])
            body.extend(f"- {self._link(child.id)}" for child in children)
            body.append("")
        if task.result:
            body.extend([
                callout("summary", "Reported back", escape_external(task.result.summary)),
                "",
            ])
            if task.result.evidence_ids:
                body.extend(["## Evidence found", ""])
                body.extend(
                    f"- {self._link(document_id)}" for document_id in task.result.evidence_ids[:40]
                )
                body.append("")
            if task.result.claim_ids:
                body.extend(["## Claims made", ""])
                body.extend(f"- {self._link(claim_id)}" for claim_id in task.result.claim_ids)
                body.append("")
        if task.error:
            body.extend([callout("failure", "Did not finish", escape_external(task.error)), ""])
        body.append(f"Part of {self._link(self.investigation_id)}")
        return "\n".join(body)

    # -- canvas ---------------------------------------------------------
    def _canvas(self, data: ReportData, *, folder: str, slug: str) -> dict[str, Any]:
        """An Obsidian canvas of the claim/evidence graph.

        Laid out in two deterministic columns rather than by a force
        simulation, so re-exporting an investigation does not rearrange a
        canvas the reader has been looking at.
        """
        base = f"{folder}/{slug}"
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        claims = data.assessments
        sources = [source.document for source in data.sources]
        for index, assessment in enumerate(claims):
            nodes.append({
                "id": assessment.claim_id.replace(":", "-"),
                "type": "file",
                "file": f"{base}/{self._paths[assessment.claim_id]}",
                "x": -700,
                "y": index * 180,
                "width": 560,
                "height": 140,
                "color": "5",
            })
        for index, document in enumerate(sources):
            nodes.append({
                "id": document.id.replace(":", "-"),
                "type": "file",
                "file": f"{base}/{self._paths[document.id]}",
                "x": 200,
                "y": index * 180,
                "width": 560,
                "height": 140,
                **({"color": "1"} if document.metadata.get("is_retracted") else {}),
            })

        # Copies are represented by the document they copy, exactly as they
        # are in every count.
        represented = {document.id: document.id for document in sources}
        for source in data.sources:
            for copy in source.copies:
                represented[copy.id] = source.document.id

        for assessment in claims:
            for link in self.store.claims.evidence_links(assessment.claim_id):
                target = represented.get(link.document_id)
                if not target:
                    continue
                edges.append({
                    "id": f"{target}-{assessment.claim_id}-{link.stance}".replace(":", "-"),
                    "fromNode": target.replace(":", "-"),
                    "fromSide": "left",
                    "toNode": assessment.claim_id.replace(":", "-"),
                    "toSide": "right",
                    "label": str(link.stance),
                    "color": _STANCE_COLOUR.get(link.stance, "6"),
                })
        # The marker goes first so the same "did we write this?" check works
        # on a canvas as on a note. If a reader edits the canvas in Obsidian
        # the key may not survive the save - at which point the file is
        # theirs, and declining to overwrite it is the right answer.
        return {
            "generated_by": MARKER,
            "nodes": nodes,
            "edges": _unique_edges(edges),
        }

    # -- writing --------------------------------------------------------
    def _write(
        self,
        path: Path,
        body: str,
        result: ExportResult,
        *,
        force: bool,
    ) -> None:
        """Write one file, refusing to clobber anything we did not generate."""
        resolved = path.resolve()
        if result.folder not in resolved.parents and resolved != result.folder:
            result.skipped.append((path, "outside the export folder"))
            return
        if resolved.exists() and not force:
            existing = resolved.read_text(encoding="utf-8", errors="replace")
            if MARKER not in existing[:400]:
                result.skipped.append(
                    (resolved, "a note already exists here and was not generated by this export")
                )
                return
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(body.rstrip() + "\n", encoding="utf-8")
        result.written.append(resolved)


def _unique_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for edge in edges:
        if edge["id"] in seen:
            continue
        seen.add(edge["id"])
        unique.append(edge)
    return unique
