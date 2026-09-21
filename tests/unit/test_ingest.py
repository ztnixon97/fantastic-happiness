"""Reading local files in as evidence: extraction, walking, and its limits."""

from __future__ import annotations

import json

import pytest

from research.acquisition.ingest import LocalIngest
from research.models.common import SourceFamily, SourceType
from research.normalize.files import detect_kind, extract_file
from research.normalize import pdf
from research.normalize.pdf import extract_pdf, looks_like_prose
from research.storage.store import ResearchStore

PARAGRAPHS = [
    "Construction cost overruns in small modular reactor programmes",
    "Realized overnight capital costs exceeded the initial estimates by a median",
    "of one hundred and seventeen percent across the sample of builds studied.",
]


class TestPdfExtraction:
    def test_text_comes_out_of_a_compressed_pdf(self, build_pdf) -> None:
        extracted = extract_pdf(build_pdf(PARAGRAPHS, title="Cost overruns"))
        assert extracted.ok
        assert "overnight capital costs" in extracted.text
        assert extracted.title == "Cost overruns"
        assert extracted.pages == 1

    def test_the_built_in_reader_works_without_pypdf(self, build_pdf, monkeypatch) -> None:
        """The dependency is optional, so the suite exercises life without it."""
        monkeypatch.setattr(pdf, "_with_pypdf", lambda data: None)
        extracted = extract_pdf(build_pdf(PARAGRAPHS, title="Cost overruns"))
        assert extracted.method == "builtin"
        assert "overnight capital costs" in extracted.text

    def test_a_file_pypdf_refuses_still_gets_read(self, build_pdf, monkeypatch) -> None:
        """pypdf is stricter about structure than the format is in practice."""
        monkeypatch.setattr(
            pdf,
            "_with_pypdf",
            lambda data: pdf.PdfText(method="pypdf", warnings=["startxref not found"]),
        )
        extracted = extract_pdf(build_pdf(PARAGRAPHS))
        assert extracted.method == "builtin"
        assert "overnight capital costs" in extracted.text

    def test_an_uncompressed_pdf_reads_the_same(self, build_pdf) -> None:
        compressed = extract_pdf(build_pdf(PARAGRAPHS))
        plain = extract_pdf(build_pdf(PARAGRAPHS, compress=False))
        assert compressed.text == plain.text

    def test_unmappable_font_codes_are_refused_not_stored(self, build_pdf) -> None:
        """The dangerous failure: noise that looks like text to everything downstream."""
        extracted = extract_pdf(build_pdf(PARAGRAPHS, hex_strings=True))
        assert not extracted.ok
        assert extracted.method == "none"
        assert any("not legible" in warning for warning in extracted.warnings)
        assert any("pypdf" in warning for warning in extracted.warnings)

    def test_something_that_is_not_a_pdf_is_refused_by_its_header(self) -> None:
        extracted = extract_pdf(b"<html><body>not a pdf</body></html>")
        assert not extracted.ok
        assert "not a PDF" in extracted.warnings[0]

    def test_a_pdf_with_no_text_says_so(self) -> None:
        extracted = extract_pdf(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF")
        assert not extracted.ok
        assert any("no text" in warning for warning in extracted.warnings)

    @pytest.mark.parametrize(
        "text, prose",
        [
            ("The utility said it had agreed to supply four hundred megawatts.", True),
            ("⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌" * 8, False),
            ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", False),
            ("short", False),
        ],
    )
    def test_legibility_separates_prose_from_noise(self, text: str, prose: bool) -> None:
        assert looks_like_prose(text) is prose


class TestFileExtraction:
    def test_content_decides_the_kind_not_the_extension(self, build_pdf) -> None:
        assert detect_kind("notes.txt", build_pdf(PARAGRAPHS)) == "pdf"
        assert detect_kind("page.txt", b"<!DOCTYPE html><html>") == "html"
        assert detect_kind("notes.md", b"# heading") == "markdown"
        assert detect_kind("archive.zip", b"PK\x03\x04") is None

    def test_markdown_frontmatter_supplies_title_and_author(self) -> None:
        source = (
            "---\ntitle: Interconnection queue notes\nauthor: A. Researcher\n---\n\n"
            "# Ignored heading\n\nThe queue wait is four years in that region.\n"
        )
        extracted = extract_file("notes.md", source.encode())
        assert extracted.title == "Interconnection queue notes"
        assert extracted.authors == ["A. Researcher"]
        assert "queue wait is four years" in extracted.text
        assert "title: Interconnection" not in extracted.text

    def test_markdown_without_frontmatter_takes_its_first_heading(self) -> None:
        extracted = extract_file("notes.md", b"# Board minutes\n\nThe motion carried.\n")
        assert extracted.title == "Board minutes"

    def test_html_is_parsed_rather_than_stored_as_markup(self) -> None:
        html = b"<html><head><title>Utility statement</title></head><body><p>Four hundred megawatts.</p></body></html>"
        extracted = extract_file("press.html", html)
        assert extracted.title == "Utility statement"
        assert "<p>" not in extracted.text
        assert "Four hundred megawatts" in extracted.text

    def test_json_is_kept_as_data(self) -> None:
        payload = {"title": "Capacity filing", "megawatts": 400}
        extracted = extract_file("filing.json", json.dumps(payload).encode())
        assert extracted.title == "Capacity filing"
        assert json.loads(extracted.text) == payload

    def test_plain_text_takes_a_short_opening_line_as_its_title(self) -> None:
        extracted = extract_file("note.txt", b"Meeting notes\n\nThe operator confirmed it.\n")
        assert extracted.title == "Meeting notes"

    def test_an_unsupported_type_is_named_rather_than_guessed_at(self) -> None:
        extracted = extract_file("disk.img", b"\x00\x01\x02\x03")
        assert not extracted.ok
        assert "unsupported file type" in extracted.warnings[0]

    def test_text_is_capped(self) -> None:
        extracted = extract_file("big.txt", b"word " * 10_000, max_characters=200)
        assert len(extracted.text) <= 200


@pytest.fixture
def folder(tmp_path, build_pdf):
    """A folder shaped like the one somebody would actually point this at."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "paper.pdf").write_bytes(build_pdf(PARAGRAPHS, title="Cost overruns"))
    (tmp_path / "notes" / "meeting.md").write_text(
        "# Interconnection queue notes\n\nThe operator said the wait is four years.\n"
    )
    (tmp_path / "notes" / "press.html").write_text(
        "<html><head><title>Utility statement</title></head>"
        "<body><p>It agreed to supply four hundred megawatts.</p></body></html>"
    )
    (tmp_path / "photo.jpeg").write_bytes(b"\xff\xd8\xff\xe0not an image really")
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "secret.md").write_text("# Hidden\n\nShould never be read.\n")
    return tmp_path


class TestLocalIngest:
    def test_a_folder_becomes_evidence(self, store: ResearchStore, investigation, folder) -> None:
        report = LocalIngest(store, investigation_id=investigation.id).ingest([folder])
        assert report.summary() == {
            "read": 3, "new_evidence": 3, "already_held": 0, "skipped": 0
        }
        titles = {document.title for document in report.documents}
        assert "Cost overruns" in titles
        assert "Interconnection queue notes" in titles

    def test_unreadable_types_are_passed_over_silently(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        report = LocalIngest(store, investigation_id=investigation.id).ingest([folder])
        read = {entry.path.name for entry in report.files if entry.result}
        assert "photo.jpeg" not in read

    def test_hidden_directories_are_not_walked(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        report = LocalIngest(store, investigation_id=investigation.id).ingest([folder])
        assert not any(".cache" in str(entry.path) for entry in report.files)

    def test_a_symlink_out_of_the_folder_is_refused(
        self, store: ResearchStore, investigation, folder, tmp_path
    ) -> None:
        outside = tmp_path.parent / "outside.md"
        outside.write_text("# Outside\n\nNot in the folder that was named.\n")
        (folder / "notes" / "link.md").symlink_to(outside)

        report = LocalIngest(store, investigation_id=investigation.id).ingest([folder])
        skipped = {entry.path.name: entry.skipped for entry in report.skipped}
        assert "link.md" in skipped
        assert "outside" in skipped["link.md"]
        assert "Outside" not in {document.title for document in report.documents}

    def test_a_named_file_is_read_even_without_a_folder(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        report = LocalIngest(store, investigation_id=investigation.id).ingest(
            [folder / "paper.pdf"]
        )
        assert report.summary()["new_evidence"] == 1

    def test_no_recurse_reads_nothing_below_the_folder(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        report = LocalIngest(store, investigation_id=investigation.id).ingest(
            [folder], recursive=False
        )
        assert [entry.path.name for entry in report.files if entry.result] == ["paper.pdf"]

    def test_files_above_the_size_limit_are_skipped_not_loaded(
        self, store: ResearchStore, investigation, tmp_path
    ) -> None:
        big = tmp_path / "big.txt"
        big.write_text("word " * 5000)
        report = LocalIngest(
            store, investigation_id=investigation.id, max_file_bytes=1024
        ).ingest([big])
        assert report.skipped[0].skipped.endswith("limit")
        assert report.documents == []

    def test_a_missing_path_is_reported_rather_than_raised(
        self, store: ResearchStore, investigation, tmp_path
    ) -> None:
        report = LocalIngest(store, investigation_id=investigation.id).ingest(
            [tmp_path / "nowhere.pdf"]
        )
        assert report.skipped[0].skipped == "no such file or directory"

    def test_ingesting_twice_merges_rather_than_duplicating(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        ingest = LocalIngest(store, investigation_id=investigation.id)
        first = ingest.ingest([folder])
        second = ingest.ingest([folder])
        assert second.summary()["new_evidence"] == 0
        assert second.summary()["already_held"] == 3
        assert {document.id for document in first.documents} == {
            document.id for document in second.documents
        }

    def test_a_file_named_twice_is_read_once(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        report = LocalIngest(store, investigation_id=investigation.id).ingest(
            [folder / "paper.pdf", folder / "paper.pdf", folder]
        )
        assert report.summary()["read"] == 3

    def test_provenance_records_where_the_document_came_from(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        report = LocalIngest(store, investigation_id=investigation.id).ingest(
            [folder / "paper.pdf"], notes="handed over by the analyst"
        )
        document = report.documents[0]
        assert document.provenance.provider == "local"
        assert str(document.provenance.retrieval_method) == "seed"
        assert document.provenance.notes == "handed over by the analyst"
        assert document.metadata["local_path"].endswith("paper.pdf")
        assert document.metadata["file_kind"] == "pdf"

    def test_a_file_is_not_dated_by_when_it_was_saved(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        """An mtime is not a publication date and must not be treated as one."""
        report = LocalIngest(store, investigation_id=investigation.id).ingest([folder])
        assert all(document.published_at is None for document in report.documents)
        assert all(
            document.metadata.get("file_modified_at") for document in report.documents
        )

    def test_the_caller_says_what_the_material_is(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        report = LocalIngest(store, investigation_id=investigation.id).ingest(
            [folder / "paper.pdf"],
            source_type=SourceType.ACADEMIC_PEER_REVIEWED,
            source_family=SourceFamily.ACADEMIC,
        )
        document = report.documents[0]
        assert document.source_type is SourceType.ACADEMIC_PEER_REVIEWED
        assert document.source_family is SourceFamily.ACADEMIC

    def test_an_illegible_pdf_is_skipped_with_its_reason(
        self, store: ResearchStore, investigation, tmp_path, build_pdf
    ) -> None:
        scanned = tmp_path / "scanned.pdf"
        scanned.write_bytes(build_pdf(PARAGRAPHS, hex_strings=True))
        report = LocalIngest(store, investigation_id=investigation.id).ingest([scanned])
        assert report.documents == []
        assert "not legible" in report.skipped[0].skipped

    async def test_ingested_evidence_is_immediately_searchable(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        from research.retrieval.search import CorpusSearch

        LocalIngest(store, investigation_id=investigation.id).ingest([folder])
        hits = await CorpusSearch(store, investigation.id).search("overnight capital costs")
        assert hits[0].document.title == "Cost overruns"

    def test_ingested_evidence_is_read_back_as_untrusted(
        self, store: ResearchStore, investigation, folder
    ) -> None:
        from research.acquisition.untrusted import as_external_evidence

        report = LocalIngest(store, investigation_id=investigation.id).ingest([folder])
        rendered = as_external_evidence(report.documents[0])
        assert rendered.startswith("<external_evidence")


class TestFetchedPdfs:
    """A record published as a PDF should be reachable over HTTP, too."""

    async def _fetch(self, body: bytes, content_type: str = "application/pdf"):
        import httpx

        from research.config import AcquisitionPolicy
        from research.sources.fetcher import DirectFetchSource
        from research.sources.http import SafeHttpClient

        client = SafeHttpClient(
            AcquisitionPolicy(per_host_min_interval_seconds=0.0),
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=body, headers={"content-type": content_type}
                )
            ),
        )
        return await DirectFetchSource(client).fetch_url(
            "https://regulator.test/filing.pdf"
        )

    async def test_a_fetched_pdf_becomes_readable_evidence(self, build_pdf) -> None:
        document = await self._fetch(build_pdf(PARAGRAPHS, title="Cost overruns"))
        assert "overnight capital costs" in (document.text or "")
        assert document.title == "Cost overruns"
        assert document.metadata["pdf_extractor"] in ("builtin", "pypdf")

    async def test_a_pdf_served_as_something_else_is_still_read_as_a_pdf(
        self, build_pdf
    ) -> None:
        document = await self._fetch(build_pdf(PARAGRAPHS), content_type="text/plain")
        assert "overnight capital costs" in (document.text or "")

    async def test_an_illegible_pdf_produces_no_text_rather_than_noise(
        self, build_pdf
    ) -> None:
        document = await self._fetch(build_pdf(PARAGRAPHS, hex_strings=True))
        assert not (document.text or "").strip()
        assert "not legible" in document.metadata.get("pdf_warnings", "")
