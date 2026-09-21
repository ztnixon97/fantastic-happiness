"""Smoke tests for the real ML stack. Deselected by default.

    uv run pytest -m models

These exist because the rest of the suite deliberately runs no models, and
so cannot tell a working install from a broken one. A torch/torchvision
wheel mismatch once left the embedder raising on every load while all 674
other tests passed. Run these after changing dependencies.

They need the model weights: already cached, or a network to fetch them on
first run.
"""

from __future__ import annotations

import pytest

from research.normalize import docling_reader
from research.normalize.files import extract_file
from research.retrieval.embeddings import LocalEmbedder, cosine

pytestmark = pytest.mark.models


class TestTheEmbedderActuallyRuns:
    @pytest.mark.asyncio
    async def test_it_loads_and_returns_vectors_of_the_stated_width(self) -> None:
        embedder = LocalEmbedder()
        vectors = await embedder.embed(["A notice about reactor construction costs."])
        assert embedder.unavailable is None, embedder.unavailable
        assert len(vectors) == 1
        assert len(vectors[0]) == embedder.dimensions

    @pytest.mark.asyncio
    async def test_it_is_semantic_not_lexical(self) -> None:
        """The whole reason for vectors: same meaning, different words."""
        embedder = LocalEmbedder()
        same, different, query = await embedder.embed(
            [
                "Reactor builds exceeded their capital cost estimates.",
                "Benthic invertebrate communities of the Baltic Sea.",
                "nuclear construction budget overrun",
            ]
        )
        assert cosine(query, same) > cosine(query, different)
        assert cosine(query, same) > 0.3, "not a single word is shared"


class TestDoclingActuallyRuns:
    def test_it_is_installed(self) -> None:
        assert docling_reader.available(), "docling is a dependency, not an extra"

    def test_it_converts_a_pdf(self, build_pdf) -> None:
        text = [
            "Construction cost overruns in small modular reactor programmes",
            "Realized overnight capital costs exceeded the initial estimates.",
        ]
        converted = docling_reader.convert(build_pdf(text), "paper.pdf")
        assert converted is not None
        assert converted.ok, converted.warnings
        assert "capital costs" in converted.text

    def test_the_default_settings_read_a_word_document(self, tmp_path) -> None:
        import zipfile

        path = tmp_path / "note.docx"
        body = (
            '<w:p><w:r><w:t xml:space="preserve">'
            "The operator stated that the queue wait is four years."
            "</w:t></w:r></w:p>"
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                'package/2006/content-types"><Default Extension="xml" ContentType='
                '"application/xml"/><Default Extension="rels" ContentType='
                '"application/vnd.openxmlformats-package.relationships+xml"/><Override '
                'PartName="/word/document.xml" ContentType="application/vnd.'
                'openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
            )
            archive.writestr(
                "_rels/.rels",
                '<?xml version="1.0"?><Relationships xmlns="http://schemas.'
                'openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships'
                '/officeDocument" Target="word/document.xml"/></Relationships>',
            )
            archive.writestr(
                "word/document.xml",
                '<?xml version="1.0"?><w:document xmlns:w="http://schemas.'
                'openxmlformats.org/wordprocessingml/2006/main"><w:body>'
                + body
                + "</w:body></w:document>",
            )

        from research.config import IngestSettings

        extracted = extract_file(
            path.name,
            path.read_bytes(),
            docling=docling_reader.converter_for(IngestSettings()),
        )
        assert extracted.ok, extracted.warnings
        assert "four years" in extracted.text
