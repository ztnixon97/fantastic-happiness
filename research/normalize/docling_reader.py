"""Document conversion through Docling, when an operator has enabled it.

The built-in readers get text out of a PDF that has text in it. They do not
do layout analysis, they lose table structure, they cannot open a .docx, and
they cannot read a scan at all. Docling does all four, so this is the seam
for it: a converter asked first when it is switched on, whose absence
changes nothing.

Three things make this unlike the rest of :mod:`research.normalize`, and
they are why it is off by default rather than simply used:

* **It is not pure.** Docling runs layout and table models, and downloads
  their weights on first use - from Hugging Face, and, for OCR, from
  whichever host the chosen engine uses. Point ``artifacts_path`` at a
  directory you have pre-populated and it converts without reaching out at
  all, which is the right configuration for a machine that should not.
* **It runs inference over hostile input.** Everything this system reads is
  external material, and OCR means native image decoders parsing bytes from
  the open web. That is a materially larger attack surface than a regular
  expression over a content stream. A reasonable trade for being able to
  read a scanned filing; not a reasonable default.
* **It is slow enough to notice.** Seconds per page, tens of seconds with
  OCR, against milliseconds for the built-in reader.

Nothing here raises. A conversion that fails returns a result carrying its
reason, and the caller falls back to the readers that need no dependency.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

#: Formats Docling adds. PDFs, HTML, text and JSON are read without it;
#: these cannot be read at all without it.
DOCLING_ONLY_EXTENSIONS: dict[str, str] = {
    ".docx": "word",
    ".doc": "word",
    ".odt": "word",
    ".rtf": "word",
    ".pptx": "presentation",
    ".ppt": "presentation",
    ".xlsx": "spreadsheet",
    ".xls": "spreadsheet",
    ".epub": "book",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
}

#: A picture with no readable text exports as this placeholder. It is not
#: content, and it must not count as body text.
_IMAGE_PLACEHOLDER_RE = re.compile(r"^<!--\s*image\s*-->$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,3}\s+(\S.*?)\s*$", re.MULTILINE)

#: A converter the rest of the package can hold without importing Docling:
#: ``(data, filename, ocr=bool) -> DoclingResult | None``.
DoclingConverter = Callable[..., "DoclingResult | None"]


@dataclass(slots=True)
class DoclingResult:
    text: str = ""
    title: str | None = None
    #: Where the title came from, when there is one. Docling infers it from
    #: layout rather than reading it off a metadata field, so a document
    #: should record that its title was inferred.
    title_source: str | None = None
    pages: int = 0
    tables: int = 0
    #: "docling" or "docling+ocr", so a document records how it was read.
    method: str = "docling"
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def available() -> bool:
    """Whether Docling is installed. Does not import it to find out."""
    try:
        return importlib.util.find_spec("docling") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def converter_for(settings: Any) -> DoclingConverter | None:
    """Bind an :class:`~research.config.IngestSettings` to a converter.

    Returns ``None`` when Docling is switched off or not installed, which is
    how every caller asks "should I use it?" without importing it.
    """
    if not getattr(settings, "docling_enabled", False) or not available():
        return None

    def convert_with_settings(
        data: bytes, filename: str, *, ocr: bool = False
    ) -> DoclingResult | None:
        return convert(
            data,
            filename,
            ocr=ocr,
            languages=getattr(settings, "ocr_languages", ("en",)),
            max_pages=getattr(settings, "max_pages", None),
            artifacts_path=getattr(settings, "docling_artifacts_path", None),
        )

    return convert_with_settings


def convert(
    data: bytes,
    filename: str,
    *,
    ocr: bool = False,
    languages: Sequence[str] = ("en",),
    max_pages: int | None = None,
    artifacts_path: str | None = None,
) -> DoclingResult | None:
    """Convert one document. ``None`` means Docling is not installed at all.

    Every other failure - an unsupported format, a version whose API has
    moved, a conversion that did not succeed - comes back as a result with
    no text and a warning saying why, so the caller can fall back and still
    report what was tried.
    """
    if not available():
        return None
    try:
        converter, stream = _prepare(
            data, filename, ocr=ocr, languages=languages, artifacts_path=artifacts_path
        )
    except Exception as exc:  # an API that moved, a missing OCR engine
        return DoclingResult(warnings=[f"docling could not be configured: {_short(exc)}"])

    limits: dict[str, Any] = {"raises_on_error": False}
    if max_pages:
        limits["max_num_pages"] = int(max_pages)
    try:
        outcome = converter.convert(stream, **limits)
    except TypeError:  # pragma: no cover - an older signature
        try:
            outcome = converter.convert(stream)
        except Exception as exc:
            return DoclingResult(warnings=[f"docling failed: {_short(exc)}"])
    except Exception as exc:
        return DoclingResult(warnings=[f"docling failed: {_short(exc)}"])

    return _read(outcome, ocr=ocr)


def _prepare(
    data: bytes,
    filename: str,
    *,
    ocr: bool,
    languages: Sequence[str],
    artifacts_path: str | None,
) -> tuple[Any, Any]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = bool(ocr)
    options.do_table_structure = True
    if getattr(options, "table_structure_options", None) is not None:
        options.table_structure_options.do_cell_matching = True
    if artifacts_path:
        options.artifacts_path = artifacts_path
    if ocr and languages and getattr(options, "ocr_options", None) is not None:
        # Language codes differ between OCR engines, and a rejected list
        # should not lose the conversion.
        with contextlib.suppress(Exception):
            options.ocr_options.lang = list(languages)

    format_options: dict[Any, Any] = {InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    try:
        from docling.document_converter import ImageFormatOption

        format_options[InputFormat.IMAGE] = ImageFormatOption(pipeline_options=options)
    except ImportError:  # pragma: no cover - older builds route images differently
        format_options[InputFormat.IMAGE] = PdfFormatOption(pipeline_options=options)

    return DocumentConverter(format_options=format_options), _stream(data, filename)


def _stream(data: bytes, filename: str) -> Any:
    """Wrap bytes in whatever this Docling version calls a document stream."""
    for module_name in ("docling_core.types.io", "docling.datamodel.base_models"):
        try:
            module = __import__(module_name, fromlist=["DocumentStream"])
        except ImportError:
            continue
        stream_class = getattr(module, "DocumentStream", None)
        if stream_class is not None:
            return stream_class(name=filename, stream=io.BytesIO(data))
    raise RuntimeError("this docling build has no DocumentStream")  # pragma: no cover


def _read(outcome: Any, *, ocr: bool) -> DoclingResult:
    result = DoclingResult(method="docling+ocr" if ocr else "docling")

    status = str(getattr(outcome, "status", "") or "").lower()
    if "fail" in status:
        errors = getattr(outcome, "errors", None) or []
        result.warnings.append(
            "docling could not convert the document: "
            + (_short(errors[0]) if errors else "no reason given")
        )
        return result
    if "partial" in status:
        result.warnings.append("docling converted only part of the document")

    document = getattr(outcome, "document", None)
    if document is None:
        result.warnings.append("docling returned no document")
        return result

    try:
        markdown = document.export_to_markdown() or ""
    except Exception as exc:
        result.warnings.append(f"docling produced a document that could not be read: {_short(exc)}")
        return result

    result.text = _IMAGE_PLACEHOLDER_RE.sub("", markdown).strip()
    result.pages = len(getattr(document, "pages", ()) or ())
    result.tables = len(getattr(document, "tables", ()) or ())
    result.title, result.title_source = _title(document, result.text)
    return result


def _title(document: Any, markdown: str) -> tuple[str | None, str | None]:
    """A title, and an honest account of where it came from.

    Docling does not read a title off a metadata field; it decides from
    layout what the title is. An item it labelled a title is the best answer
    available, and the first heading is the next best - which for a paper is
    usually its actual title, and is in any case better than nothing for a
    PDF that carries no metadata at all. ``document.name`` is not used: it is
    the file's own name wearing a different hat.
    """
    for item in getattr(document, "texts", ()) or ():
        label = str(getattr(item, "label", "")).lower()
        text = " ".join(str(getattr(item, "text", "") or "").split())
        if label.endswith("title") and "section" not in label and 3 < len(text) <= 300:
            return text, "docling layout (title)"

    heading = _HEADING_RE.search(markdown)
    if heading:
        text = " ".join(heading.group(1).split())
        if 3 < len(text) <= 300:
            return text, "first heading"
    return None, None


def _short(value: Any, limit: int = 200) -> str:
    return " ".join(str(value).split())[:limit]
