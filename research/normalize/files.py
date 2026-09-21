"""Reading a local file into the shape the rest of the system works with.

Pure translation, like everything else in this package: bytes and a file
name in, title and text out. No filesystem walking, no storage, no policy -
those belong to the acquisition layer, which is the only place allowed to
decide *which* files an investigation may read.

Supported kinds are the ones research actually arrives in: PDFs, plain text,
Markdown, HTML and structured data. Anything else is refused by name rather
than guessed at, because a document stored with the wrong contents is worse
than one that was never stored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from research.normalize.docling_reader import (
    DOCLING_ONLY_EXTENSIONS,
    DoclingConverter,
    DoclingResult,
)
from research.normalize.html import extract_page
from research.normalize.pdf import extract_pdf, is_noise, scrub
from research.normalize.text import clean_text

#: Extension to kind. Deliberately a short list.
EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "text",
    ".text": "text",
    ".log": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".json": "json",
    ".csv": "text",
    ".tsv": "text",
    ".xml": "html",
}

MEDIA_TYPES = {
    "pdf": "application/pdf",
    "text": "text/plain",
    "markdown": "text/markdown",
    "html": "text/html",
    "json": "application/json",
    "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "book": "application/epub+zip",
    "image": "image/*",
}

#: Magic bytes for the formats only Docling can read. An Office file is a
#: zip, so its extension is all that separates a .docx from a .xlsx.
_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF8", "image"),
    (b"BM", "image"),
    (b"II*\x00", "image"),
    (b"MM\x00*", "image"),
)


def readable_kinds(*, docling: bool = False) -> dict[str, str]:
    """Extension to kind, for the readers actually available."""
    if not docling:
        return dict(EXTENSIONS)
    return {**EXTENSIONS, **DOCLING_ONLY_EXTENSIONS}

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_MD_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass(slots=True)
class ExtractedFile:
    text: str = ""
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    kind: str = "text"
    media_type: str | None = None
    warnings: list[str] = field(default_factory=list)
    #: Anything the extractor learned that is worth keeping, e.g. page count.
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def detect_kind(filename: str, data: bytes, *, docling: bool = False) -> str | None:
    """Decide what a file is, from its magic bytes first and its name second.

    Content wins over extension: a ``.txt`` that begins ``%PDF`` is a PDF,
    whatever it is called. ``docling`` widens the answer to the formats that
    cannot be read without it - returning a kind nothing can read would only
    produce a failure further along.
    """
    if data.startswith(b"%PDF"):
        return "pdf"
    if docling:
        for magic, kind in _IMAGE_MAGIC:
            if data.startswith(magic):
                return kind
    head = data[:1024].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        return "html"
    suffix = filename.rsplit(".", 1)
    extension = f".{suffix[-1].lower()}" if len(suffix) > 1 else ""
    return readable_kinds(docling=docling).get(extension)


def extract_file(
    filename: str,
    data: bytes,
    *,
    kind: str | None = None,
    max_characters: int = 400_000,
    docling: DoclingConverter | None = None,
    ocr: str = "off",
) -> ExtractedFile:
    """Turn a file's bytes into title and text, or say why it could not be."""
    kind = kind or detect_kind(filename, data, docling=docling is not None)
    if kind is None:
        readable = sorted(set(readable_kinds(docling=docling is not None).values()))
        extra = (
            ""
            if docling is not None
            else " Enabling docling adds Word, PowerPoint, Excel and images."
        )
        return ExtractedFile(
            warnings=[
                f"unsupported file type: {filename}. Readable kinds are "
                + ", ".join(readable)
                + "." + extra
            ]
        )

    if kind == "pdf":
        return _named(
            _from_pdf(data, filename, max_characters=max_characters, docling=docling, ocr=ocr),
            filename,
        )
    if kind in set(DOCLING_ONLY_EXTENSIONS.values()):
        return _named(
            _from_docling_only(
                data, filename, kind, max_characters=max_characters, docling=docling, ocr=ocr
            ),
            filename,
        )
    text = _decode(data)
    if kind == "html":
        return _from_html(text, max_characters=max_characters)
    if kind == "json":
        return _from_json(text, filename, max_characters=max_characters)
    if kind == "markdown":
        return _from_markdown(text, filename, max_characters=max_characters)
    return _from_text(text, filename, max_characters=max_characters)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")



def _named(extracted: ExtractedFile, filename: str) -> ExtractedFile:
    """Fall back to the file's own name when nothing supplied a title.

    A PDF often carries no title at all, and a converter infers one from
    layout or not at all. The name the person gave the file is real
    information, unlike a guess - and where it is used, the document says
    so, because a file name is not a title somebody wrote.
    """
    if extracted.title or not extracted.ok:
        return extracted
    extracted.title = _name_as_title(filename)
    extracted.metadata.setdefault("title_source", "file name")
    return extracted


def _from_pdf(
    data: bytes,
    filename: str,
    *,
    max_characters: int,
    docling: DoclingConverter | None = None,
    ocr: str = "off",
) -> ExtractedFile:
    extracted = extract_pdf(data, filename=filename, docling=docling, ocr=ocr)
    metadata: dict[str, object] = {
        "pdf_pages": extracted.pages,
        "pdf_extractor": extracted.method,
    }
    if extracted.tables:
        metadata["tables"] = extracted.tables
    if extracted.title_source:
        metadata["title_source"] = extracted.title_source
    return ExtractedFile(
        text=clean_text(extracted.text)[:max_characters],
        title=extracted.title,
        authors=list(extracted.authors),
        kind="pdf",
        media_type=MEDIA_TYPES["pdf"],
        warnings=list(extracted.warnings),
        metadata=metadata,
    )


def _from_docling_only(
    data: bytes,
    filename: str,
    kind: str,
    *,
    max_characters: int,
    docling: DoclingConverter | None,
    ocr: str,
) -> ExtractedFile:
    """Word, PowerPoint, Excel, EPUB and images: Docling or nothing.

    An image is a scan by definition - there is no text layer to try first -
    so it goes straight to OCR, and is refused when OCR is switched off
    rather than stored as an empty document.
    """
    if docling is None:
        return ExtractedFile(
            kind=kind,
            warnings=[f"{kind} files need docling, which is not enabled"],
        )
    wants_ocr = kind == "image" or ocr == "always"
    if kind == "image" and ocr == "off":
        return ExtractedFile(
            kind=kind,
            warnings=["an image can only be read by OCR, which is switched off"],
        )

    converted = docling(data, filename, ocr=wants_ocr)
    if converted is None:  # pragma: no cover - converter_for checked this
        return ExtractedFile(kind=kind, warnings=["docling is not installed"])
    if not converted.ok and ocr == "auto" and not wants_ocr:
        # Same escalation as a PDF: nothing readable means there may be
        # nothing to read without OCR.
        retried = docling(data, filename, ocr=True)
        if retried is not None and retried.ok:
            converted = retried

    return _from_converted(converted, kind, max_characters=max_characters)


def _from_converted(
    converted: DoclingResult, kind: str, *, max_characters: int
) -> ExtractedFile:
    text, dropped = scrub(converted.text)
    warnings = list(converted.warnings)
    if is_noise(text):
        return ExtractedFile(
            kind=kind,
            warnings=[*warnings, "the text that came out is not legible and was discarded"],
        )
    if dropped:
        warnings.append(f"{dropped} characters had no text meaning and were dropped")
    if not text.strip():
        warnings.append("no text could be read from this document")
    metadata: dict[str, object] = {"pdf_extractor": converted.method}
    if converted.pages:
        metadata["pdf_pages"] = converted.pages
    if converted.tables:
        metadata["tables"] = converted.tables
    if converted.title_source:
        metadata["title_source"] = converted.title_source
    return ExtractedFile(
        text=clean_text(text)[:max_characters],
        title=converted.title,
        kind=kind,
        media_type=MEDIA_TYPES.get(kind),
        warnings=warnings,
        metadata=metadata,
    )


def _from_html(text: str, *, max_characters: int) -> ExtractedFile:
    page = extract_page(text, max_characters=max_characters)
    return ExtractedFile(
        text=page.text[:max_characters],
        title=page.title,
        authors=list(page.authors),
        kind="html",
        media_type=MEDIA_TYPES["html"],
        metadata={"described_canonical_url": page.canonical_url} if page.canonical_url else {},
    )


def _from_json(text: str, filename: str, *, max_characters: int) -> ExtractedFile:
    """Keep JSON as data: pretty-printed and quotable, never interpreted."""
    try:
        payload = json.loads(text)
    except ValueError as exc:
        return ExtractedFile(
            text=clean_text(text)[:max_characters],
            title=_name_as_title(filename),
            kind="json",
            media_type=MEDIA_TYPES["json"],
            warnings=[f"not valid JSON, stored as text: {exc}"],
        )
    title = None
    if isinstance(payload, dict):
        for key in ("title", "name", "headline"):
            if isinstance(payload.get(key), str):
                title = payload[key]
                break
    return ExtractedFile(
        text=json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)[:max_characters],
        title=title or _name_as_title(filename),
        kind="json",
        media_type=MEDIA_TYPES["json"],
    )


def _from_markdown(text: str, filename: str, *, max_characters: int) -> ExtractedFile:
    metadata: dict[str, object] = {}
    title = None
    authors: list[str] = []

    frontmatter = _FRONTMATTER_RE.match(text)
    if frontmatter:
        fields = _simple_frontmatter(frontmatter.group(1))
        title = fields.get("title")
        author = fields.get("author") or fields.get("authors")
        if author:
            authors = [part.strip() for part in author.split(",") if part.strip()]
        if fields:
            metadata["frontmatter"] = fields
        text = text[frontmatter.end() :]

    if not title:
        heading = _MD_TITLE_RE.search(text)
        title = heading.group(1).strip() if heading else _name_as_title(filename)

    return ExtractedFile(
        text=clean_text(text)[:max_characters],
        title=title,
        authors=authors,
        kind="markdown",
        media_type=MEDIA_TYPES["markdown"],
        metadata=metadata,
    )


def _simple_frontmatter(block: str) -> dict[str, str]:
    """Read ``key: value`` lines out of YAML frontmatter.

    Deliberately not a YAML parser: this reads a note's own header, and the
    only fields wanted from it are strings.
    """
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if line.startswith((" ", "\t", "-")) or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip("'\"")
        if key.strip() and value:
            fields[key.strip().lower()] = value
    return fields


def _from_text(text: str, filename: str, *, max_characters: int) -> ExtractedFile:
    cleaned = clean_text(text)
    title = None
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            # A short opening line is a heading; a long one is the document.
            title = candidate if len(candidate) <= 120 else None
            break
    return ExtractedFile(
        text=cleaned[:max_characters],
        title=title or _name_as_title(filename),
        kind="text",
        media_type=MEDIA_TYPES["text"],
    )


def _name_as_title(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return re.sub(r"[._-]+", " ", stem).strip() or filename
