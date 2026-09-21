"""Getting text out of a PDF.

Papers, filings, regulator notices and board minutes arrive as PDFs, and a
research environment that cannot read one is missing the format most primary
records are published in.

Two extractors, in order of preference:

* ``pypdf``, when it is installed. It handles the font encodings real
  documents use, and is the right answer whenever it is available.
* a small built-in reader, so that ingesting a PDF does not *require* a
  dependency. It decodes Flate-compressed content streams and the text
  operators, which covers PDFs produced from text. It does not resolve
  embedded CMaps, so a document using a CID font comes out as noise.

That last failure is the dangerous one, because noise looks like text to
everything downstream: it would be stored as evidence, indexed, and quoted.
So extraction is gated on legibility. Text that does not look like prose is
discarded with a warning naming the remedy, rather than passed on as a
document whose contents are wrong.

Nothing here executes anything. A PDF can carry JavaScript, embedded files
and launch actions; this reads bytes out of content streams and ignores
every other structure in the file.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field

#: Content streams beyond this are truncated rather than decompressed
#: indefinitely - a decompression bomb is a real thing in this format.
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024

#: In a TJ array, a kerning adjustment more negative than this is a word gap
#: rather than letter spacing. The unit is 1/1000 em.
_WORD_GAP = 180.0

_OBJECT_RE = re.compile(rb"\d+\s+\d+\s+obj\b(.*?)\bendobj", re.DOTALL)
_STREAM_RE = re.compile(rb"stream\r?\n(.*?)endstream", re.DOTALL)
_INFO_RE = re.compile(rb"/(Title|Author)\s*\(")

#: Filters whose payload is not text and must not be guessed at.
_BINARY_FILTERS = (b"/DCTDecode", b"/JPXDecode", b"/CCITTFaxDecode", b"/JBIG2Decode")


@dataclass(slots=True)
class PdfText:
    """Extracted text, and an honest account of how it was obtained."""

    text: str = ""
    pages: int = 0
    #: "pypdf", "builtin", or "none" when nothing legible came out.
    method: str = "none"
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def looks_like_prose(text: str, *, minimum: int = 40) -> bool:
    """Whether extracted text is readable, or noise from a font we cannot map.

    A CID-encoded PDF read without its CMap yields characters in roughly the
    right quantity and entirely the wrong identity. The checks are crude on
    purpose: they are meant to catch garbage, not to grade writing.
    """
    stripped = text.strip()
    if len(stripped) < minimum:
        return False
    letters = sum(1 for character in stripped if character.isalpha())
    spaces = stripped.count(" ") + stripped.count("\n")
    if letters / len(stripped) < 0.5:
        return False
    if spaces / len(stripped) < 0.06:
        # Real prose is about one space in six characters. Far fewer means
        # the word boundaries did not survive, if they were ever read.
        return False
    words = [word for word in re.split(r"\s+", stripped) if word]
    if not words:
        return False
    plausible = sum(1 for word in words if re.fullmatch(r"[A-Za-z][A-Za-z'\-]{1,20}", word))
    return plausible / len(words) >= 0.35


def extract_pdf(data: bytes) -> PdfText:
    """Read a PDF's text, preferring a real parser when one is installed."""
    if not data.startswith(b"%PDF"):
        return PdfText(warnings=["not a PDF: the file does not start with %PDF"])

    result = _with_pypdf(data)
    if result is None or not result.text.strip():
        # pypdf is stricter about file structure than this format is in
        # practice; a PDF it refuses may still be one whose content streams
        # can be read, so the simpler reader gets a turn before giving up.
        fallback = _builtin(data)
        if result is None or fallback.text.strip():
            result = fallback

    result.text, dropped = scrub(result.text)
    if dropped:
        result.warnings.append(
            f"{dropped} characters had no text meaning and were dropped; the "
            "document uses font encodings this reader cannot fully map"
        )

    if result.text and not looks_like_prose(result.text):
        return PdfText(
            pages=result.pages,
            method="none",
            title=result.title,
            warnings=[
                *result.warnings,
                "extracted text is not legible - the PDF is probably scanned, or "
                "uses embedded font encodings the built-in reader cannot map. "
                "Install the 'pdf' extra (pypdf) for these documents.",
            ],
        )
    if not result.text:
        result.warnings.append("no text could be extracted; the PDF may be a scan")
    return result


# -- pypdf ---------------------------------------------------------------
def _with_pypdf(data: bytes) -> PdfText | None:
    """Extract with pypdf, or return None when it is not installed."""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return None

    import io

    result = PdfText(method="pypdf")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                result.warnings.append("the PDF is encrypted and could not be opened")
                return result
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # one unreadable page should not lose the rest
                result.warnings.append("a page could not be read")
        result.pages = len(reader.pages)
        result.text = "\n\n".join(part for part in pages if part.strip())
        info = reader.metadata or {}
        result.title = _clean_meta(info.get("/Title"))
        author = _clean_meta(info.get("/Author"))
        result.authors = [author] if author else []
    except Exception as exc:
        result.warnings.append(f"pypdf could not read the file: {exc}")
    return result


def _clean_meta(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


# -- built-in reader -----------------------------------------------------
def _builtin(data: bytes) -> PdfText:
    result = PdfText(method="builtin")
    parts: list[str] = []
    budget = MAX_DECOMPRESSED_BYTES

    for match in _OBJECT_RE.finditer(data):
        body = match.group(1)
        stream = _STREAM_RE.search(body)
        if stream is None:
            continue
        header = body[: stream.start()]
        if any(binary in header for binary in _BINARY_FILTERS):
            continue
        payload = stream.group(1).rstrip(b"\r\n")
        if b"/FlateDecode" in header:
            payload = _inflate(payload, budget)
        budget -= len(payload)
        if budget <= 0:
            result.warnings.append("stopped decompressing: the PDF is implausibly large")
            break
        if b"BT" not in payload or (b"Tj" not in payload and b"TJ" not in payload):
            continue
        text = _content_stream_text(payload)
        if text.strip():
            parts.append(text)

    result.pages = len(parts)
    result.text = "\n\n".join(parts)
    result.title = _builtin_info(data)
    if not parts and b"/Encrypt" in data:
        result.warnings.append("the PDF appears to be encrypted")
    return result


def _inflate(payload: bytes, budget: int) -> bytes:
    for candidate in (payload, payload.lstrip(b"\r\n")):
        try:
            return zlib.decompressobj().decompress(candidate, max(budget, 0))
        except zlib.error:
            continue
    return b""


def _builtin_info(data: bytes) -> str | None:
    """The /Title from a document information dictionary, when it has one."""
    match = _INFO_RE.search(data)
    if match is None or match.group(1) != b"Title":
        return None
    value, _ = _read_literal_string(data, match.end() - 1)
    title = _decode_pdf_bytes(value).strip()
    return title or None


def _content_stream_text(stream: bytes) -> str:
    """Read the text-showing operators out of one content stream."""
    out: list[str] = []
    operands: list[tuple[str, object]] = []
    position = 0
    length = len(stream)

    while position < length:
        character = stream[position : position + 1]
        if character == b"(":
            value, position = _read_literal_string(stream, position)
            operands.append(("string", value))
        elif character == b"<" and stream[position + 1 : position + 2] != b"<":
            value, position = _read_hex_string(stream, position)
            operands.append(("string", value))
        elif character in b"[] \t\r\n":
            # Array delimiters and whitespace carry nothing on their own.
            position += 1
        elif character == b"<" or character == b">":
            # Dictionary delimiters: inline image parameters and the like.
            position += 2
        elif character == b"/":
            _, position = _read_token(stream, position + 1)
        else:
            token, position = _read_token(stream, position)
            if not token:
                position += 1
                continue
            if _is_number(token):
                operands.append(("number", float(token)))
                continue
            out.append(_apply_operator(token.decode("latin-1"), operands))
            operands = []

    return "".join(out)


def _apply_operator(operator: str, operands: list[tuple[str, object]]) -> str:
    strings = [value for kind, value in operands if kind == "string"]
    if operator in ("Tj", "'", '"'):
        prefix = "\n" if operator in ("'", '"') else ""
        return prefix + (_decode_pdf_bytes(strings[-1]) if strings else "")  # type: ignore[arg-type]
    if operator == "TJ":
        pieces: list[str] = []
        for kind, value in operands:
            if kind == "string":
                pieces.append(_decode_pdf_bytes(value))  # type: ignore[arg-type]
            elif isinstance(value, float) and value <= -_WORD_GAP:
                pieces.append(" ")
        return "".join(pieces)
    if operator in ("Td", "TD", "T*", "Tm", "ET"):
        return "\n"
    return ""


def _is_number(token: bytes) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _read_token(data: bytes, position: int) -> tuple[bytes, int]:
    start = position
    while position < len(data) and data[position : position + 1] not in b" \t\r\n[]()<>/%":
        position += 1
    return data[start:position], max(position, start + 1) if position == start else position


_ESCAPES = {
    b"n": "\n", b"r": "\r", b"t": "\t", b"b": "\b", b"f": "\f",
    b"(": "(", b")": ")", b"\\": "\\",
}


def _read_literal_string(data: bytes, position: int) -> tuple[bytes, int]:
    """Read a ``(...)`` string, honouring escapes and nested parentheses."""
    position += 1  # the opening parenthesis
    depth = 1
    out = bytearray()
    while position < len(data):
        character = data[position : position + 1]
        if character == b"\\":
            nxt = data[position + 1 : position + 2]
            if nxt in _ESCAPES:
                out.extend(_ESCAPES[nxt].encode("latin-1"))
                position += 2
                continue
            if nxt.isdigit():
                digits = data[position + 1 : position + 4]
                octal = bytes(digit for digit in digits if 0x30 <= digit <= 0x37)
                if octal:
                    out.append(int(octal, 8) & 0xFF)
                    position += 1 + len(octal)
                    continue
            if nxt in (b"\n", b"\r"):  # line continuation
                position += 2
                continue
            position += 2
            continue
        if character == b"(":
            depth += 1
        elif character == b")":
            depth -= 1
            if depth == 0:
                return bytes(out), position + 1
        out.extend(character)
        position += 1
    return bytes(out), position


def _read_hex_string(data: bytes, position: int) -> tuple[bytes, int]:
    end = data.find(b">", position)
    if end == -1:
        return b"", len(data)
    digits = bytes(
        character
        for character in data[position + 1 : end]
        if chr(character) in "0123456789abcdefABCDEF"
    )
    if len(digits) % 2:
        digits += b"0"
    try:
        return bytes.fromhex(digits.decode("ascii")), end + 1
    except ValueError:  # pragma: no cover - filtered above
        return b"", end + 1


def _decode_pdf_bytes(raw: bytes) -> str:
    """Turn a PDF string into text.

    Simple fonts are usually WinAnsi, which is cp1252, so that is tried
    first - it is what makes quotation marks, dashes and ellipses come out
    right rather than as control characters. Two-byte encodings are common
    for text from modern typesetters: where the bytes are UTF-16 they decode
    correctly, and where they are font CIDs nothing here can map them, which
    the legibility gate then rejects rather than storing.
    """
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace")
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


#: Everything a subset font can leave behind that is not text: NULs, the C0
#: range apart from tab and newline, and the C1 range.
_UNMAPPED_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")


def scrub(text: str) -> tuple[str, int]:
    """Drop characters that are not text, and say how many there were.

    A subsetted font can map a glyph to a slot with no Unicode meaning - the
    ligature in "firmly" arriving as \\x02 is the common case. Nothing here
    can recover the missing letters, and guessing at them would put words in
    a document that it does not contain. They are removed, and the count is
    kept so the document can be marked as imperfectly extracted rather than
    quietly presented as complete.
    """
    cleaned, dropped = _UNMAPPED_RE.subn("", text)
    return cleaned, dropped
