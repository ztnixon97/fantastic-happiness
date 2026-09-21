"""DOI and arXiv identifier handling.

DOIs are the strongest deduplication signal in academic material: four
providers will return four records for one paper, and the DOI is what makes
them one document.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

#: DOIs are case-insensitive and are compared in lowercase (per DOI handbook).
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)
_DOI_TRAILING = ".,;:)]}>\"'"

_ARXIV_NEW_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_ARXIV_OLD_RE = re.compile(
    r"\b([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b", re.IGNORECASE
)
_ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(?P<id>.+)$", re.IGNORECASE)


def normalize_doi(value: str | None) -> str | None:
    """Return a bare lowercase DOI (``10.1234/abc``) or ``None``.

    Accepts DOIs wrapped as URLs (``https://doi.org/...``), prefixed
    (``doi:10.x``), percent-encoded, or padded with punctuation.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None

    if "://" in text:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower()
        if host in {"doi.org", "dx.doi.org", "www.doi.org", "hdl.handle.net"}:
            text = unquote(parts.path.lstrip("/"))
        else:
            text = unquote(parts.path) or text

    text = re.sub(r"^\s*(doi|DOI)\s*[:=]\s*", "", text).strip()
    match = _DOI_RE.search(text)
    if not match:
        return None
    doi = match.group(1)
    trimmed = doi.rstrip(_DOI_TRAILING)
    # A trailing bracket may belong to the DOI itself (10.1234/abc(1)) or to
    # the surrounding prose ("see 10.1234/abc)."). Restore only what closes
    # an opening bracket inside the DOI.
    for opener, closer in (("(", ")"), ("[", "]")):
        while trimmed.count(opener) > trimmed.count(closer) and doi[len(trimmed) :].startswith(
            closer
        ):
            trimmed = doi[: len(trimmed) + 1]
    return trimmed.lower() or None


def extract_doi(text: str | None, *, limit: int = 20) -> list[str]:
    """Find DOIs inside free text (reference lists, article bodies)."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in _DOI_RE.finditer(text):
        doi = normalize_doi(match.group(1))
        if doi and doi not in seen:
            seen.add(doi)
            found.append(doi)
            if len(found) >= limit:
                break
    return found


def doi_to_url(doi: str | None) -> str | None:
    normalized = normalize_doi(doi)
    return f"https://doi.org/{normalized}" if normalized else None


def normalize_arxiv_id(value: str | None) -> str | None:
    """Return a bare arXiv id without a version suffix, or ``None``.

    ``arXiv:2401.01234v2``, ``https://arxiv.org/abs/2401.01234`` and
    ``10.48550/arXiv.2401.01234`` all normalise to ``2401.01234``.
    """
    if not value:
        return None
    text = str(value).strip()

    doi_match = _ARXIV_DOI_RE.match(text)
    if doi_match:
        text = doi_match.group("id")

    if "://" in text:
        parts = urlsplit(text)
        if "arxiv.org" not in (parts.hostname or "").lower():
            return None
        text = parts.path
        for marker in ("/abs/", "/pdf/", "/html/"):
            if marker in text:
                text = text.split(marker, 1)[1]
                break
        if text.endswith(".pdf"):
            text = text[: -len(".pdf")]

    text = re.sub(r"^\s*arxiv\s*[:/]\s*", "", text, flags=re.IGNORECASE).strip()

    match = _ARXIV_NEW_RE.search(text) or _ARXIV_OLD_RE.search(text)
    if not match:
        return None
    return match.group(1).lower()
