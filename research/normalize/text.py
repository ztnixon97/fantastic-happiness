"""Text cleaning and title normalisation."""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"[^\S\n]+")
_BLANKS_RE = re.compile(r"\n{3,}")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

#: Boilerplate that outlets append to article bodies; stripped before hashing
#: so a shared wire story is not made to look different by a house footer.
_BOILERPLATE_PATTERNS = (
    re.compile(r"^\s*(sign up|subscribe|share this|advertisement|read more)\b.*$", re.I | re.M),
    re.compile(r"©\s*\d{4}[^\n]*", re.I),
    re.compile(r"all rights reserved[^\n]*", re.I),
)

#: Leading words dropped from title keys: providers inconsistently prefix them.
_TITLE_PREFIXES = re.compile(
    r"^(the|a|an|exclusive|breaking|opinion|analysis|update \d+|updated|video|watch|live)\s+",
    re.IGNORECASE,
)


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS_RE.sub("\n\n", text).strip()


def clean_text(text: str | None, *, strip_boilerplate: bool = True) -> str:
    """Normalise whitespace and optionally drop common site boilerplate."""
    cleaned = normalize_whitespace(text)
    if strip_boilerplate and cleaned:
        for pattern in _BOILERPLATE_PATTERNS:
            cleaned = pattern.sub("", cleaned)
        cleaned = normalize_whitespace(cleaned)
    return cleaned


def normalize_title(title: str | None) -> str:
    """Case-folded, punctuation-light title for display-independent compare."""
    if not title:
        return ""
    text = unicodedata.normalize("NFKD", title)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().strip()
    # Drop a trailing outlet suffix: "Headline - Reuters", "Headline | BBC".
    text = re.sub(r"\s+[|–—-]\s+[^|–—-]{1,40}$", "", text)
    text = _TITLE_PREFIXES.sub("", text)
    return _NON_ALNUM_RE.sub(" ", text).strip()


def title_key(title: str | None) -> str | None:
    """Compact identity key for a title, or ``None`` when too weak to use.

    Very short titles ("Introduction", "Q3 results") collide across unrelated
    documents, so they are refused as identity keys rather than causing bad
    merges.
    """
    normalized = normalize_title(title)
    if not normalized:
        return None
    tokens = normalized.split()
    if len(tokens) < 3 or len(normalized.replace(" ", "")) < 12:
        return None
    return "-".join(tokens)


def truncate(text: str | None, limit: int, *, suffix: str = "…") -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix
