"""Content hashing and near-duplicate fingerprints.

Exact hashing catches byte-identical republication. SimHash catches the more
common case: the same wire story with a different headline, a house standfirst
and a newsletter footer.
"""

from __future__ import annotations

import hashlib
import re

from research.normalize.text import normalize_whitespace

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SIMHASH_BITS = 64
_MASK = (1 << _SIMHASH_BITS) - 1


def content_hash(*parts: str | None) -> str:
    """Stable ``sha256:`` hash over normalised text parts.

    Whitespace and case differences must not produce different hashes, or
    trivially reformatted copies would each count as independent evidence.
    """
    digest = hashlib.sha256()
    for index, part in enumerate(parts):
        if index:
            # A field separator, so ("ab", "c") and ("a", "bc") differ.
            digest.update(b"\x1f")
        digest.update(_flatten(part).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def _flatten(text: str | None) -> str:
    """Case-fold and collapse every whitespace run, newlines included.

    Reflowing a page - different paragraph breaks, different line wrapping -
    must not make a republished copy look like new material.
    """
    return " ".join(normalize_whitespace(text).casefold().split())


def tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.casefold())


def shingles(text: str | None, size: int = 5) -> set[str]:
    """Word n-grams, the unit of comparison for near-duplicate detection."""
    words = tokens(text)
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def containment(subset: set[str], superset: set[str]) -> float:
    """Fraction of ``subset`` that also appears in ``superset``.

    Jaccard punishes length differences, which is wrong for the case that
    matters here: an article that reproduces another one and adds house
    content. Containment asks the question directly - how much of this text
    came from that text.
    """
    if not subset or not superset:
        return 0.0
    return len(subset & superset) / len(subset)


def simhash(text: str | None, *, shingle_size: int = 4) -> int | None:
    """64-bit SimHash over shingles, or ``None`` for text too short to judge."""
    grams = shingles(text, shingle_size)
    if not grams or len(tokens(text)) < shingle_size * 2:
        return None
    vector = [0] * _SIMHASH_BITS
    for gram in grams:
        digest = int.from_bytes(
            hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(_SIMHASH_BITS):
            vector[bit] += 1 if digest >> bit & 1 else -1
    value = 0
    for bit in range(_SIMHASH_BITS):
        if vector[bit] > 0:
            value |= 1 << bit
    return value & _MASK


def hamming_distance(left: int, right: int) -> int:
    return ((left ^ right) & _MASK).bit_count()


def simhash_bands(value: int | None, *, bands: int = 4) -> list[str]:
    """Split a SimHash into indexable bands.

    Two fingerprints within a few bits almost always share at least one band,
    which turns near-duplicate lookup into an indexed query instead of a scan
    over every document in the investigation.
    """
    if value is None:
        return []
    width = _SIMHASH_BITS // bands
    mask = (1 << width) - 1
    return [
        f"{index}:{(value >> (index * width)) & mask:0{width // 4}x}"
        for index in range(bands)
    ]
