"""Finding the part of a document that bears on the question.

Reading a document used to mean reading its first four thousand characters,
whatever was being asked. For a press release that is the document; for a
twenty-nine page filing it is the cover page, and the paragraph that settles
the claim is on page eleven where nobody looks.

This splits a document into passages and ranks them against a question, the
same way the corpus is ranked against a query: BM25 over the document's own
passages, optionally a vector opinion, fused by reciprocal rank. It is the
inside-a-document counterpart of :mod:`research.retrieval.search`.

Two properties matter more than the ranking:

* **Passages are verbatim, with offsets.** A passage is a slice of the
  stored text, never a rewrite of it, so a quotation taken from one still
  passes the excerpt check that refuses fabricated quotes. Summarising here
  would quietly turn evidence into analysis.
* **Omission is visible.** A worker reading passages is told it is reading
  parts of a longer document and where they came from, because a document
  read in fragments can mislead in ways a whole one cannot.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from research.retrieval.fusion import reciprocal_rank_fusion

#: Long enough to carry an argument, short enough that several fit in a
#: prompt. Paragraphs are preferred to this number where they are close.
TARGET_CHARACTERS = 900
MIN_CHARACTERS = 200

_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[\w][\w'-]*", re.UNICODE)

#: Suffixes folded before matching. The corpus index uses FTS5's porter
#: stemmer; this is a much cruder stand-in, and it is here because without
#: it "reactor costs" in a question would not match "capital cost" in a
#: filing, and the two indexes would disagree about what a word is.
_SUFFIXES = ("iness", "ingly", "ness", "ing", "ies", "ied", "es", "ed", "ly", "s")


def fold(word: str) -> str:
    """A crude stem: enough to match singular against plural, no more."""
    lowered = word.lower()
    for suffix in _SUFFIXES:
        if len(lowered) >= len(suffix) + 4 and lowered.endswith(suffix):
            stem = lowered[: -len(suffix)]
            return stem[:-1] if suffix == "ies" and stem.endswith("i") else stem + (
                "y" if suffix == "ies" else ""
            )
    return lowered


@dataclass(slots=True)
class Passage:
    """A verbatim slice of a document, with where it came from."""

    text: str
    start: int
    end: int
    index: int = 0
    score: float = 0.0
    #: Which rankers found it, and where they placed it.
    ranks: dict[str, int] = field(default_factory=dict)

    def locate(self) -> str:
        return f"characters {self.start}-{self.end}"


def split_passages(
    text: str, *, target: int = TARGET_CHARACTERS, minimum: int = MIN_CHARACTERS
) -> list[Passage]:
    """Cut a document into passages on its own boundaries.

    Paragraphs first, sentences when a paragraph is too long to be one
    passage, and a hard cut only when a single sentence is longer than the
    target. Offsets are into the original string, so every passage can be
    located exactly.
    """
    if not text:
        return []

    passages: list[Passage] = []
    buffer: list[tuple[str, int]] = []
    buffered = 0

    def flush() -> None:
        nonlocal buffer, buffered
        if not buffer:
            return
        start = buffer[0][1]
        end = buffer[-1][1] + len(buffer[-1][0])
        passages.append(Passage(text=text[start:end], start=start, end=end))
        buffer = []
        buffered = 0

    for chunk, offset in _blocks(text, target):
        if buffered and buffered + len(chunk) > target:
            flush()
        buffer.append((chunk, offset))
        buffered += len(chunk)
        if buffered >= target:
            flush()
    flush()

    # A trailing scrap is part of the passage before it, not a passage.
    if len(passages) > 1 and len(passages[-1].text) < minimum:
        last = passages.pop()
        previous = passages[-1]
        passages[-1] = Passage(
            text=text[previous.start : last.end], start=previous.start, end=last.end
        )
    for position, passage in enumerate(passages):
        passage.index = position
    return passages


def _blocks(text: str, target: int) -> list[tuple[str, int]]:
    """Paragraphs, split into sentences when they are too long."""
    blocks: list[tuple[str, int]] = []
    position = 0
    for paragraph in _PARAGRAPH_RE.split(text):
        if not paragraph.strip():
            position = text.find(paragraph, position) + len(paragraph)
            continue
        start = text.find(paragraph, position)
        if start < 0:  # pragma: no cover - split always yields substrings
            start = position
        position = start + len(paragraph)
        if len(paragraph) <= target:
            blocks.append((paragraph, start))
            continue
        cursor = start
        for sentence in _SENTENCE_RE.split(paragraph):
            if not sentence:
                continue
            found = text.find(sentence, cursor)
            if found < 0:  # pragma: no cover
                found = cursor
            cursor = found + len(sentence)
            if len(sentence) <= target:
                blocks.append((sentence, found))
                continue
            # One sentence longer than a passage: cut it on word boundaries.
            for piece_start in range(0, len(sentence), target):
                piece = sentence[piece_start : piece_start + target]
                blocks.append((piece, found + piece_start))
    return blocks


def score_lexically(passages: Sequence[Passage], query: str) -> list[tuple[int, float]]:
    """BM25 of each passage against the query, within this document.

    The statistics are the document's own: a term common to every passage
    tells you nothing about which passage to read, which is exactly what
    inverse document frequency encodes.
    """
    terms = [fold(term) for term in _WORD_RE.findall(query or "") if len(term) > 1]
    if not terms or not passages:
        return []

    tokenised = [[fold(word) for word in _WORD_RE.findall(p.text)] for p in passages]
    lengths = [len(tokens) for tokens in tokenised]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    if not average:
        return []

    k1, b = 1.5, 0.75
    total = len(passages)
    document_frequency = {
        term: sum(1 for tokens in tokenised if term in tokens) for term in set(terms)
    }
    scored: list[tuple[int, float]] = []
    for position, tokens in enumerate(tokenised):
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        score = 0.0
        for term in set(terms):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            containing = document_frequency[term]
            # The unsmoothed form, which goes negative for a term in most of
            # the passages. That is the point: "the" appears everywhere and
            # tells you nothing about which passage to read, and a smoothed
            # idf keeps it very slightly positive - enough to float a cover
            # page into the results on nothing but stopwords.
            idf = math.log((total - containing + 0.5) / (containing + 0.5))
            if idf <= 0:
                continue
            denominator = frequency + k1 * (1 - b + b * lengths[position] / average)
            score += idf * (frequency * (k1 + 1) / denominator)
        if score > 0:
            scored.append((position, score))
    scored.sort(key=lambda entry: (-entry[1], entry[0]))
    return scored


async def select_passages(
    text: str,
    query: str,
    *,
    limit: int = 4,
    embedder: Any = None,
    target: int = TARGET_CHARACTERS,
) -> list[Passage]:
    """The passages of ``text`` that bear on ``query``, best first.

    Returns nothing when the query matches nothing, which the caller should
    treat as "read the document from the top" rather than as "this document
    is irrelevant" - a short document may simply not repeat the question's
    words.
    """
    passages = split_passages(text, target=target)
    if len(passages) <= 1 or not (query or "").strip():
        return passages[:limit]

    rankings: dict[str, Sequence[str]] = {}
    lexical = score_lexically(passages, query)
    if lexical:
        rankings["lexical"] = [str(position) for position, _ in lexical]

    if embedder is not None:
        vector = await _score_by_vector(passages, query, embedder)
        if vector:
            rankings["vector"] = [str(position) for position, _ in vector]

    if not rankings:
        return []

    fused = reciprocal_rank_fusion(rankings, weights={"lexical": 1.0, "vector": 0.9})
    chosen: list[Passage] = []
    for hit in fused[:limit]:
        passage = passages[int(hit.document_id)]
        passage.score = hit.score
        passage.ranks = dict(hit.ranks)
        chosen.append(passage)
    # Read in the order the document is written, not the order it was ranked:
    # passages out of sequence read as a different argument than the one made.
    chosen.sort(key=lambda passage: passage.start)
    return chosen


async def _score_by_vector(
    passages: Sequence[Passage], query: str, embedder: Any
) -> list[tuple[int, float]]:
    from research.retrieval.embeddings import cosine

    try:
        vectors = await embedder.embed([query, *(passage.text for passage in passages)])
    except Exception:
        return []
    if len(vectors) != len(passages) + 1 or not vectors[0]:
        return []
    query_vector, rest = vectors[0], vectors[1:]
    scored = [
        (position, cosine(query_vector, vector))
        for position, vector in enumerate(rest)
        if vector
    ]
    scored.sort(key=lambda entry: (-entry[1], entry[0]))
    return [entry for entry in scored if entry[1] > 0]
