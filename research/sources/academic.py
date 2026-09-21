"""Shared machinery for academic providers.

Four providers describe the same literature four different ways. This module
holds the translation that would otherwise be copied into each adapter:
abstracts, dates, author names and the peer-reviewed/preprint distinction.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable

from research.models.common import Provenance, RetrievalMethod, SourceFamily, SourceType
from research.models.evidence import EvidenceDocument
from research.models.query import SearchHit
from research.normalize.doi import normalize_arxiv_id, normalize_doi
from research.normalize.text import normalize_whitespace
from research.sources.base import SourceAdapter
from research.sources.http import SafeHttpClient

#: Venue/type markers that mean 'not peer reviewed (yet)'.
PREPRINT_MARKERS = (
    "arxiv", "biorxiv", "medrxiv", "ssrn", "preprint", "research square",
    "chemrxiv", "osf", "techrxiv", "authorea",
)

_JATS_TAG_RE = re.compile(r"<[^>]+>")


def inverted_index_to_text(
    index: dict[str, list[int]] | None, *, max_words: int = 4000
) -> str | None:
    """Rebuild an abstract from OpenAlex's inverted index."""
    if not index:
        return None
    positions: list[tuple[int, str]] = []
    for word, slots in index.items():
        if not isinstance(slots, list):
            continue
        for slot in slots:
            if isinstance(slot, int) and 0 <= slot < max_words:
                positions.append((slot, word))
    if not positions:
        return None
    positions.sort()
    return normalize_whitespace(" ".join(word for _, word in positions)) or None


def strip_jats(abstract: str | None) -> str | None:
    """Crossref abstracts arrive as JATS XML; keep the prose."""
    if not abstract:
        return None
    text = normalize_whitespace(_JATS_TAG_RE.sub(" ", abstract))
    if text.lower().startswith("abstract "):
        text = text[len("abstract "):]
    return text or None


def date_from_parts(parts: Any) -> datetime | None:
    """Crossref ``date-parts`` -> datetime, with missing components as 1."""
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
        return None
    values = [value for value in parts[0][:3] if isinstance(value, int)]
    if not values:
        return None
    year = values[0]
    month = values[1] if len(values) > 1 else 1
    day = values[2] if len(values) > 2 else 1
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_iso_date(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def classify_work(
    *,
    work_type: str | None = None,
    venue: str | None = None,
    doi: str | None = None,
    url: str | None = None,
) -> SourceType:
    """Peer-reviewed or preprint.

    A conservative mapping: anything that looks like a preprint server is
    labelled a preprint, since treating an unreviewed result as reviewed is
    the costlier error. The label describes the publication route only; it is
    not a judgement about quality.
    """
    haystack = " ".join(
        part.lower() for part in (work_type or "", venue or "", doi or "", url or "") if part
    )
    if any(marker in haystack for marker in PREPRINT_MARKERS):
        return SourceType.ACADEMIC_PREPRINT
    if (work_type or "").lower() in {"posted-content", "preprint", "submitted"}:
        return SourceType.ACADEMIC_PREPRINT
    return SourceType.ACADEMIC_PEER_REVIEWED


def author_names(values: Iterable[Any], *, limit: int = 50) -> list[str]:
    """Normalise whatever shape a provider uses for authors into plain names."""
    names: list[str] = []
    for value in values:
        name: str | None = None
        if isinstance(value, str):
            name = value
        elif isinstance(value, dict):
            nested = value.get("author")
            if isinstance(nested, dict):
                name = nested.get("display_name") or nested.get("name")
            if not name:
                name = value.get("name") or value.get("display_name")
            if not name:
                given = value.get("given") or ""
                family = value.get("family") or ""
                name = f"{given} {family}".strip() or None
        if name:
            cleaned = normalize_whitespace(name)
            if cleaned and cleaned not in names:
                names.append(cleaned)
        if len(names) >= limit:
            break
    return names


class AcademicAdapter(SourceAdapter):
    """Base class for the academic providers.

    Adapters translate; they do not persist, log or charge budgets. That
    happens in :mod:`research.operations`, the only layer that knows about an
    investigation.
    """

    family = SourceFamily.ACADEMIC
    source_type = SourceType.ACADEMIC_PEER_REVIEWED
    base_url: str = ""

    def __init__(
        self,
        client: SafeHttpClient,
        *,
        base_url: str | None = None,
        contact_email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.client = client
        self.base_url = (base_url or self.base_url).rstrip("/")
        self.contact_email = contact_email
        self._api_key = api_key

    def _provenance(
        self,
        *,
        endpoint: str,
        method: RetrievalMethod = RetrievalMethod.SEARCH,
        parent_document_id: str | None = None,
        requested_url: str | None = None,
    ) -> Provenance:
        return Provenance(
            provider=self.name,
            retrieval_method=method,
            provider_endpoint=endpoint,
            requested_url=requested_url,
            parent_document_id=parent_document_id,
        )

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        """Promote a hit to a document.

        Academic search responses already carry the record: title, authors,
        venue, date, identifiers and usually an abstract. Re-requesting it
        would spend a provider call to learn nothing, so the default is to
        normalise what search returned. Adapters that can offer more override
        this.
        """
        from research.normalize.document import document_from_hit

        return document_from_hit(
            hit,
            provenance=self._provenance(
                endpoint=str(hit.raw.get("_endpoint") or self.base_url),
                method=RetrievalMethod.coerce(
                    hit.raw.get("_retrieval_method"), RetrievalMethod.SEARCH
                ),
                requested_url=hit.url,
            ),
        )

    @staticmethod
    def identifiers_of(raw: dict[str, Any]) -> dict[str, str]:
        """Pull the authoritative identifiers a record carries."""
        identifiers: dict[str, str] = {}
        doi = normalize_doi(raw.get("doi"))
        if doi:
            identifiers["doi"] = doi
        arxiv = normalize_arxiv_id(raw.get("arxiv_id") or raw.get("arxiv"))
        if arxiv:
            identifiers["arxiv_id"] = arxiv
        for key in ("pmid", "pmcid", "openalex_id", "semantic_scholar_id"):
            if raw.get(key):
                identifiers[key] = str(raw[key])
        return identifiers
