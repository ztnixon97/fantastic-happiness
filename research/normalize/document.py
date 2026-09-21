"""Assembling normalised evidence documents.

Sources hand back whatever shape their API has. Everything downstream - the
store, deduplication, the claim graph, the report - sees only
:class:`~research.models.evidence.EvidenceDocument`.

This lives in :mod:`research.normalize` rather than in the acquisition layer
because it is pure translation: no storage, no network, no policy. Source
adapters need it, and a source adapter must never depend on the layer that
persists what it returns.

Documents leave this module with an empty ``id``: identity belongs to the
store, which is the only component that knows whether this material is
already held.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from research.models.common import (
    Provenance,
    SourceFamily,
    SourceType,
    utcnow,
)
from research.models.evidence import EvidenceDocument
from research.models.query import SearchHit
from research.normalize.doi import normalize_arxiv_id, normalize_doi
from research.normalize.fingerprint import content_hash, simhash
from research.normalize.html import ExtractedPage
from research.normalize.text import clean_text, title_key, truncate
from research.normalize.urls import canonicalize_url, registrable_domain, url_host

#: Wire services whose copy is routinely republished verbatim. A credit line
#: is the cheapest reliable signal that an article is not independent
#: reporting - ten outlets carrying one AP story are one source, not ten.
WIRE_SERVICES = {
    "reuters": "Reuters",
    "associated press": "Associated Press",
    "ap": "Associated Press",
    "afp": "Agence France-Presse",
    "agence france-presse": "Agence France-Presse",
    "bloomberg": "Bloomberg",
    "pa media": "PA Media",
    "press association": "PA Media",
    "dpa": "dpa",
    "kyodo": "Kyodo News",
    "xinhua": "Xinhua",
    "efe": "EFE",
    "ansa": "ANSA",
    "tass": "TASS",
    "ani": "ANI",
    "pti": "Press Trust of India",
}

_WIRE_NAMES = "|".join(re.escape(name) for name in WIRE_SERVICES)
_WIRE_CREDIT_RE = re.compile(
    # Either a parenthesised credit - "WASHINGTON (Reuters) - ..." - or a
    # dashed byline at the start of a line: "By Associated Press - ...".
    rf"\(\s*({_WIRE_NAMES})\s*\)"
    rf"|(?:^|\n)\s*(?:by\s+)?({_WIRE_NAMES})\s*[–—-]\s",
    re.IGNORECASE,
)


def detect_wire_service(text: str | None, publisher: str | None = None) -> str | None:
    """Identify a wire-service credit in an article's opening, if present."""
    if publisher:
        canonical = WIRE_SERVICES.get(publisher.strip().casefold())
        if canonical:
            return canonical
    if not text:
        return None
    match = _WIRE_CREDIT_RE.search(text[:600])
    if not match:
        return None
    name = (match.group(1) or match.group(2) or "").strip().casefold()
    return WIRE_SERVICES.get(name)


def build_document(
    *,
    provider: str,
    source_type: SourceType,
    source_family: SourceFamily,
    provenance: Provenance,
    title: str | None = None,
    text: str | None = None,
    abstract: str | None = None,
    url: str | None = None,
    external_id: str | None = None,
    authors: list[str] | None = None,
    published_at: datetime | None = None,
    publisher: str | None = None,
    language: str | None = None,
    doi: str | None = None,
    metadata: dict[str, Any] | None = None,
    investigation_id: str | None = None,
    max_text_characters: int = 400_000,
) -> EvidenceDocument:
    """Assemble a normalised document. ``id`` is left for the store to assign."""
    canonical_url = canonicalize_url(url)
    clean_body = truncate(clean_text(text), max_text_characters) if text else None
    clean_abstract = clean_text(abstract) if abstract else None
    normalized_doi = normalize_doi(doi) or normalize_doi(canonical_url)
    meta = dict(metadata or {})

    arxiv_id = (
        meta.get("arxiv_id")
        or normalize_arxiv_id(external_id)
        or normalize_arxiv_id(canonical_url)
    )
    if arxiv_id:
        meta["arxiv_id"] = arxiv_id

    wire = detect_wire_service(clean_body or clean_abstract, publisher)
    if wire:
        meta.setdefault("wire_service", wire)

    document = EvidenceDocument(
        id="",
        investigation_id=investigation_id,
        source_type=source_type,
        source_family=source_family,
        provider=provider,
        external_id=external_id,
        canonical_url=canonical_url,
        title=clean_text(title) or None,
        authors=[a for a in (authors or []) if a][:50],
        published_at=published_at,
        fetched_at=utcnow(),
        text=clean_body,
        abstract=clean_abstract,
        metadata=meta,
        content_hash=content_hash(title, clean_body or clean_abstract),
        provenance=provenance,
        doi=normalized_doi,
        canonical_host=url_host(canonical_url),
        title_key=title_key(title),
        simhash=simhash(clean_body or clean_abstract),
        publisher=publisher or (
            registrable_domain(canonical_url)
            if source_family is SourceFamily.NEWS
            else None
        ),
        language=language,
    )
    document.independence_key = None  # assigned at persistence time
    return document


def document_from_hit(
    hit: SearchHit,
    *,
    provenance: Provenance,
    investigation_id: str | None = None,
    text: str | None = None,
    max_text_characters: int = 400_000,
) -> EvidenceDocument:
    """Promote a search hit to a document, optionally with fetched body text."""
    metadata = {key: value for key, value in hit.raw.items() if not key.startswith("_")}
    return build_document(
        provider=hit.provider,
        source_type=hit.source_type,
        source_family=hit.source_family,
        provenance=provenance,
        title=hit.title,
        text=text,
        abstract=hit.snippet,
        url=hit.url,
        external_id=hit.external_id,
        authors=list(hit.authors),
        published_at=hit.published_at,
        publisher=hit.publisher,
        language=hit.language,
        doi=hit.doi,
        metadata=metadata,
        investigation_id=investigation_id,
        max_text_characters=max_text_characters,
    )


def document_from_page(
    page: ExtractedPage,
    *,
    url: str,
    provider: str,
    provenance: Provenance,
    source_type: SourceType = SourceType.WEB_PAGE,
    source_family: SourceFamily = SourceFamily.WEB,
    investigation_id: str | None = None,
    max_text_characters: int = 400_000,
) -> EvidenceDocument:
    """Build a document from a fetched HTML page.

    A publisher-declared canonical URL wins over the URL we happened to
    request: that is how an AMP mirror and a syndicated copy resolve to the
    same underlying article.
    """
    canonical = canonicalize_url(page.canonical_url) or canonicalize_url(url)
    document = build_document(
        provider=provider,
        source_type=source_type,
        source_family=source_family,
        provenance=provenance,
        title=page.title,
        text=page.text,
        abstract=page.description,
        url=canonical,
        authors=list(page.authors),
        published_at=page.published_at,
        publisher=page.site_name,
        language=page.language,
        metadata={
            "requested_url": url,
            "declared_canonical_url": page.canonical_url,
            "outbound_links": page.links[:50],
        },
        investigation_id=investigation_id,
        max_text_characters=max_text_characters,
    )
    if page.canonical_url and canonicalize_url(page.canonical_url) != canonicalize_url(url):
        document.metadata["canonical_differs_from_requested"] = True
    return document


def enrich(original: EvidenceDocument, incoming: EvidenceDocument) -> bool:
    """Fill gaps in a held document from a newly seen copy of the same thing.

    Returns ``True`` if anything changed. Existing values are never
    overwritten: the first acquisition keeps its provenance, and the second
    only contributes what was missing.
    """
    changed = False
    for field_name in ("doi", "abstract", "published_at", "publisher", "language", "canonical_url"):
        if getattr(original, field_name) is None and getattr(incoming, field_name) is not None:
            setattr(original, field_name, getattr(incoming, field_name))
            changed = True
    if not original.text and incoming.text:
        original.text = incoming.text
        original.content_hash = content_hash(original.title, original.text)
        original.simhash = simhash(original.text)
        changed = True
    if not original.authors and incoming.authors:
        original.authors = list(incoming.authors)
        changed = True
    for key, value in incoming.metadata.items():
        if key not in original.metadata:
            original.metadata[key] = value
            changed = True
    # Record every provider that independently held this same record; the
    # count is useful when judging provider coverage, never as corroboration.
    seen = original.metadata.setdefault("also_provided_by", [])
    if incoming.provider != original.provider and incoming.provider not in seen:
        seen.append(incoming.provider)
        changed = True

    # And every address it turned out to be reachable at. A story that
    # resolves to one canonical document from five URLs is one story - and
    # the five addresses are themselves evidence of how far it spread.
    requested = (
        incoming.provenance.requested_url if incoming.provenance else None
    ) or incoming.canonical_url
    if requested and requested != original.canonical_url:
        addresses = original.metadata.setdefault("also_retrieved_from", [])
        if requested not in addresses:
            addresses.append(requested)
            changed = True
    return changed
