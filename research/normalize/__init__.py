"""Deterministic normalisation.

Nothing in this package calls a model. URL canonicalisation, DOI handling,
hashing and fingerprinting are ordinary code problems and are solved with
ordinary code.
"""

from research.normalize.document import (
    build_document,
    detect_wire_service,
    document_from_hit,
    document_from_page,
    enrich,
)
from research.normalize.doi import (
    doi_to_url,
    extract_doi,
    normalize_arxiv_id,
    normalize_doi,
)
from research.normalize.fingerprint import (
    containment,
    content_hash,
    hamming_distance,
    jaccard,
    shingles,
    simhash,
    simhash_bands,
)
from research.normalize.text import (
    clean_text,
    normalize_title,
    normalize_whitespace,
    title_key,
    truncate,
)
from research.normalize.urls import (
    canonicalize_url,
    registrable_domain,
    url_host,
    url_identity_key,
)

__all__ = [
    "build_document",
    "canonicalize_url",
    "clean_text",
    "containment",
    "content_hash",
    "detect_wire_service",
    "document_from_hit",
    "document_from_page",
    "doi_to_url",
    "enrich",
    "extract_doi",
    "hamming_distance",
    "jaccard",
    "normalize_arxiv_id",
    "normalize_doi",
    "normalize_title",
    "normalize_whitespace",
    "registrable_domain",
    "shingles",
    "simhash",
    "simhash_bands",
    "title_key",
    "truncate",
    "url_host",
    "url_identity_key",
]
