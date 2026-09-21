"""Acquisition: retrieval, normalisation, deduplication, persistence."""

from research.acquisition.deduplicate import (
    DuplicateDetector,
    DuplicateVerdict,
    independent_documents,
)
# Document assembly lives in research.normalize (it is pure translation and
# source adapters depend on it); re-exported here for callers that think of it
# as part of acquisition.
from research.normalize.document import (
    build_document,
    detect_wire_service,
    document_from_hit,
    document_from_page,
    enrich,
)
from research.acquisition.pipeline import (
    AcquisitionResult,
    EvidenceAcquirer,
    summarise,
)
from research.acquisition.untrusted import (
    as_evidence_bundle,
    as_external_evidence,
    sanitize_external_text,
)

__all__ = [
    "AcquisitionResult",
    "DuplicateDetector",
    "DuplicateVerdict",
    "EvidenceAcquirer",
    "as_evidence_bundle",
    "as_external_evidence",
    "build_document",
    "detect_wire_service",
    "document_from_hit",
    "document_from_page",
    "enrich",
    "independent_documents",
    "sanitize_external_text",
    "summarise",
]
