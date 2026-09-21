"""Row <-> domain-object mapping.

Kept apart from the repositories so the SQL and the dataclasses can each be
read on their own.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from research.models.claim import Claim, ClaimEvidenceLink, ClaimStatus, EvidenceStance
from research.models.common import (
    DuplicateRelation,
    Provenance,
    RetrievalMethod,
    SourceFamily,
    SourceType,
)
from research.models.entity import Entity, EntityIdentifier, EntityType
from research.models.event import DatePrecision, Event
from research.models.evidence import EvidenceDocument
from research.models.investigation import (
    Investigation,
    InvestigationStatus,
    StopReason,
)
from research.models.task import (
    Operation,
    ResearchRole,
    ResearchTask,
    TaskResult,
    TaskStatus,
)
from research.storage.database import decode_dt, encode_dt, from_json, to_json


def provenance_to_json(provenance: Provenance | None) -> str:
    if provenance is None:
        return "{}"
    return to_json(
        {
            "provider": provenance.provider,
            "retrieval_method": str(provenance.retrieval_method),
            "retrieved_at": encode_dt(provenance.retrieved_at),
            "provider_endpoint": provenance.provider_endpoint,
            "query_id": provenance.query_id,
            "fetch_id": provenance.fetch_id,
            "requested_url": provenance.requested_url,
            "final_url": provenance.final_url,
            "parent_document_id": provenance.parent_document_id,
            "task_id": provenance.task_id,
            "notes": provenance.notes,
        }
    )


def provenance_from_json(raw: str | None) -> Provenance | None:
    data = from_json(raw, {})
    if not data:
        return None
    return Provenance(
        provider=data.get("provider", "unknown"),
        retrieval_method=RetrievalMethod.coerce(
            data.get("retrieval_method"), RetrievalMethod.SEARCH
        ),
        retrieved_at=decode_dt(data.get("retrieved_at")) or decode_dt("1970-01-01T00:00:00+00:00"),
        provider_endpoint=data.get("provider_endpoint"),
        query_id=data.get("query_id"),
        fetch_id=data.get("fetch_id"),
        requested_url=data.get("requested_url"),
        final_url=data.get("final_url"),
        parent_document_id=data.get("parent_document_id"),
        task_id=data.get("task_id"),
        notes=data.get("notes"),
    )


def document_to_params(document: EvidenceDocument, url_key: str | None) -> dict[str, Any]:
    return {
        "id": document.id,
        "investigation_id": document.investigation_id,
        "source_type": str(document.source_type),
        "source_family": str(document.source_family),
        "provider": document.provider,
        "external_id": document.external_id,
        "canonical_url": document.canonical_url,
        "url_key": url_key,
        "title": document.title,
        "title_key": document.title_key,
        "authors": to_json(document.authors),
        "published_at": encode_dt(document.published_at),
        "fetched_at": encode_dt(document.fetched_at),
        "text": document.text,
        "abstract": document.abstract,
        "metadata": to_json(document.metadata),
        "content_hash": document.content_hash,
        "doi": document.doi,
        "canonical_host": document.canonical_host,
        "publisher": document.publisher,
        "language": document.language,
        "simhash": f"{document.simhash:016x}" if document.simhash is not None else None,
        "duplicate_of": document.duplicate_of,
        "duplicate_relation": str(document.duplicate_relation)
        if document.duplicate_relation
        else None,
        "derived_from": document.derived_from,
        "independence_key": document.independence_key,
        "provenance": provenance_to_json(document.provenance),
    }


def row_to_document(row: sqlite3.Row) -> EvidenceDocument:
    return EvidenceDocument(
        id=row["id"],
        investigation_id=row["investigation_id"],
        source_type=SourceType.coerce(row["source_type"], SourceType.OTHER),
        source_family=SourceFamily.coerce(row["source_family"], SourceFamily.WEB),
        provider=row["provider"],
        external_id=row["external_id"],
        canonical_url=row["canonical_url"],
        title=row["title"],
        title_key=row["title_key"],
        authors=list(from_json(row["authors"], [])),
        published_at=decode_dt(row["published_at"]),
        fetched_at=decode_dt(row["fetched_at"]),
        text=row["text"],
        abstract=row["abstract"],
        metadata=dict(from_json(row["metadata"], {})),
        content_hash=row["content_hash"],
        doi=row["doi"],
        canonical_host=row["canonical_host"],
        publisher=row["publisher"],
        language=row["language"],
        simhash=int(row["simhash"], 16) if row["simhash"] else None,
        duplicate_of=row["duplicate_of"],
        duplicate_relation=DuplicateRelation.coerce(row["duplicate_relation"], None),
        derived_from=row["derived_from"],
        independence_key=row["independence_key"],
        provenance=provenance_from_json(row["provenance"]),
    )


def row_to_investigation(row: sqlite3.Row) -> Investigation:
    return Investigation(
        id=row["id"],
        question=row["question"],
        brief=row["brief"],
        status=InvestigationStatus.coerce(row["status"], InvestigationStatus.CREATED),
        stop_reason=StopReason.coerce(row["stop_reason"], None),
        stop_detail=row["stop_detail"],
        budget=dict(from_json(row["budget"], {})),
        tags=list(from_json(row["tags"], [])),
        metadata=dict(from_json(row["metadata"], {})),
        created_at=decode_dt(row["created_at"]),
        updated_at=decode_dt(row["updated_at"]),
        completed_at=decode_dt(row["completed_at"]),
    )


def row_to_task(row: sqlite3.Row) -> ResearchTask:
    raw_result = from_json(row["result"], {}) if row["result"] else {}
    return ResearchTask(
        id=row["id"],
        investigation_id=row["investigation_id"],
        parent_task_id=row["parent_task_id"],
        role=ResearchRole.coerce(row["role"], ResearchRole.SCOUT),
        operation=Operation.coerce(row["operation"], Operation.SEARCH_WEB),
        objective=row["objective"],
        status=TaskStatus.coerce(row["status"], TaskStatus.PENDING),
        depth=row["depth"],
        priority=row["priority"],
        parameters=dict(from_json(row["parameters"], {})),
        result=TaskResult.from_dict(raw_result) if raw_result else None,
        error=row["error"],
        created_at=decode_dt(row["created_at"]),
        started_at=decode_dt(row["started_at"]),
        finished_at=decode_dt(row["finished_at"]),
    )


def row_to_claim(row: sqlite3.Row) -> Claim:
    return Claim(
        id=row["id"],
        investigation_id=row["investigation_id"],
        text=row["text"],
        status=ClaimStatus.coerce(row["status"], ClaimStatus.UNVERIFIED),
        notes=row["notes"],
        metadata=dict(from_json(row["metadata"], {})),
        created_by_task_id=row["created_by_task_id"],
        created_at=decode_dt(row["created_at"]),
        updated_at=decode_dt(row["updated_at"]),
    )


def row_to_claim_link(row: sqlite3.Row) -> ClaimEvidenceLink:
    return ClaimEvidenceLink(
        claim_id=row["claim_id"],
        document_id=row["document_id"],
        stance=EvidenceStance.coerce(row["stance"], EvidenceStance.MENTIONS),
        excerpt=row["excerpt"],
        analysis=row["analysis"],
        created_at=decode_dt(row["created_at"]),
        created_by_task_id=row["created_by_task_id"],
    )


def row_to_entity(
    row: sqlite3.Row, identifiers: list[EntityIdentifier], aliases: list[str]
) -> Entity:
    return Entity(
        id=row["id"],
        investigation_id=row["investigation_id"],
        entity_type=EntityType.coerce(row["entity_type"], EntityType.OTHER),
        name=row["name"],
        description=row["description"],
        identifiers=identifiers,
        aliases=aliases,
        metadata=dict(from_json(row["metadata"], {})),
        created_at=decode_dt(row["created_at"]),
        updated_at=decode_dt(row["updated_at"]),
    )


def row_to_event(row: sqlite3.Row, entity_ids: list[str], evidence_ids: list[str]) -> Event:
    return Event(
        id=row["id"],
        investigation_id=row["investigation_id"],
        description=row["description"],
        date_start=decode_dt(row["date_start"]),
        date_end=decode_dt(row["date_end"]),
        date_precision=DatePrecision.coerce(row["date_precision"], DatePrecision.UNKNOWN),
        entity_ids=entity_ids,
        evidence_ids=evidence_ids,
        metadata=dict(from_json(row["metadata"], {})),
        created_by_task_id=row["created_by_task_id"],
        created_at=decode_dt(row["created_at"]),
    )
