"""Tracing an assertion back toward the record behind it.

An article says a filing shows something; a summary cites a paper; a figure
comes from a dataset. The article is evidence that the assertion was made.
The filing is evidence about the world. This operation tries to turn the
first into the second.

Two deterministic routes are used before any judgement is needed: links the
document itself carries, and identifiers (DOIs) written into its text. Both
are provenance-preserving - whatever is acquired records the document it was
reached from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.models.common import (
    PRIMARY_SOURCE_DISTANCE,
    RetrievalMethod,
    SourceFamily,
    SourceType,
)
from research.normalize.doi import extract_doi
from research.normalize.urls import canonicalize_url, registrable_domain
from research.operations.search import SearchOperation
from research.sources.classify import classify_url
from research.sources.registry import SourceRegistry
from research.storage.store import ResearchStore

#: Source types worth spending a fetch on when chasing a record.
_WORTH_CHASING = {
    SourceType.CORPORATE_FILING,
    SourceType.REGULATORY_DOCUMENT,
    SourceType.GOVERNMENT_DOCUMENT,
    SourceType.TRANSCRIPT,
    SourceType.PRESS_RELEASE,
    SourceType.ACADEMIC_PEER_REVIEWED,
    SourceType.ACADEMIC_PREPRINT,
}


@dataclass(slots=True)
class PrimarySourceResult:
    document_id: str
    assertion: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    acquired: list[str] = field(default_factory=list)
    closer_than_original: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "from_document": self.document_id,
            "assertion": self.assertion,
            "candidates_considered": len(self.candidates),
            "acquired": list(self.acquired),
            "closer_to_the_record": list(self.closer_than_original),
            "notes": list(self.notes),
        }


class PrimarySourceChase:
    def __init__(
        self,
        store: ResearchStore,
        registry: SourceRegistry,
        search: SearchOperation,
        *,
        investigation_id: str,
        task_id: str | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.search = search
        self.investigation_id = investigation_id
        self.task_id = task_id

    async def chase(
        self,
        document_id: str,
        *,
        assertion: str | None = None,
        max_candidates: int = 4,
    ) -> PrimarySourceResult:
        document = self.store.documents.get(document_id)
        result = PrimarySourceResult(document_id=document_id, assertion=assertion)
        origin_distance = PRIMARY_SOURCE_DISTANCE.get(document.source_type, 4)

        for candidate in self._link_candidates(document)[:max_candidates]:
            result.candidates.append(candidate)
            acquisition = await self.search.fetch_source(
                candidate["url"],
                family=candidate["family"],
                parent_document_id=document_id,
                retrieval_method=RetrievalMethod.PRIMARY_SOURCE_CHASE,
            )
            if acquisition is None:
                result.notes.append(f"could not retrieve {candidate['url']}")
                continue
            acquired = acquisition.document
            # Record the relation as a graph edge, not as `derived_from`.
            # `derived_from` means "not independent of", and an article that
            # cites a filing is not a copy of it - it is an account of it.
            self.store.relationships.add(
                self.investigation_id,
                subject_type="document",
                subject_id=document_id,
                predicate="REPORTS_ON",
                object_type="document",
                object_id=acquired.id,
                evidence_document_id=acquired.id,
                created_by_task_id=self.task_id,
            )
            result.acquired.append(acquired.id)
            if PRIMARY_SOURCE_DISTANCE.get(acquired.source_type, 4) < origin_distance:
                result.closer_than_original.append(acquired.id)

        await self._chase_dois(document, result)

        if not result.acquired:
            result.notes.append(
                "no linked record was retrievable; the assertion may need a named search "
                "for the underlying filing, dataset or paper"
            )
        return result

    def _link_candidates(self, document: Any) -> list[dict[str, Any]]:
        """Outbound links that look like the record rather than more coverage."""
        own_domain = registrable_domain(document.canonical_host or document.canonical_url)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        for link in document.metadata.get("outbound_links") or []:
            href = link[0] if isinstance(link, (list, tuple)) and link else link
            anchor = link[1] if isinstance(link, (list, tuple)) and len(link) > 1 else ""
            url = canonicalize_url(href) if isinstance(href, str) else None
            if not url or url in seen:
                continue
            if registrable_domain(url) == own_domain:
                # A link back into the same outlet is more of the same coverage.
                continue
            source_type = classify_url(url)
            if source_type not in _WORTH_CHASING:
                continue
            seen.add(url)
            candidates.append(
                {
                    "url": url,
                    "anchor": anchor,
                    "source_type": str(source_type),
                    "family": _family_for(source_type),
                    "distance": PRIMARY_SOURCE_DISTANCE.get(source_type, 4),
                }
            )

        candidates.sort(key=lambda candidate: candidate["distance"])
        return candidates

    async def _chase_dois(self, document: Any, result: PrimarySourceResult) -> None:
        """Resolve DOIs written into the text - the citation an article gives."""
        dois = extract_doi(document.best_text, limit=3)
        if not dois:
            return
        resolvers = self.registry.doi_resolvers()
        if not resolvers:
            result.notes.append(f"{len(dois)} DOI(s) named but no provider can resolve them")
            return

        for doi in dois:
            if self.store.documents.find_by_identity(self.investigation_id, "doi", doi):
                result.notes.append(f"{doi} is already held")
                continue
            for resolver in resolvers:
                hit = await resolver.lookup_doi(doi)
                if hit is None:
                    continue
                outcome = await self.search.search(
                    hit.title or doi,
                    family=SourceFamily.ACADEMIC,
                    limit=3,
                    objective=f"resolve {doi} cited by {document.id}",
                    providers=[resolver.name],
                )
                result.acquired.extend(outcome.new_document_ids)
                result.closer_than_original.extend(outcome.new_document_ids)
                break


def _family_for(source_type: SourceType) -> SourceFamily:
    if source_type in (SourceType.ACADEMIC_PEER_REVIEWED, SourceType.ACADEMIC_PREPRINT):
        return SourceFamily.ACADEMIC
    if source_type in (SourceType.CORPORATE_FILING, SourceType.PRESS_RELEASE):
        return SourceFamily.CORPORATE
    if source_type in (SourceType.REGULATORY_DOCUMENT, SourceType.GOVERNMENT_DOCUMENT):
        return SourceFamily.GOVERNMENT
    return SourceFamily.WEB
