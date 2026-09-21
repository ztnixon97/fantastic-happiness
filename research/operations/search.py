"""Search as a research operation.

A research-native verb - 'search academic literature', 'search current news'
- fans out across the providers that serve that family, records every call,
charges the budget, turns results into evidence, and reports what failed.

Partial failure is the normal case, not an exception: if two of three
providers answer, the investigation continues with two and says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from research.acquisition.pipeline import AcquisitionResult, EvidenceAcquirer, summarise
from research.errors import BudgetExceeded, SourceError
from research.models.common import RetrievalMethod, SourceFamily
from research.models.query import ResearchQuery, SearchHit
from research.budgets import BudgetLedger, Resource
from research.sources.base import ResearchSource
from research.sources.registry import SourceRegistry
from research.storage.store import ResearchStore


@dataclass(slots=True)
class SearchOutcome:
    """Everything one logical search did, for the record and for the caller."""

    query_text: str
    family: SourceFamily
    providers_used: list[str] = field(default_factory=list)
    provider_errors: dict[str, str] = field(default_factory=dict)
    hits: list[SearchHit] = field(default_factory=list)
    results: list[AcquisitionResult] = field(default_factory=list)
    #: Hits whose body could not be retrieved. A search snippet is written by
    #: the search engine, so an unfetchable result is not stored as evidence.
    unretrieved: list[SearchHit] = field(default_factory=list)
    stopped_by: str | None = None

    @property
    def document_ids(self) -> list[str]:
        return [result.document.id for result in self.results if result.document.id]

    @property
    def new_document_ids(self) -> list[str]:
        return [result.document.id for result in self.results if result.is_new_evidence]

    def summary(self) -> dict[str, Any]:
        data = summarise(self.results)
        data.update(
            {
                "query": self.query_text,
                "family": str(self.family),
                "providers": list(self.providers_used),
                "provider_errors": dict(self.provider_errors),
                "unretrieved": len(self.unretrieved),
            }
        )
        if self.stopped_by:
            data["stopped_by"] = self.stopped_by
        return data


class SearchOperation:
    """Runs searches for one investigation."""

    def __init__(
        self,
        store: ResearchStore,
        registry: SourceRegistry,
        *,
        investigation_id: str,
        ledger: BudgetLedger | None = None,
        acquirer: EvidenceAcquirer | None = None,
        task_id: str | None = None,
        max_providers_per_search: int = 3,
    ) -> None:
        self.store = store
        self.registry = registry
        self.investigation_id = investigation_id
        self.ledger = ledger
        self.acquirer = acquirer or EvidenceAcquirer(store, ledger=ledger)
        self.task_id = task_id
        self.max_providers_per_search = max_providers_per_search

    async def search(
        self,
        text: str,
        *,
        family: SourceFamily,
        limit: int = 10,
        objective: str | None = None,
        published_after: datetime | None = None,
        published_before: datetime | None = None,
        language: str | None = None,
        filters: dict[str, Any] | None = None,
        providers: Sequence[str] | None = None,
        fetch_bodies: bool | None = None,
        max_fetches: int | None = None,
    ) -> SearchOutcome:
        outcome = SearchOutcome(query_text=text, family=family)

        if self.ledger is not None and not self.ledger.try_spend(Resource.SEARCHES):
            outcome.stopped_by = "search budget exhausted"
            return outcome

        query = ResearchQuery(
            text=text,
            families=[family],
            limit=limit,
            published_after=published_after,
            published_before=published_before,
            language=language,
            filters=dict(filters or {}),
            objective=objective,
            investigation_id=self.investigation_id,
            task_id=self.task_id,
        )

        for source in self._sources(family, providers):
            hits = await self._run_provider(source, query, family, outcome)
            outcome.hits.extend(hits)

        await self._acquire(outcome, family, fetch_bodies=fetch_bodies, max_fetches=max_fetches)
        return outcome

    # -- providers ------------------------------------------------------
    def _sources(
        self, family: SourceFamily, providers: Sequence[str] | None
    ) -> list[ResearchSource]:
        if providers:
            chosen = [self.registry.get(name) for name in providers]
            return [source for source in chosen if source is not None]
        return self.registry.for_family(family)[: self.max_providers_per_search]

    async def _run_provider(
        self,
        source: ResearchSource,
        query: ResearchQuery,
        family: SourceFamily,
        outcome: SearchOutcome,
    ) -> list[SearchHit]:
        if self.ledger is not None and not self.ledger.try_spend(Resource.PROVIDER_CALLS):
            outcome.stopped_by = "provider call budget exhausted"
            return []
        started = time.monotonic()
        try:
            hits = await source.search(query)
        except SourceError as exc:
            # One provider failing is information, not a crash: it is recorded
            # against the investigation and the others still run.
            duration = int((time.monotonic() - started) * 1000)
            self.store.queries.record(
                query,
                provider=source.name,
                family=family,
                status="failed",
                error=str(exc),
                duration_ms=duration,
            )
            outcome.provider_errors[source.name] = str(exc)
            if self.ledger is not None:
                self.ledger.try_spend(Resource.FAILED_SOURCE_CALLS)
            return []

        duration = int((time.monotonic() - started) * 1000)
        query_id = self.store.queries.record(
            query,
            provider=source.name,
            family=family,
            status="ok",
            result_count=len(hits),
            duration_ms=duration,
        )
        for hit in hits:
            hit.raw["_query_id"] = query_id
        outcome.providers_used.append(source.name)
        return hits

    # -- acquisition ----------------------------------------------------
    async def _acquire(
        self,
        outcome: SearchOutcome,
        family: SourceFamily,
        *,
        fetch_bodies: bool | None,
        max_fetches: int | None,
    ) -> None:
        fetches_remaining = max_fetches if max_fetches is not None else len(outcome.hits)
        for hit in outcome.hits:
            source = self.registry.get(hit.provider)
            capability = source.capabilities() if source else None
            inline = bool(capability and capability.returns_inline_documents)
            wants_body = fetch_bodies if fetch_bodies is not None else not inline

            document = None
            if inline and not wants_body and source is not None:
                document = await source.fetch(hit)
            elif self.registry.fetcher is not None and hit.url and fetches_remaining > 0:
                fetches_remaining -= 1
                document = await self._fetch_body(hit, family)
            elif inline and source is not None:
                document = await source.fetch(hit)

            if document is None:
                outcome.unretrieved.append(hit)
                continue
            self._attach_query(document, hit)
            result = self.acquirer.persist(document, investigation_id=self.investigation_id)
            outcome.results.append(result)
            if result.refused:
                outcome.stopped_by = result.refused
                break

    @staticmethod
    def _attach_query(document: Any, hit: SearchHit) -> None:
        query_id = hit.raw.get("_query_id")
        if query_id and document.provenance is not None:
            object.__setattr__(document.provenance, "query_id", query_id)

    async def _fetch_body(self, hit: SearchHit, family: SourceFamily) -> Any:
        """Retrieve a result's body, recording the call either way."""
        fetcher = self.registry.fetcher
        if fetcher is None or not hit.url:
            return None
        if self.ledger is not None and not self.ledger.try_spend(Resource.PROVIDER_CALLS):
            return None

        fetch_id = self.store.fetches.begin(
            provider=fetcher.name,
            url=hit.url,
            investigation_id=self.investigation_id,
            task_id=self.task_id,
        )
        started = time.monotonic()
        try:
            document = await fetcher.fetch_url(
                hit.url,
                source_family=family,
                source_type=hit.source_type,
                fetch_id=fetch_id,
                task_id=self.task_id,
                published_at_hint=hit.published_at,
                publisher_hint=hit.publisher,
            )
        except (SourceError, ValueError) as exc:
            self.store.fetches.complete(
                fetch_id,
                ok=False,
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            if self.ledger is not None:
                self.ledger.try_spend(Resource.FAILED_SOURCE_CALLS)
            return None

        self.store.fetches.complete(
            fetch_id,
            ok=True,
            status_code=document.metadata.get("http_status"),
            content_type=document.metadata.get("content_type"),
            num_bytes=len(document.text or ""),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return document

    # -- direct fetch ---------------------------------------------------
    async def fetch_source(
        self,
        url: str,
        *,
        family: SourceFamily = SourceFamily.WEB,
        parent_document_id: str | None = None,
        retrieval_method: RetrievalMethod = RetrievalMethod.DIRECT_FETCH,
    ) -> AcquisitionResult | None:
        """Retrieve one known URL and persist it as evidence."""
        fetcher = self.registry.fetcher
        if fetcher is None:
            return None
        if self.ledger is not None and not self.ledger.try_spend(Resource.PROVIDER_CALLS):
            return None

        fetch_id = self.store.fetches.begin(
            provider=fetcher.name,
            url=url,
            investigation_id=self.investigation_id,
            task_id=self.task_id,
        )
        started = time.monotonic()
        try:
            document = await fetcher.fetch_url(
                url,
                source_family=family,
                parent_document_id=parent_document_id,
                retrieval_method=retrieval_method,
                fetch_id=fetch_id,
                task_id=self.task_id,
            )
        except (SourceError, ValueError) as exc:
            self.store.fetches.complete(
                fetch_id,
                ok=False,
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            if self.ledger is not None:
                self.ledger.try_spend(Resource.FAILED_SOURCE_CALLS)
            return None

        self.store.fetches.complete(
            fetch_id,
            ok=True,
            status_code=document.metadata.get("http_status"),
            content_type=document.metadata.get("content_type"),
            num_bytes=len(document.text or ""),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        try:
            return self.acquirer.persist(document, investigation_id=self.investigation_id)
        except BudgetExceeded:
            return None
