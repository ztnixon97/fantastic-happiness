"""The corpus an evaluation runs against.

Built directly from the bundled demo corpus rather than through the provider
plumbing: an evaluation of retrieval should not be able to fail because a
mock transport changed. Documents keep their corpus identifiers (A1, N3, P2)
in a map, so judgements written by hand stay valid when the store renumbers.

Building it twice gives the same store. That is what makes a number from
this harness comparable to the same number last week.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research.acquisition.pipeline import EvidenceAcquirer
from research.models.common import Provenance, SourceFamily, SourceType, utcnow
from research.normalize.document import build_document
from research.normalize.html import extract_page, parse_date
from research.sources.offline import load_corpus
from research.storage.store import ResearchStore

EVAL_DATA = Path(__file__).resolve().parent.parent / "data" / "eval"


@dataclass(slots=True)
class EvaluationCorpus:
    """A store built from labelled material, with the labels still attached."""

    store: ResearchStore
    investigation_id: str
    #: corpus id -> document id, e.g. "N3" -> "evidence:11"
    ids: dict[str, str] = field(default_factory=dict)
    #: document id -> corpus id, for reading a ranking back in labelled terms
    labels: dict[str, str] = field(default_factory=dict)

    def resolve(self, corpus_ids: list[str]) -> list[str]:
        return [self.ids[key] for key in corpus_ids if key in self.ids]

    def label(self, document_ids: list[str]) -> list[str]:
        return [self.labels.get(document_id, document_id) for document_id in document_ids]

    @property
    def size(self) -> int:
        return self.store.documents.count(self.investigation_id)


def load_dataset(name: str) -> dict[str, Any]:
    """Read a labelled set from research/data/eval."""
    path = EVAL_DATA / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no evaluation dataset named {name!r} in {EVAL_DATA}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_corpus(store: ResearchStore, *, question: str | None = None) -> EvaluationCorpus:
    """Load the demo corpus into a store, keeping its labels.

    Deduplication runs as it always does, so the syndicated copies come out
    marked as copies - which is the thing the independence evaluation then
    checks, rather than something it sets up for itself.
    """
    corpus = load_corpus()
    investigation = store.investigations.create(question or corpus.get("question", "evaluation"))
    acquirer = EvidenceAcquirer(store)
    evaluation = EvaluationCorpus(store=store, investigation_id=investigation.id)

    for entry in corpus.get("academic", []):
        document = build_document(
            provider="offline_academic",
            source_type=SourceType.ACADEMIC_PEER_REVIEWED,
            source_family=SourceFamily.ACADEMIC,
            provenance=Provenance(provider="offline_academic", retrieved_at=utcnow()),
            title=entry.get("title"),
            abstract=entry.get("abstract"),
            text=entry.get("abstract"),
            url=f"https://doi.org/{entry['doi']}" if entry.get("doi") else None,
            external_id=entry.get("id"),
            authors=entry.get("authors") or [],
            published_at=parse_date(entry.get("published_at")),
            publisher=entry.get("venue"),
            doi=entry.get("doi"),
        )
        _record(evaluation, acquirer, entry["id"], document)

    for entry in corpus.get("web", []):
        page = extract_page(entry.get("html") or "")
        document = build_document(
            provider="offline_web",
            source_type=_source_type(entry),
            source_family=_family(entry),
            provenance=Provenance(provider="offline_web", retrieved_at=utcnow()),
            title=entry.get("title") or page.title,
            text=page.text or entry.get("snippet"),
            abstract=entry.get("snippet"),
            url=entry.get("url"),
            external_id=entry.get("id"),
            published_at=parse_date(entry.get("published_at")),
            publisher=entry.get("publisher"),
        )
        _record(evaluation, acquirer, entry["id"], document)

    _link_citations(evaluation, corpus)
    return evaluation


def _record(
    evaluation: EvaluationCorpus, acquirer: EvidenceAcquirer, key: str, document: Any
) -> None:
    result = acquirer.persist(document, investigation_id=evaluation.investigation_id)
    evaluation.ids[key] = result.document.id
    # A merged record maps two corpus ids to one document; the first wins as
    # the label, which is what a reader of the ranking expects to see.
    evaluation.labels.setdefault(result.document.id, key)


def _link_citations(evaluation: EvaluationCorpus, corpus: dict[str, Any]) -> None:
    """Record the corpus's own reference edges, so graph expansion has a graph."""
    for entry in corpus.get("academic", []):
        citing = evaluation.ids.get(entry["id"])
        if not citing:
            continue
        for reference in entry.get("references") or []:
            cited = evaluation.ids.get(reference)
            if cited:
                evaluation.store.citations.add(
                    evaluation.investigation_id,
                    citing_document_id=citing,
                    cited_document_id=cited,
                    provider="offline_academic",
                )


def _family(entry: dict[str, Any]) -> SourceFamily:
    try:
        return SourceFamily(entry.get("family") or "web")
    except ValueError:
        return SourceFamily.WEB


def _source_type(entry: dict[str, Any]) -> SourceType:
    try:
        return SourceType(entry.get("source_type") or "web_page")
    except ValueError:
        return SourceType.WEB_PAGE
