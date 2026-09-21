"""Deduplication and source independence.

The question this module answers is not 'have I seen these bytes before?'
but 'is this a *new* piece of evidence?'. Ten outlets running one wire story
are one piece of evidence; the same paper returned by four academic
providers is one paper; an article written from another article is not
independent corroboration of it.

Decisions are made by ordinary code and recorded with the reason and the
matched key, so a later, better implementation can be judged against this
one. Where the evidence is ambiguous the conservative answer is 'not
independent', because the failure mode that matters is counting one source
several times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from research.models.common import DuplicateRelation, SourceFamily
from research.models.evidence import EvidenceDocument
from research.normalize.fingerprint import containment, jaccard, shingles, simhash_bands
from research.normalize.urls import registrable_domain, url_identity_key
from research.storage.repositories import DocumentRepository

#: Identity schemes that name the same record outright.
_RECORD_SCHEMES = frozenset({"doi", "arxiv", "pmid", "isbn", "provider_id"})
#: Identity schemes that mean the same retrieved artifact.
_ARTIFACT_SCHEMES = frozenset({"canonical_url", "url_key", "content_hash"})

#: Relations that make a document a duplicate rather than merely dependent.
DUPLICATE_RELATIONS = frozenset(
    {
        DuplicateRelation.EXACT_DUPLICATE,
        DuplicateRelation.NEAR_DUPLICATE,
        DuplicateRelation.SAME_RECORD,
        DuplicateRelation.SYNDICATED_COPY,
    }
)


@dataclass(frozen=True, slots=True)
class DuplicateVerdict:
    """Why a document is, or is not, independent evidence."""

    relation: DuplicateRelation | None = None
    matched_document_id: str | None = None
    matched_on: str | None = None
    similarity: float | None = None
    explanation: str = "no matching material held"

    @property
    def independent(self) -> bool:
        return self.relation is None

    @property
    def is_duplicate(self) -> bool:
        return self.relation in DUPLICATE_RELATIONS

    @property
    def is_derived(self) -> bool:
        return self.relation is DuplicateRelation.DERIVED_ARTICLE


class DuplicateDetector:
    """Compares an incoming document against material already held."""

    def __init__(
        self,
        documents: DocumentRepository,
        *,
        near_duplicate_threshold: float = 0.72,
        derived_threshold: float = 0.40,
        max_candidates: int = 120,
    ) -> None:
        self.documents = documents
        self.near_duplicate_threshold = near_duplicate_threshold
        self.derived_threshold = derived_threshold
        self.max_candidates = max_candidates

    # -- public ---------------------------------------------------------
    def check(
        self, document: EvidenceDocument, investigation_id: str | None
    ) -> DuplicateVerdict:
        verdict = self._check_identity(document, investigation_id)
        if verdict is not None:
            return verdict
        verdict = self._check_similarity(document, investigation_id)
        if verdict.independent:
            # A piece that credits and links another held document is written
            # from it, however much it was rewritten. Paraphrase defeats text
            # comparison; an attribution link does not.
            attribution = self._check_attribution(document, investigation_id)
            if attribution is not None:
                return attribution
        return verdict

    def _check_attribution(
        self, document: EvidenceDocument, investigation_id: str | None
    ) -> DuplicateVerdict | None:
        if document.source_family is not SourceFamily.NEWS:
            return None
        links = document.metadata.get("outbound_links") or []
        own_domain = registrable_domain(document.canonical_host or document.canonical_url)
        for link in links[:50]:
            href = link[0] if isinstance(link, (list, tuple)) and link else link
            if not isinstance(href, str):
                continue
            key = url_identity_key(href)
            if not key or registrable_domain(href) == own_domain:
                continue
            match_id = self.documents.find_by_identity(investigation_id, "url_key", key)
            if not match_id:
                continue
            return DuplicateVerdict(
                relation=DuplicateRelation.DERIVED_ARTICLE,
                matched_document_id=match_id,
                matched_on="attribution-link",
                explanation=(
                    f"credits and links {match_id}: written from it, so not counted "
                    "as independent corroboration"
                ),
            )
        return None

    # -- exact identity -------------------------------------------------
    def _check_identity(
        self, document: EvidenceDocument, investigation_id: str | None
    ) -> DuplicateVerdict | None:
        for scheme, value in document.identity_keys():
            if scheme == "title_key":
                continue  # too weak on its own; handled by similarity
            match_id = self.documents.find_by_identity(investigation_id, scheme, value)
            if not match_id:
                continue
            existing = self.documents.get(match_id)
            if scheme in _RECORD_SCHEMES:
                return DuplicateVerdict(
                    relation=DuplicateRelation.SAME_RECORD,
                    matched_document_id=match_id,
                    matched_on=f"{scheme}={value}",
                    explanation=(
                        f"same record as {match_id}: identical {scheme} "
                        f"({value}); held from {existing.provider}"
                    ),
                )
            if scheme in _ARTIFACT_SCHEMES:
                relation = DuplicateRelation.EXACT_DUPLICATE
                explanation = f"identical {scheme} to {match_id}"
                if scheme == "content_hash" and _different_outlet(document, existing):
                    relation = DuplicateRelation.SYNDICATED_COPY
                    explanation = (
                        f"byte-identical text to {match_id} on a different site "
                        f"({existing.canonical_host} vs {document.canonical_host}): "
                        "republished copy, not independent reporting"
                    )
                return DuplicateVerdict(
                    relation=relation,
                    matched_document_id=match_id,
                    matched_on=f"{scheme}={value}",
                    similarity=1.0,
                    explanation=explanation,
                )
        return None

    # -- similarity -----------------------------------------------------
    def _check_similarity(
        self, document: EvidenceDocument, investigation_id: str | None
    ) -> DuplicateVerdict:
        candidates = self._candidates(document, investigation_id)
        if not candidates:
            return DuplicateVerdict()

        incoming_shingles = shingles(document.best_text)
        best: tuple[float, EvidenceDocument] | None = None
        for candidate in candidates:
            candidate_shingles = shingles(candidate.best_text)
            # Either direction of containment counts: a copy may drop the
            # original's closing paragraphs or add a house sidebar, and both
            # depress Jaccard without making the copy independent.
            score = max(
                jaccard(incoming_shingles, candidate_shingles),
                containment(incoming_shingles, candidate_shingles),
                containment(candidate_shingles, incoming_shingles),
            )
            if best is None or score > best[0]:
                best = (score, candidate)
        assert best is not None
        score, candidate = best

        same_title = bool(document.title_key) and document.title_key == candidate.title_key
        wire = document.metadata.get("wire_service")
        same_wire = bool(wire) and wire == candidate.metadata.get("wire_service")
        different_outlet = _different_outlet(document, candidate)

        if score >= self.near_duplicate_threshold:
            if different_outlet:
                return DuplicateVerdict(
                    relation=DuplicateRelation.SYNDICATED_COPY,
                    matched_document_id=candidate.id,
                    matched_on="body-similarity",
                    similarity=round(score, 3),
                    explanation=(
                        f"{score:.0%} of the text matches {candidate.id} published by "
                        f"{candidate.canonical_host or candidate.provider}: syndicated "
                        "copy, counted as one source"
                    ),
                )
            return DuplicateVerdict(
                relation=DuplicateRelation.NEAR_DUPLICATE,
                matched_document_id=candidate.id,
                matched_on="body-similarity",
                similarity=round(score, 3),
                explanation=(
                    f"{score:.0%} of the text matches {candidate.id} on the same site"
                ),
            )

        # Thin documents (search snippets, abstracts only) cannot be judged on
        # body similarity; a shared headline plus a shared wire credit is the
        # stronger signal there.
        if same_title and (same_wire or score >= self.derived_threshold):
            relation = (
                DuplicateRelation.SYNDICATED_COPY
                if same_wire or different_outlet
                else DuplicateRelation.NEAR_DUPLICATE
            )
            reason = f"identical headline to {candidate.id}"
            if same_wire:
                reason += f" and both credited to {wire}"
            return DuplicateVerdict(
                relation=relation,
                matched_document_id=candidate.id,
                matched_on="title_key",
                similarity=round(score, 3),
                explanation=reason,
            )

        if score >= self.derived_threshold and document.source_family is SourceFamily.NEWS:
            return DuplicateVerdict(
                relation=DuplicateRelation.DERIVED_ARTICLE,
                matched_document_id=candidate.id,
                matched_on="body-similarity",
                similarity=round(score, 3),
                explanation=(
                    f"{score:.0%} overlap with {candidate.id}: likely written from it, "
                    "so not counted as independent corroboration"
                ),
            )

        return DuplicateVerdict(
            similarity=round(score, 3),
            explanation=(
                f"closest held material is {candidate.id} at {score:.0%} overlap: "
                "treated as independent"
            ),
        )

    def _candidates(
        self, document: EvidenceDocument, investigation_id: str | None
    ) -> list[EvidenceDocument]:
        """Retrieve plausible matches without scanning the whole corpus."""
        ids: list[str] = []
        seen: set[str] = set()

        def add(candidate_ids: Iterable[str]) -> None:
            for candidate_id in candidate_ids:
                if candidate_id != document.id and candidate_id not in seen:
                    seen.add(candidate_id)
                    ids.append(candidate_id)

        add(self.documents.find_by_bands(investigation_id, simhash_bands(document.simhash)))
        if document.title_key:
            match_id = self.documents.find_by_identity(
                investigation_id, "title_key", document.title_key
            )
            if match_id:
                add([match_id])
        return self.documents.get_many(ids[: self.max_candidates])


def _different_outlet(left: EvidenceDocument, right: EvidenceDocument) -> bool:
    left_domain = registrable_domain(left.canonical_host or left.canonical_url)
    right_domain = registrable_domain(right.canonical_host or right.canonical_url)
    if not left_domain or not right_domain:
        return left.provider != right.provider
    return left_domain != right_domain


def independent_documents(
    documents: DocumentRepository, document_ids: list[str]
) -> list[list[str]]:
    """Partition documents into independence groups.

    Each returned group counts as *one* source, whatever its size. This is
    what report generation must use when it says 'two independent sources'.
    """
    groups = documents.independence_groups(document_ids)
    return [sorted(members, key=_id_sort_key) for _, members in sorted(groups.items())]


def _id_sort_key(document_id: str) -> tuple[str, int]:
    prefix, _, number = document_id.partition(":")
    return (prefix, int(number) if number.isdigit() else 0)
