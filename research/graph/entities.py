"""The entity registry.

Identity is decided by authoritative identifiers wherever they exist - ORCID,
OpenAlex, ROR, Wikidata, SEC CIK, DOI. Where they do not, matching falls back
to names, and the uncertainty that introduces is recorded on the entity
rather than hidden by it.

Two rules follow from that:

* Conflicting authoritative identifiers mean *different* entities, whatever
  the names say. Two ORCIDs are two people.
* A name-only match is recorded as such, and an ambiguous one resolves to
  candidates for a caller to decide between, not to a merge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from research.models.entity import (
    AUTHORITATIVE_SCHEMES,
    Entity,
    EntityIdentifier,
    EntityResolution,
    EntityType,
    MatchConfidence,
)
from research.models.evidence import EvidenceDocument
from research.normalize.text import normalize_title
from research.storage.store import ResearchStore

#: Names that carry too little information to merge on: initials, single
#: tokens, "et al." remnants. They still become entities; they just never
#: absorb a differently-spelled one.
_WEAK_NAME_RE = re.compile(r"^(?:[a-z]\.?\s+)*[a-z][a-z'-]*$", re.IGNORECASE)


def is_weak_name(name: str) -> bool:
    normalized = normalize_title(name)
    if not normalized:
        return True
    tokens = normalized.split()
    if len(tokens) < 2:
        return True
    return all(len(token) <= 2 for token in tokens[:-1]) and len(tokens) == 2


@dataclass(slots=True)
class EntityRegistry:
    """Resolves references to entities, and records where they were seen."""

    store: ResearchStore
    investigation_id: str
    task_id: str | None = None

    # -- resolution -----------------------------------------------------
    def resolve(
        self,
        name: str,
        entity_type: EntityType,
        *,
        identifiers: Sequence[EntityIdentifier] = (),
        aliases: Sequence[str] = (),
        description: str | None = None,
        create: bool = True,
    ) -> EntityResolution:
        """Find or create the entity this reference denotes."""
        authoritative = [
            identifier for identifier in identifiers if identifier.is_authoritative
        ]

        for identifier in authoritative:
            existing = self.store.entities.find_by_identifier(
                self.investigation_id, identifier.scheme, identifier.value
            )
            if existing:
                self._enrich(existing, identifiers, aliases)
                return EntityResolution(
                    entity=existing,
                    confidence=MatchConfidence.DETERMINISTIC,
                    matched_on=identifier.key(),
                )

        candidates = self.store.entities.find_by_name(
            self.investigation_id, entity_type, name
        )
        candidates = [
            candidate
            for candidate in candidates
            if not self._conflicts(candidate, authoritative)
        ]

        if len(candidates) == 1:
            entity = candidates[0]
            self._enrich(entity, identifiers, aliases)
            if is_weak_name(name) and not authoritative:
                # "J. Smith" matching "J. Smith" is a match of labels, not of
                # people. The link is kept but flagged for review.
                self._flag_weak_match(entity, name)
            return EntityResolution(
                entity=entity,
                confidence=MatchConfidence.HEURISTIC,
                matched_on=f"name={normalize_title(name)}",
            )

        if len(candidates) > 1:
            # Several plausible entities and nothing authoritative to choose
            # between them: say so rather than pick one.
            return EntityResolution(
                entity=None,
                confidence=MatchConfidence.AMBIGUOUS,
                matched_on=f"name={normalize_title(name)}",
                candidates=candidates,
            )

        if not create:
            return EntityResolution(entity=None, confidence=MatchConfidence.UNRESOLVED)

        entity = self.store.entities.create(
            self.investigation_id,
            entity_type,
            name,
            identifiers=list(identifiers),
            aliases=list(aliases),
            description=description,
            metadata={
                "identity_basis": "identifier" if authoritative else "name",
                "weak_name": is_weak_name(name) and not authoritative,
            },
        )
        return EntityResolution(
            entity=entity,
            confidence=(
                MatchConfidence.DETERMINISTIC if authoritative else MatchConfidence.HEURISTIC
            ),
            matched_on=authoritative[0].key() if authoritative else None,
            created=True,
        )

    def _conflicts(
        self, candidate: Entity, authoritative: Sequence[EntityIdentifier]
    ) -> bool:
        """True when an authoritative identifier says these are different."""
        for identifier in authoritative:
            existing = candidate.identifier(identifier.scheme)
            if existing and existing != identifier.value:
                return True
        return False

    def _enrich(
        self,
        entity: Entity,
        identifiers: Sequence[EntityIdentifier],
        aliases: Sequence[str],
    ) -> None:
        for identifier in identifiers:
            if identifier.scheme in AUTHORITATIVE_SCHEMES or not entity.identifier(
                identifier.scheme
            ):
                self.store.entities.add_identifier(entity.id, identifier)
        for alias in aliases:
            self.store.entities.add_alias(entity.id, alias, source=self.task_id)

    def _flag_weak_match(self, entity: Entity, name: str) -> None:
        metadata = dict(entity.metadata)
        matches = metadata.setdefault("name_only_matches", [])
        if name not in matches:
            matches.append(name)
        metadata["review_needed"] = True
        self.store.entities.update_metadata(entity.id, metadata)

    # -- mentions -------------------------------------------------------
    def record_mention(
        self,
        entity_id: str,
        document_id: str,
        *,
        predicate: str = "MENTIONS",
    ) -> None:
        """Record that a document mentions an entity, with the document as proof."""
        self.store.relationships.add(
            self.investigation_id,
            subject_type="document",
            subject_id=document_id,
            predicate=predicate,
            object_type="entity",
            object_id=entity_id,
            evidence_document_id=document_id,
            created_by_task_id=self.task_id,
        )

    def register_document(self, document: EvidenceDocument) -> list[EntityResolution]:
        """Register the entities a document's own metadata names.

        Authors and the publishing venue only: these come from the provider's
        structured fields, so they are as reliable as the record itself. Any
        entity found by reading the text is a model judgement and belongs to a
        research worker, not here.
        """
        resolutions: list[EntityResolution] = []
        orcids = document.metadata.get("author_orcids") or {}

        for author in document.authors[:25]:
            identifiers = []
            orcid = orcids.get(author) if isinstance(orcids, dict) else None
            if orcid:
                identifiers.append(EntityIdentifier("orcid", str(orcid).rsplit("/", 1)[-1]))
            resolution = self.resolve(author, EntityType.PERSON, identifiers=identifiers)
            if resolution.entity:
                self.record_mention(resolution.entity.id, document.id, predicate="AUTHORED_BY")
                resolutions.append(resolution)

        if document.publisher:
            identifiers = []
            if document.metadata.get("publisher_ror"):
                identifiers.append(
                    EntityIdentifier("ror", str(document.metadata["publisher_ror"]))
                )
            resolution = self.resolve(
                document.publisher, EntityType.ORGANIZATION, identifiers=identifiers
            )
            if resolution.entity:
                self.record_mention(resolution.entity.id, document.id, predicate="PUBLISHED_BY")
                resolutions.append(resolution)
        return resolutions

    # -- reading --------------------------------------------------------
    def documents_for(self, entity_id: str) -> list[str]:
        return [
            row["subject_id"]
            for row in self.store.relationships.for_object(self.investigation_id, entity_id)
            if row["subject_type"] == "document"
        ]

    def claims_for(self, entity_id: str) -> list[str]:
        return [
            row["subject_id"]
            for row in self.store.relationships.for_object(self.investigation_id, entity_id)
            if row["subject_type"] == "claim"
        ]

    def needs_review(self) -> list[Entity]:
        """Entities merged on a name alone, for a human or a later pass."""
        return [
            entity
            for entity in self.store.entities.list(self.investigation_id, limit=500)
            if entity.metadata.get("review_needed")
        ]
