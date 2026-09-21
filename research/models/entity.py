"""Entity registry: people, organisations, places, works.

Deterministic identifiers (DOI, ORCID, OpenAlex, ROR, Wikidata, CIK) decide
identity wherever they exist. Where they do not, candidate matches are
recorded as candidates - never silently merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from research.models.common import StrEnum, utcnow


class EntityType(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    ACADEMIC_PAPER = "academic_paper"
    PRODUCT = "product"
    OTHER = "other"


class MatchConfidence(StrEnum):
    """How an entity reference was resolved."""

    DETERMINISTIC = "deterministic"  # matched on an authoritative identifier
    HEURISTIC = "heuristic"  # matched on normalised name/alias
    AMBIGUOUS = "ambiguous"  # several candidates; left unmerged
    UNRESOLVED = "unresolved"


#: Identifier schemes considered authoritative enough to merge on.
AUTHORITATIVE_SCHEMES = frozenset(
    {
        "doi",
        "orcid",
        "openalex",
        "ror",
        "wikidata",
        "sec_cik",
        "arxiv",
        "pmid",
        "grid",
        "isni",
    }
)


@dataclass(frozen=True, slots=True)
class EntityIdentifier:
    scheme: str
    value: str

    @property
    def is_authoritative(self) -> bool:
        return self.scheme in AUTHORITATIVE_SCHEMES

    def key(self) -> str:
        return f"{self.scheme}:{self.value}"


@dataclass(slots=True)
class Entity:
    id: str
    entity_type: EntityType
    name: str
    investigation_id: str | None = None
    aliases: list[str] = field(default_factory=list)
    identifiers: list[EntityIdentifier] = field(default_factory=list)
    description: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def identifier(self, scheme: str) -> str | None:
        for ident in self.identifiers:
            if ident.scheme == scheme:
                return ident.value
        return None


@dataclass(slots=True)
class EntityResolution:
    """Outcome of resolving a name/identifier pair against the registry."""

    entity: Entity | None
    confidence: MatchConfidence
    matched_on: str | None = None
    #: Populated when the match was ambiguous; the caller decides what to do.
    candidates: list[Entity] = field(default_factory=list)
    created: bool = False
