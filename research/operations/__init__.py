"""Research-native operations.

The verbs an investigation is made of. Each one composes sources, storage,
budgets and provenance; none of them is a general-purpose tool, and none of
them can execute anything.
"""

from research.operations.citation_chase import CitationChase, CitationChaseResult
from research.operations.claims import ClaimOperations, LinkResult
from research.operations.counterevidence import (
    CounterevidenceResult,
    CounterevidenceSearch,
)
from research.operations.primary_source import PrimarySourceChase, PrimarySourceResult
from research.operations.search import SearchOperation, SearchOutcome
from research.operations.timeline import TimelineOperations

__all__ = [
    "CitationChase",
    "CitationChaseResult",
    "ClaimOperations",
    "CounterevidenceResult",
    "CounterevidenceSearch",
    "LinkResult",
    "PrimarySourceChase",
    "PrimarySourceResult",
    "SearchOperation",
    "SearchOutcome",
    "TimelineOperations",
]
