"""Research-native operations.

The verbs an investigation is made of. Each one composes sources, storage,
budgets and provenance; none of them is a general-purpose tool, and none of
them can execute anything.
"""

from research.operations.citation_chase import CitationChase, CitationChaseResult
from research.operations.search import SearchOperation, SearchOutcome

__all__ = [
    "CitationChase",
    "CitationChaseResult",
    "SearchOperation",
    "SearchOutcome",
]
