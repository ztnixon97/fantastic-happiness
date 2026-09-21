"""One handle onto the persistent investigation state."""

from __future__ import annotations

from pathlib import Path

from research.storage.database import Database
from research.storage.graph_repositories import (
    CitationRepository,
    ClaimRepository,
    EntityRepository,
    EventRepository,
    RelationshipRepository,
)
from research.storage.repositories import (
    BudgetRepository,
    DocumentRepository,
    InvestigationRepository,
    SearchQueryLog,
    SourceFetchLog,
    TaskRepository,
)

DEFAULT_DB_PATH = Path.home() / ".research" / "research.sqlite3"


class ResearchStore:
    """Aggregate of the repositories, sharing one connection.

    Everything an investigation knows is reachable from here, which is what
    makes a run resumable: a new process opens the same file and continues.
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self.investigations = InvestigationRepository(db)
        self.tasks = TaskRepository(db)
        self.documents = DocumentRepository(db)
        self.claims = ClaimRepository(db)
        self.entities = EntityRepository(db)
        self.events = EventRepository(db)
        self.relationships = RelationshipRepository(db)
        self.citations = CitationRepository(db)
        self.queries = SearchQueryLog(db)
        self.fetches = SourceFetchLog(db)
        self.budget = BudgetRepository(db)

    @classmethod
    def open(cls, path: str | Path | None = None) -> "ResearchStore":
        return cls(Database(path or DEFAULT_DB_PATH))

    @classmethod
    def in_memory(cls) -> "ResearchStore":
        return cls(Database(":memory:"))

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "ResearchStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
