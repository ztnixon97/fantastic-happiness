"""Taking local files into an investigation as evidence.

A research environment that can only reach material over HTTP cannot use the
papers, filings and notes a person already has. This reads them in, through
the same pipeline as anything retrieved: normalised into documents,
deduplicated against what is held, indexed, and recorded with provenance
saying where each one came from.

Ingested material is *evidence*, not instruction, and gets no special
standing for having arrived from disk. It is deduplicated against fetched
material - a paper saved locally and the same paper found through a provider
are one source - and it is wrapped in the same untrusted-content envelope
when a worker reads it.

Filesystem access is narrow and explicit, and it lives here rather than in
the action vocabulary. There is no action that opens a path: ingestion is
something a person runs, naming the files, and a worker only ever sees the
resulting documents. Within a run:

* the paths named on the command line are the roots, and nothing outside
  them is read - a symlink pointing out of a root is skipped, not followed,
* files above a size limit are skipped rather than loaded,
* hidden files and directories are skipped, so pointing this at a project
  directory does not ingest its .git,
* file types are an allowlist, and content decides the type, not the name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from research.acquisition.pipeline import AcquisitionResult, EvidenceAcquirer
from research.budgets import BudgetLedger
from research.models.common import (
    Provenance,
    RetrievalMethod,
    SourceFamily,
    SourceType,
    utcnow,
)
from research.models.evidence import EvidenceDocument
from research.normalize.document import build_document
from research.normalize.files import EXTENSIONS, extract_file
from research.storage.store import ResearchStore

#: Larger than any paper or filing; small enough that a stray video or disk
#: image is refused rather than read into memory.
MAX_FILE_BYTES = 32 * 1024 * 1024

#: How deep a directory walk goes. Deep enough for an organised folder, not
#: deep enough to wander into a source tree.
MAX_DEPTH = 8

PROVIDER = "local"


@dataclass(slots=True)
class IngestedFile:
    """What happened to one file."""

    path: Path
    #: Set when the file became - or merged into - a document.
    result: AcquisitionResult | None = None
    skipped: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def document(self) -> EvidenceDocument | None:
        return self.result.document if self.result else None

    @property
    def is_new(self) -> bool:
        return bool(self.result and self.result.created)


@dataclass(slots=True)
class IngestReport:
    files: list[IngestedFile] = field(default_factory=list)

    @property
    def documents(self) -> list[EvidenceDocument]:
        return [entry.document for entry in self.files if entry.document is not None]

    def summary(self) -> dict[str, int]:
        return {
            "read": sum(1 for entry in self.files if entry.result is not None),
            "new_evidence": sum(1 for entry in self.files if entry.is_new),
            "already_held": sum(
                1 for entry in self.files if entry.result and not entry.result.created
            ),
            "skipped": sum(1 for entry in self.files if entry.skipped),
        }

    @property
    def skipped(self) -> list[IngestedFile]:
        return [entry for entry in self.files if entry.skipped]


class LocalIngest:
    """Reads named files and folders into one investigation."""

    def __init__(
        self,
        store: ResearchStore,
        *,
        investigation_id: str,
        ledger: BudgetLedger | None = None,
        acquirer: EvidenceAcquirer | None = None,
        max_file_bytes: int = MAX_FILE_BYTES,
        max_text_characters: int = 400_000,
    ) -> None:
        self.store = store
        self.investigation_id = investigation_id
        self.acquirer = acquirer or EvidenceAcquirer(store, ledger=ledger)
        self.max_file_bytes = max_file_bytes
        self.max_text_characters = max_text_characters

    def ingest(
        self,
        paths: Sequence[Path | str],
        *,
        source_type: SourceType = SourceType.OTHER,
        source_family: SourceFamily = SourceFamily.WEB,
        recursive: bool = True,
        task_id: str | None = None,
        notes: str | None = None,
    ) -> IngestReport:
        report = IngestReport()
        for candidate in self._collect(paths, recursive=recursive):
            report.files.append(
                self._ingest_one(
                    candidate,
                    source_type=source_type,
                    source_family=source_family,
                    task_id=task_id,
                    notes=notes,
                )
            )
        return report

    # -- walking --------------------------------------------------------
    def _collect(
        self, paths: Sequence[Path | str], *, recursive: bool
    ) -> Iterator[Path | tuple[Path, str]]:
        """Yield files to read, or ``(path, reason)`` for ones refused."""
        seen: set[Path] = set()
        for raw in paths:
            path = Path(raw).expanduser()
            if not path.exists():
                yield (path, "no such file or directory")
                continue
            root = path.resolve()
            if path.is_dir():
                yield from self._walk(root, root, recursive=recursive, seen=seen)
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved

    def _walk(
        self, directory: Path, root: Path, *, recursive: bool, seen: set[Path], depth: int = 0
    ) -> Iterator[Path | tuple[Path, str]]:
        try:
            entries = sorted(directory.iterdir())
        except OSError as exc:
            yield (directory, f"could not be read: {exc.strerror or exc}")
            return

        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_symlink():
                target = entry.resolve()
                if not _within(target, root):
                    # A link out of the named root is out of scope: the
                    # person named this folder, not wherever it points.
                    yield (entry, "symlink points outside the ingested folder")
                    continue
            if entry.is_dir():
                if not recursive:
                    continue
                if depth + 1 > MAX_DEPTH:
                    yield (entry, f"deeper than {MAX_DEPTH} levels")
                    continue
                yield from self._walk(
                    entry, root, recursive=recursive, seen=seen, depth=depth + 1
                )
                continue
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in EXTENSIONS:
                continue
            resolved = entry.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved

    # -- one file -------------------------------------------------------
    def _ingest_one(
        self,
        candidate: Path | tuple[Path, str],
        *,
        source_type: SourceType,
        source_family: SourceFamily,
        task_id: str | None,
        notes: str | None,
    ) -> IngestedFile:
        if isinstance(candidate, tuple):
            path, reason = candidate
            return IngestedFile(path=path, skipped=reason)

        path = candidate
        try:
            size = path.stat().st_size
        except OSError as exc:
            return IngestedFile(path=path, skipped=f"could not be read: {exc.strerror or exc}")
        if size > self.max_file_bytes:
            megabytes = self.max_file_bytes / 1024 / 1024
            return IngestedFile(
                path=path,
                skipped=f"{size / 1024 / 1024:.0f} MB exceeds the {megabytes:.0f} MB limit",
            )
        try:
            data = path.read_bytes()
        except OSError as exc:
            return IngestedFile(path=path, skipped=f"could not be read: {exc.strerror or exc}")

        extracted = extract_file(
            path.name, data, max_characters=self.max_text_characters
        )
        if not extracted.ok:
            return IngestedFile(
                path=path,
                skipped=extracted.warnings[0] if extracted.warnings else "no text could be read",
                warnings=list(extracted.warnings),
            )

        document = build_document(
            provider=PROVIDER,
            source_type=source_type,
            source_family=source_family,
            provenance=Provenance(
                provider=PROVIDER,
                retrieval_method=RetrievalMethod.SEED,
                retrieved_at=utcnow(),
                requested_url=None,
                task_id=task_id,
                notes=notes or f"ingested from {path}",
            ),
            title=extracted.title,
            text=extracted.text,
            authors=extracted.authors,
            # No published_at. A file's modification time is when somebody
            # saved it, which is not when the material was published, and a
            # document dated by its mtime would sit in a timeline and be
            # weighed for recency as if that date meant something.
            # The absolute path is the identity of a local file: ingesting
            # the same one twice merges rather than duplicating.
            external_id=str(path),
            metadata={
                "local_path": str(path),
                "file_name": path.name,
                "file_kind": extracted.kind,
                "file_bytes": size,
                "file_modified_at": _modified_at(path),
                "media_type": extracted.media_type,
                "ingested": True,
                **extracted.metadata,
                **({"extraction_warnings": extracted.warnings} if extracted.warnings else {}),
            },
            max_text_characters=self.max_text_characters,
        )
        result = self.acquirer.persist(document, investigation_id=self.investigation_id)
        return IngestedFile(path=path, result=result, warnings=list(extracted.warnings))


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _modified_at(path: Path) -> str | None:
    """When the file was last saved, recorded as metadata rather than as a
    publication date."""
    from datetime import datetime, timezone

    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:  # pragma: no cover - stat succeeded moments earlier
        return None


def readable_extensions() -> Iterable[str]:
    return sorted(EXTENSIONS)
