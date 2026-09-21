"""Executing a worker's chosen action.

Everything a model asks for passes through here, and nothing else reaches the
operations layer. The runtime:

* refuses actions the worker's role is not allowed to use,
* validates every argument, including identifier syntax, before use,
* executes through the ordinary operations - the same code the CLI calls,
* returns a *compact* observation: identifiers, counts and one-line
  descriptions, never page text.

That last point is the load-bearing one. A worker reads a document only by
asking for it, and gets it back labelled as untrusted external evidence.
Search results never arrive as text, so a page cannot address the model
simply by being found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.acquisition.untrusted import as_external_evidence, as_external_passages
from research.agents.actions import ACTIONS, ActionSpec
from research.errors import BudgetExceeded, IntegrityError, NotFound, ResearchError
from research.graph.entities import EntityRegistry
from research.ids import is_id
from research.models.claim import EvidenceStance
from research.models.common import SourceFamily
from research.models.entity import EntityIdentifier, EntityType
from research.models.event import DatePrecision
from research.models.evidence import EvidenceDocument
from research.models.task import FollowUp, Operation, ResearchRole, ResearchTask, TaskStatus
from research.normalize.html import parse_date
from research.normalize.text import truncate
from research.operations.citation_chase import CitationChase
from research.operations.claims import ClaimOperations
from research.operations.counterevidence import CounterevidenceSearch
from research.operations.primary_source import PrimarySourceChase
from research.operations.search import SearchOperation
from research.operations.timeline import TimelineOperations
from research.retrieval.passages import TARGET_CHARACTERS as PASSAGE_CHARACTERS
from research.retrieval.passages import select_passages
from research.retrieval.search import CorpusSearch
from research.budgets import BudgetLedger, Resource
from research.sources.registry import SourceRegistry
from research.storage.store import ResearchStore

#: How much of a document a worker may read in one go.
DEFAULT_READ_CHARACTERS = 4000
MAX_READ_CHARACTERS = 12000


@dataclass(slots=True)
class ActionRequest:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    thought: str | None = None


@dataclass(slots=True)
class Observation:
    """What the worker learns from an action. Compact by construction."""

    action: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    terminal: bool = False

    #: Lists in an observation are capped by *length*, not by truncating the
    #: serialised text. Cutting JSON mid-object hands the worker something it
    #: cannot parse, which is worse than handing it less.
    MAX_LIST_ITEMS = 12

    def render(self, *, limit: int = 6000) -> str:
        import json

        if not self.ok:
            return f"{self.action} failed: {self.error}"
        body = json.dumps(
            _cap(self.data, self.MAX_LIST_ITEMS),
            ensure_ascii=False,
            default=str,
            sort_keys=True,
        )
        if len(body) > limit:
            # Still too big: shed list contents entirely rather than emit
            # broken JSON, and say how much was withheld.
            body = json.dumps(
                _cap(self.data, 3), ensure_ascii=False, default=str, sort_keys=True
            )
        return f"{self.action} -> {truncate(body, limit)}"


def _cap(value: Any, limit: int) -> Any:
    """Shorten long lists in place of truncating serialised output.

    The list stays homogeneous - a count of what was withheld goes in a
    sibling key, not as a string appended to a list of objects. A consumer
    that iterates the list should not have to guess at the type of the last
    element.
    """
    if isinstance(value, dict):
        capped: dict[str, Any] = {}
        for key, item in value.items():
            capped[key] = _cap(item, limit)
            if isinstance(item, list) and len(item) > limit:
                capped[f"{key}_omitted"] = len(item) - limit
        return capped
    if isinstance(value, list):
        return [_cap(item, limit) for item in value[:limit]]
    return value


def document_brief(document: EvidenceDocument) -> dict[str, Any]:
    """The shape a worker sees a document in before reading it."""
    brief: dict[str, Any] = {
        "id": document.id,
        "title": truncate(document.title or "(untitled)", 120),
        "type": str(document.source_type),
        "source": document.publisher or document.canonical_host or document.provider,
        "published": document.published_at.date().isoformat() if document.published_at else None,
    }
    if document.duplicate_of or document.derived_from:
        brief["not_independent_of"] = document.duplicate_of or document.derived_from
        brief["relation"] = str(document.duplicate_relation)
    if document.metadata.get("is_retracted"):
        brief["retracted"] = True
    if document.doi:
        brief["doi"] = document.doi
    return brief


class ActionRuntime:
    """Dispatches validated actions for one running task."""

    def __init__(
        self,
        store: ResearchStore,
        registry: SourceRegistry,
        *,
        investigation_id: str,
        task: ResearchTask,
        ledger: BudgetLedger,
        corpus: CorpusSearch | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.investigation_id = investigation_id
        self.task = task
        self.ledger = ledger

        self.search = SearchOperation(
            store, registry, investigation_id=investigation_id, ledger=ledger, task_id=task.id
        )
        self.citations = CitationChase(
            store, registry, investigation_id=investigation_id, ledger=ledger, task_id=task.id
        )
        self.claims = ClaimOperations(
            store, investigation_id=investigation_id, task_id=task.id
        )
        self.timeline = TimelineOperations(
            store, investigation_id=investigation_id, task_id=task.id
        )
        self.entities = EntityRegistry(store, investigation_id, task_id=task.id)
        self.primary_source = PrimarySourceChase(
            store, registry, self.search, investigation_id=investigation_id, task_id=task.id
        )
        #: Retrieval over held evidence. Lexical and graph only by default: a
        #: worker should not need an embedding provider - or its credentials -
        #: to ask what the investigation already has.
        self.corpus = corpus or CorpusSearch(store, investigation_id)
        self.counterevidence = CounterevidenceSearch(
            store,
            self.search,
            investigation_id=investigation_id,
            citation_chase=self.citations,
            task_id=task.id,
        )
        #: Populated when the worker completes; read by the scheduler.
        self.result_payload: dict[str, Any] | None = None
        self.spawned: list[ResearchTask] = []

    # -- dispatch -------------------------------------------------------
    async def execute(self, request: ActionRequest) -> Observation:
        spec = ACTIONS.get(request.name)
        if spec is None:
            return Observation(
                action=request.name,
                ok=False,
                error=(
                    f"unknown action {request.name!r}; choose one of: "
                    + ", ".join(sorted(ACTIONS))
                ),
            )
        if not spec.allows(self.task.role):
            return Observation(
                action=spec.name,
                ok=False,
                error=(
                    f"the {self.task.role} role cannot use {spec.name}; "
                    "spawn a task for the role that can"
                ),
            )
        missing = [key for key in spec.required if not request.arguments.get(key)]
        if missing:
            return Observation(
                action=spec.name,
                ok=False,
                error=f"missing required argument(s): {', '.join(missing)}",
            )

        handler = getattr(self, f"_do_{spec.name}")
        try:
            return await handler(spec, request.arguments)
        except BudgetExceeded as exc:
            return Observation(action=spec.name, ok=False, error=f"budget: {exc}")
        except (IntegrityError, NotFound) as exc:
            return Observation(action=spec.name, ok=False, error=str(exc))
        except ResearchError as exc:
            return Observation(action=spec.name, ok=False, error=f"{type(exc).__name__}: {exc}")

    # -- searching ------------------------------------------------------
    async def _search(
        self, family: SourceFamily, arguments: dict[str, Any], action: str
    ) -> Observation:
        outcome = await self.search.search(
            str(arguments["query"]),
            family=family,
            limit=_int(arguments.get("limit"), default=10, maximum=25),
            objective=arguments.get("objective") or self.task.objective,
            published_after=parse_date(arguments.get("published_after")),
            published_before=parse_date(arguments.get("published_before")),
        )
        documents = self.store.documents.get_many(outcome.document_ids)
        summary = outcome.summary()
        if outcome.stopped_by and "no configured provider" in outcome.stopped_by:
            return Observation(
                action=action,
                ok=False,
                error=(
                    f"{outcome.stopped_by}. Try a different source family, or "
                    "fetch a known URL directly."
                ),
            )
        return Observation(
            action=action,
            ok=True,
            data={
                "query": outcome.query_text,
                "providers": summary["providers"],
                "new_evidence": summary["new_evidence"],
                "already_held_or_copies": (
                    summary["duplicates"] + summary["merged"] + summary["derived"]
                ),
                "results": [document_brief(document) for document in documents],
                **(
                    {"provider_errors": summary["provider_errors"]}
                    if summary["provider_errors"]
                    else {}
                ),
                **({"stopped_by": summary["stopped_by"]} if summary.get("stopped_by") else {}),
            },
        )

    async def _do_search_news(self, spec: ActionSpec, arguments: dict[str, Any]) -> Observation:
        return await self._search(SourceFamily.NEWS, arguments, spec.name)

    async def _do_search_academic(
        self, spec: ActionSpec, arguments: dict[str, Any]
    ) -> Observation:
        return await self._search(SourceFamily.ACADEMIC, arguments, spec.name)

    async def _do_search_web(self, spec: ActionSpec, arguments: dict[str, Any]) -> Observation:
        return await self._search(SourceFamily.WEB, arguments, spec.name)

    async def _do_search_social(
        self, spec: ActionSpec, arguments: dict[str, Any]
    ) -> Observation:
        return await self._search(SourceFamily.SOCIAL, arguments, spec.name)

    async def _do_search_corpus(self, spec: ActionSpec, arguments: dict[str, Any]) -> Observation:
        query = str(arguments["query"])
        limit = _int(arguments.get("limit"), default=10, maximum=25)
        expand = arguments.get("expand") is not False
        hits = await self.corpus.search(query, limit=limit, expand=expand)
        held = self.store.documents.count(self.investigation_id)
        results = []
        for hit in hits:
            entry = document_brief(hit.document)
            if hit.reasons:
                entry["why"] = hit.reasons[:3]
            if hit.snippet:
                entry["matched_text"] = truncate(hit.snippet, 240)
            if hit.copies:
                entry["copies_folded_in"] = len(hit.copies)
            results.append(entry)
        return Observation(
            action=spec.name,
            ok=True,
            data={
                "query": query,
                "held_documents": held,
                "matched": len(hits),
                "results": results,
                **(
                    {}
                    if hits
                    else {
                        "note": (
                            "nothing held matches; this is the moment to search "
                            "outside, not to rephrase"
                        )
                    }
                ),
            },
        )

    async def _do_fetch_source(self, spec: ActionSpec, arguments: dict[str, Any]) -> Observation:
        result = await self.search.fetch_source(str(arguments["url"]), family=SourceFamily.WEB)
        if result is None:
            return Observation(
                action=spec.name,
                ok=False,
                error="the URL could not be retrieved; it is recorded as a failed fetch",
            )
        if result.refused:
            return Observation(action=spec.name, ok=False, error=result.refused)
        return Observation(
            action=spec.name,
            ok=True,
            data={
                "document": document_brief(result.document),
                "new_evidence": result.is_new_evidence,
                "assessment": result.verdict.explanation,
            },
        )

    async def _do_follow_citations(
        self, spec: ActionSpec, arguments: dict[str, Any]
    ) -> Observation:
        document_id = _require_id(arguments["document_id"], "evidence")
        direction = str(arguments.get("direction") or "backward")
        if direction not in ("backward", "forward", "both"):
            return Observation(
                action=spec.name, ok=False, error="direction must be backward, forward or both"
            )
        result = await self.citations.chase(
            document_id,
            direction=direction,  # type: ignore[arg-type]
            max_depth=_int(arguments.get("depth"), default=1, maximum=3),
            max_documents=_int(arguments.get("limit"), default=15, maximum=40),
        )
        added = self.store.documents.get_many(result.added_document_ids)
        return Observation(
            action=spec.name,
            ok=True,
            data={
                **result.summary(),
                "new_documents": [document_brief(document) for document in added],
            },
        )

    async def _do_find_primary_source(
        self, spec: ActionSpec, arguments: dict[str, Any]
    ) -> Observation:
        document_id = _require_id(arguments["document_id"], "evidence")
        result = await self.primary_source.chase(
            document_id, assertion=arguments.get("assertion")
        )
        acquired = self.store.documents.get_many(result.acquired)
        return Observation(
            action=spec.name,
            ok=True,
            data={
                **result.summary(),
                "documents": [document_brief(document) for document in acquired],
            },
        )

    async def _do_find_counterevidence(
        self, spec: ActionSpec, arguments: dict[str, Any]
    ) -> Observation:
        claim_id = arguments.get("claim_id")
        if claim_id:
            claim_id = _require_id(claim_id, "claim")
        result = await self.counterevidence.run(
            str(arguments["proposition"]),
            claim_id=claim_id,
            limit=_int(arguments.get("limit"), default=6, maximum=15),
        )
        found = self.store.documents.get_many(sorted(set(result.new_document_ids)))
        return Observation(
            action=spec.name,
            ok=True,
            data={
                **result.summary(),
                "documents": [document_brief(document) for document in found],
            },
        )

    # -- the graph ------------------------------------------------------
    async def _do_create_claim(self, spec: ActionSpec, arguments: dict[str, Any]) -> Observation:
        claim = self.claims.create_claim(
            str(arguments["text"]), notes=arguments.get("notes")
        )
        return Observation(
            action=spec.name,
            ok=True,
            data={"claim_id": claim.id, "text": claim.text, "status": str(claim.status)},
        )

    async def _do_link_evidence(self, spec: ActionSpec, arguments: dict[str, Any]) -> Observation:
        stance = EvidenceStance.coerce(arguments.get("stance"), None)
        if stance is None:
            return Observation(
                action=spec.name,
                ok=False,
                error="stance must be supports, contradicts, qualifies or mentions",
            )
        result = self.claims.link_evidence(
            _require_id(arguments["claim_id"], "claim"),
            _require_id(arguments["document_id"], "evidence"),
            stance,
            excerpt=arguments.get("excerpt"),
            analysis=arguments.get("analysis"),
        )
        assessment = self.claims.get_claim(result.claim.id)
        return Observation(
            action=spec.name,
            ok=True,
            data={
                "claim_id": result.claim.id,
                "status": str(assessment.status),
                "assessment": assessment.explanation,
                "excerpt_verified": result.excerpt_verified,
                **(
                    {"not_independent_of": result.not_independent_of}
                    if result.not_independent_of
                    else {}
                ),
            },
        )

    async def _do_get_claim(self, spec: ActionSpec, arguments: dict[str, Any]) -> Observation:
        claim_id = arguments.get("claim_id")
        if claim_id:
            assessment = self.claims.get_claim(_require_id(claim_id, "claim"))
            return Observation(action=spec.name, ok=True, data=assessment.to_dict())
        claims = [
            {
                "claim_id": assessment.claim_id,
                "text": assessment.text,
                "status": str(assessment.status),
                "assessment": assessment.explanation,
            }
            for assessment in self.claims.graph.assess_all(limit=40)
        ]
        return Observation(action=spec.name, ok=True, data={"claims": claims})

    async def _do_get_evidence(self, spec: ActionSpec, arguments: dict[str, Any]) -> Observation:
        document = self.store.documents.get(_require_id(arguments["document_id"], "evidence"))
        characters = _int(
            arguments.get("characters"),
            default=DEFAULT_READ_CHARACTERS,
            maximum=MAX_READ_CHARACTERS,
        )
        # A worker reading a document is reading it *for* something. When it
        # does not say what, the task's own objective is the honest default -
        # better than the first page, which is what a document has instead of
        # an answer.
        question = str(arguments.get("about") or "").strip() or self.task.objective
        whole = bool(arguments.get("whole")) or not question

        body = document.best_text or ""
        passages: list[Any] = []
        if not whole and len(body) > characters:
            passages = await select_passages(
                body,
                question,
                limit=max(2, characters // PASSAGE_CHARACTERS),
                embedder=getattr(self.corpus.embeddings, "client", None),
            )

        if passages:
            content = as_external_passages(document, passages, question=question)
        else:
            content = as_external_evidence(document, limit=characters)

        return Observation(
            action=spec.name,
            ok=True,
            data={
                **document_brief(document),
                "read": "passages" if passages else "from the beginning",
                **({"selected_for": question} if passages else {}),
                **(
                    {"passages": [passage.locate() for passage in passages]}
                    if passages
                    else {}
                ),
                # Wrapped and labelled: the worker is told, in the payload
                # itself, that this is external material to assess.
                "content": content,
            },
        )

    async def _do_get_open_questions(
        self, spec: ActionSpec, arguments: dict[str, Any]
    ) -> Observation:
        return Observation(
            action=spec.name,
            ok=True,
            data={"open_questions": self.claims.open_questions(limit=20)},
        )

    async def _do_resolve_entity(self, spec: ActionSpec, arguments: dict[str, Any]) -> Observation:
        entity_type = EntityType.coerce(arguments.get("entity_type"), None)
        if entity_type is None:
            return Observation(
                action=spec.name,
                ok=False,
                error=(
                    "entity_type must be person, organization, location, "
                    "academic_paper or product"
                ),
            )
        raw_identifiers = arguments.get("identifiers") or {}
        identifiers = (
            [
                EntityIdentifier(scheme=str(scheme), value=str(value))
                for scheme, value in raw_identifiers.items()
                if scheme and value
            ]
            if isinstance(raw_identifiers, dict)
            else []
        )

        resolution = self.entities.resolve(
            str(arguments["name"]), entity_type, identifiers=identifiers
        )
        data: dict[str, Any] = {"confidence": str(resolution.confidence)}
        if resolution.entity:
            data.update(
                {
                    "entity_id": resolution.entity.id,
                    "name": resolution.entity.name,
                    "created": resolution.created,
                    "matched_on": resolution.matched_on,
                }
            )
        else:
            data["candidates"] = [
                {"entity_id": candidate.id, "name": candidate.name}
                for candidate in resolution.candidates
            ]
            data["note"] = "ambiguous: nothing was merged; pick one or supply an identifier"
        return Observation(action=spec.name, ok=True, data=data)

    async def _do_build_timeline(
        self, spec: ActionSpec, arguments: dict[str, Any]
    ) -> Observation:
        description = arguments.get("description")
        if not description:
            entries = self.timeline.build_timeline(include_publications=True, limit=40)
            return Observation(
                action=spec.name,
                ok=True,
                data={"timeline": [entry.to_dict() for entry in entries]},
            )
        evidence_ids = [
            _require_id(value, "evidence")
            for value in _as_list(arguments.get("evidence_ids"))
        ]
        date_start = parse_date(arguments.get("date"))
        precision = DatePrecision.coerce(
            arguments.get("precision"), DatePrecision.DAY if date_start else DatePrecision.UNKNOWN
        )
        event = self.timeline.record_event(
            str(description),
            evidence_ids=evidence_ids,
            date_start=date_start,
            date_precision=precision,
            entity_ids=[
                _require_id(value, "entity") for value in _as_list(arguments.get("entity_ids"))
            ],
        )
        return Observation(
            action=spec.name,
            ok=True,
            data={
                "event_id": event.id,
                "date": event.date_start.date().isoformat() if event.date_start else None,
                "precision": str(event.date_precision),
            },
        )

    # -- delegation and completion --------------------------------------
    async def _do_spawn_research_task(
        self, spec: ActionSpec, arguments: dict[str, Any]
    ) -> Observation:
        role = ResearchRole.coerce(arguments.get("role"), None)
        if role is None:
            return Observation(
                action=spec.name,
                ok=False,
                error=(
                    "role must be one of: scout, academic, news, primary_source, "
                    "social, skeptic"
                ),
            )
        depth = self.task.depth + 1
        if not self.ledger.allows_depth(depth):
            return Observation(
                action=spec.name,
                ok=False,
                error=(
                    f"depth limit reached ({self.ledger.policy.max_depth}); "
                    "finish this task instead"
                ),
            )
        if not self.ledger.can_spend(Resource.TASKS):
            return Observation(action=spec.name, ok=False, error="task budget exhausted")

        operation = Operation.coerce(arguments.get("operation"), _default_operation(role))
        child = self.store.tasks.create(
            ResearchTask(
                id="",
                investigation_id=self.investigation_id,
                parent_task_id=self.task.id,
                role=role,
                operation=operation,
                objective=str(arguments["objective"]),
                depth=depth,
                priority=_int(arguments.get("priority"), default=5, maximum=9),
                status=TaskStatus.PENDING,
            )
        )
        self.ledger.try_spend(Resource.TASKS)
        self.spawned.append(child)
        return Observation(
            action=spec.name,
            ok=True,
            data={
                "task_id": child.id,
                "role": str(child.role),
                "objective": child.objective,
                "depth": child.depth,
            },
        )

    async def _do_complete_research_task(
        self, spec: ActionSpec, arguments: dict[str, Any]
    ) -> Observation:
        self.result_payload = {
            "summary": str(arguments.get("summary", "")),
            "claims": [
                value
                for value in _as_list(arguments.get("claim_ids"))
                if is_id(str(value), "claim")
            ],
            "evidence": [
                value
                for value in _as_list(arguments.get("evidence_ids"))
                if is_id(str(value), "evidence")
            ],
            "open_questions": [str(value) for value in _as_list(arguments.get("open_questions"))],
            "recommended_followups": _parse_followups(arguments.get("followups")),
        }
        return Observation(
            action=spec.name,
            ok=True,
            terminal=True,
            data={"recorded": True, "summary": truncate(self.result_payload["summary"], 200)},
        )


# ------------------------------------------------------------------ helpers
def _int(value: Any, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, maximum))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _require_id(value: Any, prefix: str) -> str:
    text = str(value).strip()
    if not is_id(text, prefix):
        raise IntegrityError(
            f"{text!r} is not a {prefix} identifier; expected something like {prefix}:12"
        )
    return text


def _default_operation(role: ResearchRole) -> Operation:
    return {
        ResearchRole.ACADEMIC: Operation.SEARCH_ACADEMIC,
        ResearchRole.NEWS: Operation.SEARCH_NEWS,
        ResearchRole.PRIMARY_SOURCE: Operation.FIND_PRIMARY_SOURCE,
        ResearchRole.SKEPTIC: Operation.FIND_COUNTEREVIDENCE,
        ResearchRole.SOCIAL: Operation.SEARCH_SOCIAL,
        ResearchRole.SCOUT: Operation.SEARCH_WEB,
    }.get(role, Operation.SEARCH_WEB)


def _parse_followups(value: Any) -> list[dict[str, Any]]:
    followups: list[dict[str, Any]] = []
    for raw in _as_list(value):
        if isinstance(raw, str):
            followups.append(
                FollowUp(operation=Operation.SEARCH_WEB, objective=raw).to_dict()
            )
            continue
        if not isinstance(raw, dict) or not raw.get("objective"):
            continue
        followups.append(
            FollowUp(
                operation=Operation.coerce(raw.get("operation"), Operation.SEARCH_WEB),
                objective=str(raw["objective"]),
                rationale=raw.get("rationale"),
                priority=_int(raw.get("priority"), default=5, maximum=9),
            ).to_dict()
        )
    return followups
