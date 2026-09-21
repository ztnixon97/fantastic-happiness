"""The research command line.

Commands map to research operations, not to providers, and everything they
do is inspectable afterwards: which searches ran, which fetches failed, what
each document is, and whether two documents are really two sources.

There is deliberately no command that evaluates or executes anything. The
research loop has search, fetch, traverse and inspect - no shell.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Sequence

from research import __version__
from research.graph.independence import independent_documents
from research.cli.context import CliContext, build_context, corpus_metadata
from research.cli.format import (
    as_json,
    budget_report,
    claim_detail,
    document_detail,
    document_row,
    table,
    timeline_report,
)
from research.errors import IntegrityError, NotFound, ResearchError
from research.graph.citations import CitationGraph
from research.graph.entities import EntityRegistry
from research.models.claim import ClaimStatus, EvidenceStance
from research.models.event import DatePrecision
from research.normalize.html import parse_date
from research.models.common import SourceFamily
from research.models.investigation import InvestigationStatus, StopReason
from research.normalize.text import truncate
from research.operations.citation_chase import CitationChase
from research.operations.claims import ClaimOperations
from research.operations.timeline import TimelineOperations
from research.orchestration.scheduler import Scheduler
from research.orchestration.stopping import StoppingRules
from research.synthesis.synthesizer import Synthesizer
from research.operations.search import SearchOperation

FAMILY_CHOICES = {
    "academic": SourceFamily.ACADEMIC,
    "news": SourceFamily.NEWS,
    "web": SourceFamily.WEB,
    "government": SourceFamily.GOVERNMENT,
    "corporate": SourceFamily.CORPORATE,
    "social": SourceFamily.SOCIAL,
}


# ---------------------------------------------------------------- commands
async def cmd_sources(context: CliContext, args: argparse.Namespace) -> int:
    rows = context.registry.describe()
    if args.json:
        print(as_json(rows))
        return 0
    print(table(rows, ["name", "families", "search", "fetch", "references", "citing", "needs_key"]))
    missing = [row["name"] for row in rows if row["needs_key"]]
    print()
    print(f"{len(rows)} sources available" + (" (offline corpus)" if context.offline else ""))
    if not context.offline and not missing:
        print("Keyed providers are absent; keyless sources are being used.")
    return 0


async def cmd_new(context: CliContext, args: argparse.Namespace) -> int:
    investigation = context.store.investigations.create(
        args.question,
        brief=args.brief,
        budget=context.config.budget.to_dict(),
        tags=args.tag or [],
    )
    print(f"{investigation.id}  {investigation.question}")
    limits = ", ".join(f"{key}={value}" for key, value in sorted(investigation.budget.items()))
    print("Budget:", limits)
    return 0


async def cmd_list(context: CliContext, args: argparse.Namespace) -> int:
    investigations = context.store.investigations.list(limit=args.limit)
    rows = [
        {
            "id": investigation.id,
            "status": str(investigation.status),
            "documents": context.store.documents.count(investigation.id),
            "claims": len(context.store.claims.list(investigation.id, hydrate=False)),
            "created": investigation.created_at.date().isoformat(),
            "question": truncate(investigation.question, 60),
        }
        for investigation in investigations
    ]
    columns = ["id", "status", "documents", "claims", "created", "question"]
    print(as_json(rows) if args.json else table(rows, columns))
    return 0


async def cmd_search(context: CliContext, args: argparse.Namespace) -> int:
    ledger = context.ledger(args.investigation)
    operation = SearchOperation(
        context.store,
        context.registry,
        investigation_id=args.investigation,
        ledger=ledger,
        failure_threshold=context.config.acquisition.provider_failure_threshold,
    )
    outcome = await operation.search(
        args.query,
        family=FAMILY_CHOICES[args.family],
        limit=args.limit,
        objective=args.objective,
        providers=args.provider or None,
        fetch_bodies=None if args.fetch_bodies is None else args.fetch_bodies,
        max_fetches=args.max_fetches,
    )
    if args.json:
        print(as_json({"summary": outcome.summary(), "documents": outcome.document_ids}))
        return 0

    summary = outcome.summary()
    providers = ", ".join(summary["providers"]) or "no provider"
    print(f"searched {summary['family']} for {args.query!r} via {providers}")
    for provider, error in summary["provider_errors"].items():
        print(f"  ! {provider}: {error}")
    for provider, reason in summary.get("skipped_providers", {}).items():
        print(f"  - {provider}: {reason}")
    print(
        f"  {summary['candidates']} candidates -> {summary['new_evidence']} new, "
        f"{summary['duplicates']} duplicate, {summary['derived']} derived, "
        f"{summary['merged']} merged into held records, {summary['unretrieved']} unretrievable"
    )
    if summary.get("stopped_by"):
        print(f"  stopped: {summary['stopped_by']}")
    documents = context.store.documents.get_many(outcome.document_ids)
    if documents:
        print()
        print(table([document_row(document) for document in documents],
                    ["id", "type", "published", "title", "source", "independence"]))
    return 0


async def cmd_fetch(context: CliContext, args: argparse.Namespace) -> int:
    ledger = context.ledger(args.investigation)
    operation = SearchOperation(
        context.store,
        context.registry,
        investigation_id=args.investigation,
        ledger=ledger,
        failure_threshold=context.config.acquisition.provider_failure_threshold,
    )
    result = await operation.fetch_source(args.url, family=FAMILY_CHOICES[args.family])
    if result is None:
        print(
            f"could not retrieve {args.url} (see 'research activity' for the failure)",
            file=sys.stderr,
        )
        return 1
    if result.refused:
        print(f"refused: {result.refused}", file=sys.stderr)
        return 1
    print(f"{result.document.id}  {result.verdict.explanation}")
    print(document_detail(result.document, excerpt_characters=400))
    return 0


async def cmd_citations(context: CliContext, args: argparse.Namespace) -> int:
    ledger = context.ledger(args.investigation)
    chase = CitationChase(
        context.store, context.registry, investigation_id=args.investigation, ledger=ledger
    )
    result = await chase.chase(
        args.document,
        direction=args.direction,
        max_depth=args.depth,
        max_documents=args.max_documents,
        per_node_limit=args.per_node,
    )
    if args.json:
        print(as_json(result.summary()))
        return 0
    summary = result.summary()
    print(
        f"traversed {summary['direction']} from {summary['root']} to depth "
        f"{summary['depth_reached']}/{summary['max_depth']}"
    )
    print(
        f"  visited {summary['visited']} documents, added {summary['documents_added']} new, "
        f"{summary['already_held']} already held, recorded {summary['edges']} edges"
    )
    print(f"  stopped: {summary['stopped_by']}")
    for provider, error in summary["provider_errors"].items():
        print(f"  ! {provider}: {error}")
    return 0


async def cmd_evidence(context: CliContext, args: argparse.Namespace) -> int:
    documents = context.store.documents.list(
        args.investigation,
        family=FAMILY_CHOICES[args.family] if args.family else None,
        originals_only=args.independent_only,
        limit=args.limit,
    )
    if args.json:
        print(as_json([document_row(document) for document in documents]))
        return 0
    print(table([document_row(document) for document in documents],
                ["id", "type", "published", "title", "source", "independence"]))
    print()
    total = context.store.documents.count(args.investigation)
    originals = context.store.documents.count(args.investigation, originals_only=True)
    print(f"{total} documents held, {originals} not marked as copies of something else")
    return 0


async def cmd_show(context: CliContext, args: argparse.Namespace) -> int:
    document = context.store.documents.get(args.document)
    investigation_id = document.investigation_id or ""
    print(
        document_detail(
            document,
            duplicates=context.store.documents.duplicates_of(document.id),
            references=context.store.citations.references_of(investigation_id, document.id),
            citing=context.store.citations.citing_of(investigation_id, document.id),
            claims=[
                {"claim_id": link.claim_id, "stance": str(link.stance)}
                for link in context.store.claims.claims_for_document(document.id)
            ],
            excerpt_characters=args.characters,
        )
    )
    return 0


async def cmd_independence(context: CliContext, args: argparse.Namespace) -> int:
    documents = context.store.documents.list(args.investigation, limit=1000)
    groups = independent_documents(context.store.documents, [d.id for d in documents])
    by_id = {document.id: document for document in documents}
    rows = []
    for group in groups:
        head = by_id[group[0]]
        rows.append(
            {
                "group": group[0],
                "copies": len(group),
                "sources": ", ".join(
                    sorted(
                        {
                            by_id[member].canonical_host or by_id[member].provider
                            for member in group
                        }
                    )
                ),
                "title": truncate(head.title or "(untitled)", 52),
            }
        )
    if args.json:
        print(as_json(rows))
        return 0
    print(table(rows, ["group", "copies", "sources", "title"]))
    print()
    multi = [row for row in rows if row["copies"] > 1]
    print(
        f"{len(documents)} documents collapse to {len(groups)} independent sources; "
        f"{len(multi)} groups contain more than one copy"
    )
    if multi:
        print("Counting those copies separately would overstate corroboration.")
    return 0


async def cmd_graph(context: CliContext, args: argparse.Namespace) -> int:
    graph = CitationGraph(context.store, args.investigation)
    stats = graph.stats()
    if args.json:
        print(as_json({"stats": stats, "most_cited": graph.in_degree(limit=args.limit)}))
        return 0
    print(as_json(stats))
    rows = []
    for document_id, count in graph.in_degree(limit=args.limit):
        document = context.store.documents.get(document_id)
        rows.append(
            {
                "id": document_id,
                "cited by": count,
                "year": document.published_at.year if document.published_at else "-",
                "title": truncate(document.title or "(untitled)", 58),
            }
        )
    if rows:
        print()
        print("Most cited within this investigation's corpus:")
        print(table(rows, ["id", "cited by", "year", "title"]))
    frontier = graph.unresolved_frontier(limit=args.limit)
    if frontier:
        print()
        print(f"{len(frontier)} cited works known but not held (traversal frontier):")
        for edge in frontier[:10]:
            identifier = edge.cited_doi or edge.cited_external_id
            print(f"  {identifier}  {truncate(edge.cited_title or '', 56)}")
    return 0


async def cmd_budget(context: CliContext, args: argparse.Namespace) -> int:
    ledger = context.ledger(args.investigation)
    snapshot = ledger.snapshot()
    if args.json:
        print(as_json(snapshot))
        return 0
    print(budget_report(snapshot))
    exhausted = ledger.exhausted_resources()
    near = ledger.near_exhaustion()
    if exhausted:
        print("\nexhausted:", ", ".join(str(resource) for resource in exhausted))
    elif near:
        print("\nnearing limits:", ", ".join(str(resource) for resource in near))
    return 0


async def cmd_activity(context: CliContext, args: argparse.Namespace) -> int:
    queries = context.store.queries.list(args.investigation, limit=args.limit)
    fetches = context.store.fetches.list(args.investigation, limit=args.limit)
    if args.json:
        print(as_json({"searches": queries, "fetches": fetches}))
        return 0
    print("Searches")
    print(table(
        [
            {
                "id": query["id"],
                "provider": query["provider"],
                "family": query["family"],
                "results": query["result_count"],
                "status": query["status"],
                "ms": query["duration_ms"],
                "query": truncate(query["query_text"], 42),
            }
            for query in queries
        ],
        ["id", "provider", "family", "results", "status", "ms", "query"],
    ))
    print()
    print("Fetches")
    print(table(
        [
            {
                "id": fetch["id"],
                "ok": bool(fetch["ok"]),
                "status": fetch["status_code"],
                "bytes": fetch["bytes"],
                "document": fetch["document_id"],
                "url": truncate(fetch["url"], 52),
                "error": truncate(fetch["error"] or "", 30),
            }
            for fetch in fetches
        ],
        ["id", "ok", "status", "bytes", "document", "url", "error"],
    ))
    return 0


async def cmd_stop(context: CliContext, args: argparse.Namespace) -> int:
    context.store.investigations.set_status(
        args.investigation,
        InvestigationStatus.COMPLETED,
        stop_reason=StopReason.coerce(args.reason, StopReason.OPERATOR_STOPPED),
        stop_detail=args.detail,
    )
    investigation = context.store.investigations.get(args.investigation)
    print(f"{investigation.id} {investigation.status}: {investigation.stop_reason}")
    return 0


async def cmd_ui(context: CliContext, args: argparse.Namespace) -> int:
    """Serve the read-only investigation UI until interrupted."""
    import asyncio as _asyncio

    from research.ui.server import serve

    httpd = serve(context.store, host=args.host, port=args.port, config=context.config)
    investigations = context.store.investigations.list(limit=1)
    print(f"http://{args.host}:{args.port}")
    if investigations:
        print(f"showing {investigations[0].id}: {truncate(investigations[0].question, 60)}")
    else:
        print("no investigations yet; run 'research --offline investigate \"...\"' first")
    print("read-only; ctrl-c to stop")
    try:
        await _asyncio.get_running_loop().run_in_executor(None, httpd.serve_forever)
    except (KeyboardInterrupt, _asyncio.CancelledError):  # pragma: no cover - interactive
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


async def cmd_demo(context: CliContext, args: argparse.Namespace) -> int:
    """Run the worked example against the offline corpus.

    Exercises the whole Milestone 1-3 path - plan-free but complete: search
    three source families, traverse a citation graph, deduplicate, persist,
    and leave an investigation that can be inspected afterwards.
    """
    metadata = corpus_metadata(args.corpus)
    investigation = context.store.investigations.create(
        metadata["question"] or "offline demonstration",
        budget=context.config.budget.to_dict(),
        tags=["demo", "offline"],
    )
    ledger = context.ledger(investigation.id)
    operation = SearchOperation(
        context.store, context.registry, investigation_id=investigation.id, ledger=ledger
    )
    print(f"{investigation.id}  {investigation.question}")
    print(f"corpus: {metadata['academic']} papers, {metadata['web']} web/news pages\n")

    plan: list[tuple[str, SourceFamily, str]] = [
        ("small modular reactor levelized cost estimates", SourceFamily.ACADEMIC,
         "establish what the literature says about SMR costs"),
        ("nuclear construction cost overruns learning rate", SourceFamily.ACADEMIC,
         "look for evidence that challenges optimistic cost projections"),
        ("NuScale small modular reactor project data center power", SourceFamily.NEWS,
         "find recent industry activity and what was actually reported"),
        ("NRC design approval data center power agreement filing", SourceFamily.WEB,
         "reach primary regulatory and corporate material"),
    ]
    for query, family, objective in plan:
        outcome = await operation.search(query, family=family, limit=10, objective=objective)
        summary = outcome.summary()
        print(f"[{family}] {objective}")
        print(f"  {query!r} -> {summary['new_evidence']} new, {summary['duplicates']} duplicate, "
              f"{summary['derived']} derived, {summary['merged']} merged")

    seeds = [
        document
        for document in context.store.documents.list(investigation.id, family=SourceFamily.ACADEMIC)
        if document.metadata.get("cited_by_count")
    ]
    if seeds:
        seed = max(seeds, key=lambda document: document.metadata.get("cited_by_count", 0))
        chase = CitationChase(
            context.store, context.registry, investigation_id=investigation.id, ledger=ledger
        )
        result = await chase.chase(seed.id, direction="both", max_depth=2, max_documents=20)
        print(f"\n[citations] from {seed.id} {truncate(seed.title or '', 48)!r}")
        print(f"  {result.summary()}")

    documents = context.store.documents.list(investigation.id, limit=1000)
    groups = independent_documents(context.store.documents, [d.id for d in documents])
    print(f"\n{len(documents)} documents held; {len(groups)} independent sources")
    print(table([document_row(document) for document in documents],
                ["id", "type", "published", "title", "source", "independence"]))
    print()
    print("Inspect it:")
    print(f"  research show {documents[0].id if documents else 'evidence:1'}")
    print(f"  research independence {investigation.id}")
    print(f"  research graph {investigation.id}")
    print(f"  research activity {investigation.id}")
    print(f"  research budget {investigation.id}")
    return 0


async def cmd_claim_new(context: CliContext, args: argparse.Namespace) -> int:
    operations = ClaimOperations(context.store, investigation_id=args.investigation)
    claim = operations.create_claim(args.text, notes=args.note, entity_ids=args.entity or [])
    print(f"{claim.id}  {claim.text}")
    print(f"status          {claim.status} (nothing is linked to it yet)")
    return 0


async def cmd_claim_link(context: CliContext, args: argparse.Namespace) -> int:
    operations = ClaimOperations(context.store, investigation_id=args.investigation)
    try:
        result = operations.link_evidence(
            args.claim,
            args.document,
            EvidenceStance.coerce(args.stance, EvidenceStance.SUPPORTS),
            excerpt=args.excerpt,
            analysis=args.analysis,
        )
    except IntegrityError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    assessment = operations.get_claim(args.claim)
    print(f"linked {args.document} to {args.claim} as {args.stance}")
    if result.excerpt_verified:
        print("  excerpt checked against the document text")
    if result.not_independent_of:
        print(
            f"  note: {args.document} is a copy of {result.not_independent_of}; "
            "it adds no independent weight"
        )
    print(f"  {assessment.status}: {assessment.explanation}")
    return 0


async def cmd_claim_show(context: CliContext, args: argparse.Namespace) -> int:
    claim = context.store.claims.get(args.claim, hydrate=False)
    operations = ClaimOperations(context.store, investigation_id=claim.investigation_id)
    assessment = operations.get_claim(args.claim)
    if args.json:
        print(as_json(assessment.to_dict()))
        return 0
    print(claim_detail(assessment, links=context.store.claims.evidence_links(args.claim)))
    return 0


async def cmd_claim_list(context: CliContext, args: argparse.Namespace) -> int:
    operations = ClaimOperations(context.store, investigation_id=args.investigation)
    status = ClaimStatus.coerce(args.status, None) if args.status else None
    assessments = [
        operations.get_claim(claim.id)
        for claim in operations.list_claims(status=status, limit=args.limit)
    ]
    if args.json:
        print(as_json([assessment.to_dict() for assessment in assessments]))
        return 0
    rows = [
        {
            "id": assessment.claim_id,
            "status": str(assessment.status),
            "support": assessment.support.independent_count,
            "against": assessment.contradiction.independent_count,
            "claim": truncate(assessment.text, 58),
        }
        for assessment in assessments
    ]
    print(table(rows, ["id", "status", "support", "against", "claim"]))
    return 0


async def cmd_questions(context: CliContext, args: argparse.Namespace) -> int:
    operations = ClaimOperations(context.store, investigation_id=args.investigation)
    questions = operations.open_questions(limit=args.limit)
    if args.json:
        print(as_json(questions))
        return 0
    if not questions:
        print("every claim has evidence with no outstanding gaps")
        return 0
    for question in questions:
        print(f"{question['claim_id']}  [{question['status']}]  {truncate(question['text'], 66)}")
        print(f"  {question['explanation']}")
        for gap in question["gaps"]:
            print(f"  - {gap}")
        print()
    return 0


async def cmd_entities(context: CliContext, args: argparse.Namespace) -> int:
    registry = EntityRegistry(context.store, args.investigation)
    if args.from_documents:
        registered = 0
        for document in context.store.documents.list(args.investigation, limit=1000):
            registered += len(registry.register_document(document))
        print(f"registered {registered} entity mentions from document metadata")
    entities = context.store.entities.list(args.investigation, limit=args.limit)
    rows = [
        {
            "id": entity.id,
            "type": str(entity.entity_type),
            "name": truncate(entity.name, 40),
            "identifiers": ", ".join(
                identifier.key() for identifier in entity.identifiers[:2]
            ) or "-",
            "documents": len(registry.documents_for(entity.id)),
            "review": bool(entity.metadata.get("review_needed")),
        }
        for entity in entities
    ]
    if args.json:
        print(as_json(rows))
        return 0
    print(table(rows, ["id", "type", "name", "identifiers", "documents", "review"]))
    flagged = [row for row in rows if row["review"]]
    if flagged:
        print()
        print(
            f"{len(flagged)} entit{'y' if len(flagged) == 1 else 'ies'} matched on a name "
            "alone and may conflate different people or organisations"
        )
    return 0


async def cmd_event(context: CliContext, args: argparse.Namespace) -> int:
    operations = TimelineOperations(context.store, investigation_id=args.investigation)
    date_start = parse_date(args.date) if args.date else None
    if args.date and date_start is None:
        print(f"could not read the date {args.date!r}; use YYYY-MM-DD", file=sys.stderr)
        return 1
    precision = DatePrecision.coerce(
        args.precision, DatePrecision.DAY if date_start else DatePrecision.UNKNOWN
    )
    try:
        event = operations.record_event(
            args.description,
            evidence_ids=args.evidence,
            date_start=date_start,
            date_precision=precision,
            entity_ids=args.entity or [],
        )
    except IntegrityError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"{event.id}  {event.description}")
    print(f"  {event.date_start.date().isoformat() if event.date_start else 'undated'} "
          f"({event.date_precision}), evidence: {', '.join(event.evidence_ids)}")
    return 0


async def cmd_timeline(context: CliContext, args: argparse.Namespace) -> int:
    operations = TimelineOperations(context.store, investigation_id=args.investigation)
    entries = operations.build_timeline(
        include_publications=args.publications, limit=args.limit
    )
    if args.json:
        print(as_json([entry.to_dict() for entry in entries]))
        return 0
    if not entries:
        print("no events recorded yet (try --publications for the corpus chronology)")
        return 0
    print(timeline_report(entries))
    return 0


async def cmd_investigate(context: CliContext, args: argparse.Namespace) -> int:
    """Plan and run an investigation autonomously, within its budget."""
    if args.investigation:
        investigation = context.store.investigations.get(args.investigation)
    else:
        if not args.question:
            print("give a question, or --investigation to resume one", file=sys.stderr)
            return 1
        investigation = context.store.investigations.create(
            args.question, budget=context.config.budget.to_dict(), tags=args.tag or []
        )
    ledger = context.ledger(investigation.id)

    try:
        model = context.model(prefer_offline=args.model == "offline")
    except ResearchError as exc:
        print(f"no model available: {exc}", file=sys.stderr)
        return 1

    quiet = args.json
    def report(event: str, payload: dict[str, Any]) -> None:
        if quiet:
            return
        if event == "planned":
            print(f"plan: {payload['brief']}")
            for task in payload["tasks"]:
                print(f"  [{task['task_id']}] {task['role']}: {task['objective']}")
            for rejected in payload["rejected"]:
                print(f"  (rejected: {rejected})")
            print()
        elif event == "task_started":
            print(f"-> {payload['task']} {payload['role']}: {truncate(payload['objective'], 68)}")
        elif event == "task_finished":
            print(
                f"   {payload['status']} in {payload['steps']} steps; "
                f"{payload['evidence']} evidence, {payload['claims']} claims"
                + (
                    f"; stopped: {payload['stopped_by']}"
                    if payload["stopped_by"] != "completed"
                    else ""
                )
            )
        elif event == "followups":
            for child in payload["created"]:
                print(f"   + {child['task_id']} {child['role']} (depth {child['depth']}): "
                      f"{truncate(child['objective'], 56)}")
        elif event == "stopped":
            print(f"\nstopped: {payload['reason']} - {payload['detail']}")

    scheduler = Scheduler(
        context.store,
        context.registry,
        model,
        investigation_id=investigation.id,
        ledger=ledger,
        max_steps_per_task=args.steps or context.config.model.max_steps_per_task,
        on_event=report,
    )
    run = await scheduler.run(max_tasks=args.max_tasks, plan_size=args.plan_size)

    if args.json:
        print(as_json(run.summary()))
        return 0

    documents = context.store.documents.list(investigation.id, limit=1000)
    groups = independent_documents(context.store.documents, [d.id for d in documents])
    claims = context.store.claims.list(investigation.id, hydrate=False)
    print(
        f"\n{investigation.id}: {run.tasks_run} tasks, {len(documents)} documents from "
        f"{len(groups)} independent sources, {len(claims)} claims"
    )
    print("Inspect it:")
    print(f"  research claim list {investigation.id}")
    print(f"  research questions {investigation.id}")
    print(f"  research tasks {investigation.id}")
    print(f"  research budget {investigation.id}")
    return 0 if run.error is None else 1


async def cmd_report(context: CliContext, args: argparse.Namespace) -> int:
    """Write the report from stored state."""
    investigation = context.store.investigations.get(args.investigation)
    model = None
    if not args.no_summary:
        try:
            model = context.model()
        except ResearchError as exc:
            print(f"(writing without a summary: {exc})", file=sys.stderr)

    synthesis = await Synthesizer(
        context.store, investigation_id=investigation.id, model=model
    ).run()

    if args.json:
        print(as_json(synthesis.data.to_dict()))
        return 0
    if synthesis.invalid_references:
        print(
            "(the summary was discarded: it cited "
            + ", ".join(synthesis.invalid_references)
            + ", which do not exist in this investigation)",
            file=sys.stderr,
        )
    if args.output:
        path = Path(args.output).expanduser()
        path.write_text(synthesis.markdown, encoding="utf-8")
        print(f"wrote {path} ({len(synthesis.markdown)} characters)")
        return 0
    print(synthesis.markdown)
    return 0


async def cmd_find(context: CliContext, args: argparse.Namespace) -> int:
    """Search evidence the investigation already holds."""
    search = context.corpus_search(
        args.investigation, embeddings=True if args.embeddings else None
    )
    hits = await search.search(
        args.query,
        limit=args.limit,
        expand=not args.no_expand,
        collapse_copies=not args.include_copies,
        mode="all" if args.all_terms else "any",
    )
    if args.json:
        print(as_json([hit.to_dict() for hit in hits]))
        return 0

    stats = search.stats()
    if not hits:
        print(f"nothing held matches {args.query!r} ({stats['indexed']} documents indexed)")
        return 0
    print(f"{len(hits)} of {stats['indexed']} held documents match {args.query!r}\n")
    for hit in hits:
        document = hit.document
        found = "+".join(sorted(hit.ranks))
        print(f"{document.id:<12} [{found}] {truncate(document.title or '(untitled)', 62)}")
        published = document.published_at.date().isoformat() if document.published_at else "-"
        print(f"             {document.source_type} · "
              f"{document.publisher or document.canonical_host or document.provider} · {published}")
        for reason in hit.reasons[:2]:
            print(f"             why: {reason}")
        if hit.copies:
            print(f"             folds in {len(hit.copies)} cop"
                  f"{'y' if len(hit.copies) == 1 else 'ies'}: {', '.join(hit.copies)}")
        if hit.snippet:
            print(f"             \u201c{truncate(hit.snippet, 150)}\u201d")
        print()
    return 0


async def cmd_index(context: CliContext, args: argparse.Namespace) -> int:
    """Rebuild the retrieval indexes for an investigation."""
    search = context.corpus_search(
        args.investigation, embeddings=True if args.embeddings else None
    )
    indexed = search.rebuild_index()
    print(f"full-text index: {indexed} documents")
    if args.embeddings:
        if search.embeddings is None:
            print("embeddings are not configured", file=sys.stderr)
            return 1
        documents = context.store.documents.list(args.investigation, limit=2000)
        try:
            written = await search.embeddings.index(documents)
        except ResearchError as exc:
            print(f"embedding failed: {exc}", file=sys.stderr)
            return 1
        print(f"embeddings: {written} documents vectorised")
    print(as_json(search.stats()))
    return 0


async def cmd_export(context: CliContext, args: argparse.Namespace) -> int:
    """Write an investigation into an Obsidian vault."""
    from research.export.obsidian import ObsidianExporter

    investigation = context.store.investigations.get(args.investigation)
    exporter = ObsidianExporter(context.store, investigation.id)
    try:
        result = exporter.export(
            args.vault,
            folder=args.folder,
            force=args.force,
            text_limit=args.text_limit,
            canvas=not args.no_canvas,
        )
    except OSError as exc:
        print(f"could not write to the vault: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(as_json(result.summary()))
        return 0

    print(f"{len(result.written)} notes -> {result.folder}")
    if result.canvas:
        print(f"canvas: {result.canvas.name}")
    for path, reason in result.skipped:
        print(f"  skipped {path.name}: {reason}", file=sys.stderr)
    if result.skipped:
        print("  (re-run with --force to overwrite them)", file=sys.stderr)
    print()
    print("In Obsidian: open the vault, then the investigation note. Identifiers such")
    print("as claim:3 work as link targets, and the graph view shows the evidence graph.")
    return 0


async def cmd_tasks(context: CliContext, args: argparse.Namespace) -> int:
    tasks = context.store.tasks.list(args.investigation, limit=args.limit)
    if args.json:
        print(as_json([
            {
                "id": task.id,
                "parent": task.parent_task_id,
                "role": str(task.role),
                "operation": str(task.operation),
                "status": str(task.status),
                "depth": task.depth,
                "objective": task.objective,
                "summary": task.result.summary if task.result else None,
            }
            for task in tasks
        ]))
        return 0
    rows = [
        {
            "id": task.id,
            "role": str(task.role),
            "status": str(task.status),
            "depth": task.depth,
            "parent": task.parent_task_id or "-",
            "objective": truncate(task.objective, 52),
        }
        for task in tasks
    ]
    print(table(rows, ["id", "role", "status", "depth", "parent", "objective"]))
    for task in tasks:
        if task.result and task.result.summary:
            print()
            print(f"{task.id}: {truncate(task.result.summary, 200)}")
    return 0


async def cmd_status(context: CliContext, args: argparse.Namespace) -> int:
    investigation = context.store.investigations.get(args.investigation)
    ledger = context.ledger(investigation.id)
    rules = StoppingRules(context.store, investigation.id, ledger)
    documents = context.store.documents.list(investigation.id, limit=1000)
    groups = independent_documents(context.store.documents, [d.id for d in documents])
    payload = {
        "investigation": investigation.id,
        "question": investigation.question,
        "status": str(investigation.status),
        "stop_reason": str(investigation.stop_reason) if investigation.stop_reason else None,
        "stop_detail": investigation.stop_detail,
        "documents": len(documents),
        "independent_sources": len(groups),
        "claims": len(context.store.claims.list(investigation.id, hydrate=False)),
        "tasks": context.store.tasks.count(investigation.id),
        "stopping_rules": rules.describe(),
    }
    if args.json:
        print(as_json(payload))
        return 0
    print(f"{investigation.id}  {investigation.question}")
    stopped = (
        f" ({payload['stop_reason']}: {payload['stop_detail']})"
        if payload["stop_reason"]
        else ""
    )
    print(f"status          {payload['status']}{stopped}")
    print(f"evidence        {payload['documents']} documents, "
          f"{payload['independent_sources']} independent sources")
    print(f"claims          {payload['claims']}")
    print(f"tasks           {payload['tasks']}")
    print("stopping rules")
    for name, value in payload["stopping_rules"].items():
        print(f"  {name:<20} {value}")
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research",
        description="A persistent, evidence-driven research environment.",
    )
    parser.add_argument(
        "--version", action="version", version=f"research {__version__}"
    )
    parser.add_argument("--db", help="SQLite path (default ~/.research/research.sqlite3)")
    parser.add_argument("--config", help="path to a YAML/TOML/JSON config file")
    parser.add_argument("--offline", action="store_true", help="use the bundled offline corpus")
    parser.add_argument("--corpus", help="path to a corpus JSON file (implies --offline)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sources = subparsers.add_parser("sources", help="list available sources and capabilities")
    sources.set_defaults(handler=cmd_sources)

    new = subparsers.add_parser("new", help="start an investigation")
    new.add_argument("question")
    new.add_argument("--brief", help="longer statement of what is being asked")
    new.add_argument("--tag", action="append")
    new.set_defaults(handler=cmd_new)

    listing = subparsers.add_parser("list", help="list investigations")
    listing.add_argument("--limit", type=int, default=25)
    listing.set_defaults(handler=cmd_list)

    search = subparsers.add_parser("search", help="search a source family and persist evidence")
    search.add_argument("investigation")
    search.add_argument("query")
    search.add_argument("--family", choices=sorted(FAMILY_CHOICES), default="web")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--objective", help="why this search is being run")
    search.add_argument("--provider", action="append", help="restrict to named providers")
    search.add_argument("--max-fetches", type=int, default=None, dest="max_fetches")
    search.add_argument("--fetch-bodies", dest="fetch_bodies", action="store_true", default=None)
    search.add_argument("--no-fetch-bodies", dest="fetch_bodies", action="store_false")
    search.set_defaults(handler=cmd_search)

    fetch = subparsers.add_parser("fetch", help="retrieve one URL as evidence")
    fetch.add_argument("investigation")
    fetch.add_argument("url")
    fetch.add_argument("--family", choices=sorted(FAMILY_CHOICES), default="web")
    fetch.set_defaults(handler=cmd_fetch)

    citations = subparsers.add_parser("citations", help="traverse the citation graph")
    citations.add_argument("investigation")
    citations.add_argument("document")
    citations.add_argument(
        "--direction", choices=["backward", "forward", "both"], default="backward"
    )
    citations.add_argument("--depth", type=int, default=1)
    citations.add_argument("--max-documents", type=int, default=25, dest="max_documents")
    citations.add_argument("--per-node", type=int, default=20, dest="per_node")
    citations.set_defaults(handler=cmd_citations)

    evidence = subparsers.add_parser("evidence", help="list held evidence")
    evidence.add_argument("investigation")
    evidence.add_argument("--family", choices=sorted(FAMILY_CHOICES))
    evidence.add_argument("--independent-only", action="store_true", dest="independent_only")
    evidence.add_argument("--limit", type=int, default=100)
    evidence.set_defaults(handler=cmd_evidence)

    show = subparsers.add_parser("show", help="inspect one document and its provenance")
    show.add_argument("document")
    show.add_argument("--characters", type=int, default=1200)
    show.set_defaults(handler=cmd_show)

    independence = subparsers.add_parser(
        "independence", help="show which documents count as one source"
    )
    independence.add_argument("investigation")
    independence.set_defaults(handler=cmd_independence)

    graph = subparsers.add_parser("graph", help="citation graph summary")
    graph.add_argument("investigation")
    graph.add_argument("--limit", type=int, default=15)
    graph.set_defaults(handler=cmd_graph)

    budget = subparsers.add_parser("budget", help="show budget usage")
    budget.add_argument("investigation")
    budget.set_defaults(handler=cmd_budget)

    activity = subparsers.add_parser("activity", help="searches and fetches performed")
    activity.add_argument("investigation")
    activity.add_argument("--limit", type=int, default=25)
    activity.set_defaults(handler=cmd_activity)

    stop = subparsers.add_parser("stop", help="close an investigation and record why")
    stop.add_argument("investigation")
    stop.add_argument("--reason", default="operator_stopped")
    stop.add_argument("--detail")
    stop.set_defaults(handler=cmd_stop)

    investigate = subparsers.add_parser(
        "investigate", help="plan and run an investigation autonomously"
    )
    investigate.add_argument("question", nargs="?")
    investigate.add_argument(
        "--investigation", help="resume an existing investigation instead of starting one"
    )
    investigate.add_argument("--tag", action="append")
    investigate.add_argument("--max-tasks", type=int, default=None, dest="max_tasks")
    investigate.add_argument("--plan-size", type=int, default=4, dest="plan_size")
    investigate.add_argument("--steps", type=int, default=None, help="steps per task")
    investigate.add_argument(
        "--model", choices=["auto", "offline"], default="auto",
        help=(
            "'offline' uses the built-in rule-based researcher against whatever "
            "sources are configured: a smoke test that needs no credential"
        ),
    )
    investigate.set_defaults(handler=cmd_investigate)

    report = subparsers.add_parser("report", help="write the report from stored state")
    report.add_argument("investigation")
    report.add_argument("--output", help="write to a file instead of stdout")
    report.add_argument(
        "--no-summary", action="store_true", dest="no_summary",
        help="assemble the report without asking a model for prose",
    )
    report.set_defaults(handler=cmd_report)

    find = subparsers.add_parser(
        "find", help="search evidence this investigation already holds"
    )
    find.add_argument("investigation")
    find.add_argument("query")
    find.add_argument("--limit", type=int, default=8)
    find.add_argument(
        "--no-expand", action="store_true", dest="no_expand",
        help="lexical matching only, without expanding through the graph",
    )
    find.add_argument(
        "--include-copies", action="store_true", dest="include_copies",
        help="list syndicated copies as separate results",
    )
    find.add_argument(
        "--all-terms", action="store_true", dest="all_terms",
        help="require every term rather than ranking by overlap",
    )
    find.add_argument(
        "--embeddings", action="store_true",
        help="also rank by vector similarity (needs an embedding provider)",
    )
    find.set_defaults(handler=cmd_find)

    index = subparsers.add_parser("index", help="rebuild retrieval indexes")
    index.add_argument("investigation")
    index.add_argument(
        "--embeddings", action="store_true", help="also compute document vectors"
    )
    index.set_defaults(handler=cmd_index)

    export = subparsers.add_parser("export", help="export an investigation to other tools")
    export_targets = export.add_subparsers(dest="export_command", required=True)

    obsidian = export_targets.add_parser(
        "obsidian", help="write notes into an Obsidian vault"
    )
    obsidian.add_argument("investigation")
    obsidian.add_argument("vault", help="path to the vault folder")
    obsidian.add_argument(
        "--folder", default="Research", help="folder inside the vault (default: Research)"
    )
    obsidian.add_argument(
        "--force", action="store_true",
        help="overwrite notes that were not generated by this export",
    )
    obsidian.add_argument(
        "--text-limit", type=int, default=20000, dest="text_limit",
        help="characters of retrieved text to include per note",
    )
    obsidian.add_argument(
        "--no-canvas", action="store_true", dest="no_canvas",
        help="skip the claim/evidence canvas",
    )
    obsidian.set_defaults(handler=cmd_export)

    tasks = subparsers.add_parser("tasks", help="the research task tree")
    tasks.add_argument("investigation")
    tasks.add_argument("--limit", type=int, default=50)
    tasks.set_defaults(handler=cmd_tasks)

    status = subparsers.add_parser("status", help="where an investigation stands")
    status.add_argument("investigation")
    status.set_defaults(handler=cmd_status)

    claim = subparsers.add_parser("claim", help="state, link and inspect claims")
    claim_actions = claim.add_subparsers(dest="claim_command", required=True)

    claim_new = claim_actions.add_parser("new", help="state a proposition")
    claim_new.add_argument("investigation")
    claim_new.add_argument("text")
    claim_new.add_argument("--note")
    claim_new.add_argument("--entity", action="append")
    claim_new.set_defaults(handler=cmd_claim_new)

    claim_link = claim_actions.add_parser("link", help="attach evidence to a claim")
    claim_link.add_argument("investigation")
    claim_link.add_argument("claim")
    claim_link.add_argument("document")
    claim_link.add_argument(
        "--stance", choices=["supports", "contradicts", "qualifies", "mentions"],
        default="supports",
    )
    claim_link.add_argument(
        "--excerpt", help="verbatim quotation; checked against the document"
    )
    claim_link.add_argument("--analysis", help="reasoning about the document (not evidence)")
    claim_link.set_defaults(handler=cmd_claim_link)

    claim_show = claim_actions.add_parser("show", help="inspect a claim and its evidence")
    claim_show.add_argument("claim")
    claim_show.set_defaults(handler=cmd_claim_show)

    claim_list = claim_actions.add_parser("list", help="list claims and their status")
    claim_list.add_argument("investigation")
    claim_list.add_argument("--status")
    claim_list.add_argument("--limit", type=int, default=100)
    claim_list.set_defaults(handler=cmd_claim_list)

    questions = subparsers.add_parser(
        "questions", help="claims whose evidence is thin, and what would fix them"
    )
    questions.add_argument("investigation")
    questions.add_argument("--limit", type=int, default=50)
    questions.set_defaults(handler=cmd_questions)

    entities = subparsers.add_parser("entities", help="the entity registry")
    entities.add_argument("investigation")
    entities.add_argument(
        "--from-documents", action="store_true", dest="from_documents",
        help="register authors and publishers from document metadata first",
    )
    entities.add_argument("--limit", type=int, default=100)
    entities.set_defaults(handler=cmd_entities)

    event = subparsers.add_parser("event", help="record an evidence-backed event")
    event.add_argument("investigation")
    event.add_argument("description")
    event.add_argument(
        "--evidence", action="append", required=True,
        help="document id backing this event (repeatable, at least one)",
    )
    event.add_argument("--date", help="YYYY-MM-DD")
    event.add_argument(
        "--precision", choices=["exact", "day", "month", "quarter", "year", "unknown"],
        default=None,
    )
    event.add_argument("--entity", action="append")
    event.set_defaults(handler=cmd_event)

    timeline = subparsers.add_parser("timeline", help="evidence-backed chronology")
    timeline.add_argument("investigation")
    timeline.add_argument(
        "--publications", action="store_true",
        help="include when each independent source published",
    )
    timeline.add_argument("--limit", type=int, default=100)
    timeline.set_defaults(handler=cmd_timeline)

    ui = subparsers.add_parser("ui", help="serve the read-only investigation UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.set_defaults(handler=cmd_ui)

    demo = subparsers.add_parser("demo", help="run the offline worked example")
    demo.set_defaults(handler=cmd_demo)

    return parser


async def _run(args: argparse.Namespace) -> int:
    context = build_context(
        db_path=args.db,
        config_path=args.config,
        offline=args.offline or args.command == "demo",
        corpus_path=args.corpus,
    )
    try:
        return await args.handler(context, args)
    finally:
        await context.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "corpus"):
        args.corpus = None
    try:
        return asyncio.run(_run(args))
    except NotFound as exc:
        print(f"not found: {exc}", file=sys.stderr)
        return 2
    except ResearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
