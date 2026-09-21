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
from typing import Sequence

from research import __version__
from research.acquisition.deduplicate import independent_documents
from research.cli.context import CliContext, build_context, corpus_metadata
from research.cli.format import as_json, budget_report, document_detail, document_row, table
from research.errors import NotFound, ResearchError
from research.graph.citations import CitationGraph
from research.models.common import SourceFamily
from research.models.investigation import InvestigationStatus, StopReason
from research.normalize.text import truncate
from research.operations.citation_chase import CitationChase
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
        context.store, context.registry, investigation_id=args.investigation, ledger=ledger
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
        context.store, context.registry, investigation_id=args.investigation, ledger=ledger
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
