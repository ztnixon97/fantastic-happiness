"""What a whole investigation produced.

Not a score. The surveyed tools that measure end-to-end quality do it with
an LLM judge, which grades prose and cannot tell a well-written report from
a well-evidenced one. What is measurable here without a judge is whether
the record a run leaves behind has the properties this system claims for
it: claims backed by independent sources, quotations that appear in the
documents they are attributed to, a primary record somewhere, and a stated
reason for stopping.

Those are the things that would be wrong if the system were broken, and
they are checkable against the store rather than against an opinion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from research.budgets import BudgetLedger, Resource
from research.config import BudgetPolicy, ResearchConfig
from research.graph.claims import ClaimGraph
from research.models.common import PRIMARY_SOURCE_DISTANCE
from research.operations.claims import excerpt_appears_in
from research.orchestration.scheduler import Scheduler
from research.sources.offline import build_offline_registry, load_corpus
from research.storage.store import ResearchStore


@dataclass(slots=True)
class RunReport:
    question: str
    seconds: float = 0.0
    tasks: int = 0
    stop_reason: str = ""
    stop_detail: str = ""
    documents: int = 0
    independent_sources: int = 0
    claims: int = 0
    claims_with_evidence: int = 0
    claims_with_independent_support: int = 0
    claims_with_a_primary_record: int = 0
    claims_with_counterevidence: int = 0
    excerpts: int = 0
    #: Should always be zero: an excerpt that does not appear in its document
    #: is refused at write time, so a non-zero count here is a broken
    #: guarantee rather than a quality problem.
    unverifiable_excerpts: list[str] = field(default_factory=list)
    open_questions: int = 0
    model_calls: int = 0
    tokens: float = 0.0
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "seconds": round(self.seconds, 2),
            "tasks": self.tasks,
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
            "documents": self.documents,
            "independent_sources": self.independent_sources,
            "claims": self.claims,
            "claims_with_evidence": self.claims_with_evidence,
            "claims_with_independent_support": self.claims_with_independent_support,
            "claims_with_a_primary_record": self.claims_with_a_primary_record,
            "claims_with_counterevidence": self.claims_with_counterevidence,
            "excerpts": self.excerpts,
            "unverifiable_excerpts": self.unverifiable_excerpts,
            "open_questions": self.open_questions,
            "model_calls": self.model_calls,
            "tokens": self.tokens,
            "cost": round(self.cost, 4),
        }


async def evaluate_run(
    *,
    model: Any,
    question: str | None = None,
    max_tasks: int | None = 8,
    concurrency: int = 4,
    config: ResearchConfig | None = None,
) -> dict[str, Any]:
    """Run an investigation end to end and measure what it left behind.

    Offline by default: the corpus is the bundled one and the model is
    whatever the caller passes, so the result is comparable between runs
    rather than dependent on what the web said today.
    """
    settings = config or ResearchConfig(budget=BudgetPolicy(max_runtime_minutes=5))
    corpus = load_corpus()
    registry, _client = build_offline_registry(corpus, settings)

    with ResearchStore.in_memory() as store:
        investigation = store.investigations.create(
            question or corpus.get("question", "evaluation"),
            budget=settings.budget.to_dict(),
        )
        ledger = BudgetLedger(store.budget, investigation.id, settings.budget)
        scheduler = Scheduler(
            store,
            registry,
            model,
            investigation_id=investigation.id,
            ledger=ledger,
            max_concurrent_tasks=concurrency,
        )

        started = time.monotonic()
        run = await scheduler.run(max_tasks=max_tasks)
        report = RunReport(
            question=investigation.question,
            seconds=time.monotonic() - started,
            tasks=run.tasks_run,
            stop_reason=str(run.stop_reason),
            stop_detail=run.stop_detail,
        )
        _measure(store, investigation.id, report, ledger)
        return report.to_dict()


def _measure(
    store: ResearchStore, investigation_id: str, report: RunReport, ledger: BudgetLedger
) -> None:
    documents = store.documents.list(investigation_id, limit=1000)
    report.documents = len(documents)
    report.independent_sources = len(
        {document.independence_key or document.id for document in documents}
    )

    graph = ClaimGraph(store, investigation_id)
    claims = store.claims.list(investigation_id, hydrate=False)
    report.claims = len(claims)
    held = {document.id: document for document in documents}

    for claim in claims:
        assessment = graph.assess(claim.id)
        links = store.claims.evidence_links(claim.id)
        if links:
            report.claims_with_evidence += 1
        if assessment.support.independent_count >= 2:
            report.claims_with_independent_support += 1
        if assessment.contradiction.count:
            report.claims_with_counterevidence += 1
        if any(
            PRIMARY_SOURCE_DISTANCE.get(held[link.document_id].source_type, 9) <= 1
            for link in links
            if link.document_id in held
        ):
            report.claims_with_a_primary_record += 1

        for link in links:
            if not link.excerpt:
                continue
            report.excerpts += 1
            document = held.get(link.document_id)
            if document is not None and not excerpt_appears_in(document, link.excerpt):
                report.unverifiable_excerpts.append(f"{claim.id}/{link.document_id}")

    # A gap is a specific thing missing from a claim's evidence, so the
    # count is over claims, not over the investigation.
    report.open_questions = sum(
        1 for assessment in graph.assess_all() if assessment.gaps
    )
    report.model_calls = int(ledger.used(Resource.MODEL_CALLS))
    report.tokens = ledger.used(Resource.TOKENS)
    report.cost = ledger.used(Resource.COST)
