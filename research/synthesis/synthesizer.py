"""Writing the executive summary.

The synthesizer is given the investigation's own record - claims with their
assessments, the independent-source counts, the open questions - and asked
for prose. It is not given the corpus and cannot search: by this point the
research is done, and a synthesis step that quietly goes looking for more
evidence produces a report nobody can trace.

Its output is checked before use. A summary that cites claim or evidence
identifiers that do not exist is rejected rather than published.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from research.errors import ModelError
from research.ids import CLAIM, EVIDENCE
from research.llm.base import Message, ModelClient
from research.normalize.text import normalize_whitespace, truncate
from research.storage.store import ResearchStore
from research.synthesis.report import ReportData, collect, render_markdown

SYNTHESIZER_SYSTEM = """\
You write the executive summary of a completed investigation.

You are given the investigation's own record: the claims, what the evidence
says about each, how many independent sources stand behind them, and what
remains unsettled. Write three to six sentences that a careful reader could
check against that record.

Rules:

- Answer the question that was asked, to the extent the evidence allows, and
  say plainly where it does not allow an answer.
- Cite claim identifiers (claim:N) for every material assertion.
- Count independent sources, not documents. If a finding rests on one source,
  say so.
- Do not introduce facts that are not in the record below. You have no way to
  check them and no way for a reader to trace them.
- Do not smooth over contradiction. If the evidence is mixed, the summary
  says it is mixed.

Write prose only - no headings, no lists, no preamble.
"""

_ID_RE = re.compile(rf"\b((?:{CLAIM}|{EVIDENCE}):\d+)\b")


@dataclass(slots=True)
class Synthesis:
    markdown: str
    data: ReportData
    summary: str | None = None
    #: Identifiers the summary cited that do not exist in this investigation.
    invalid_references: list[str] = field(default_factory=list)
    model_used: str | None = None

    @property
    def summary_was_written(self) -> bool:
        return bool(self.summary)


class Synthesizer:
    def __init__(
        self,
        store: ResearchStore,
        *,
        investigation_id: str,
        model: ModelClient | None = None,
    ) -> None:
        self.store = store
        self.investigation_id = investigation_id
        self.model = model

    async def run(self) -> Synthesis:
        data = collect(self.store, self.investigation_id)
        summary: str | None = None
        invalid: list[str] = []
        model_used: str | None = None

        if self.model is not None and data.assessments:
            try:
                summary, invalid = await self._write_summary(data)
                model_used = getattr(self.model.spec, "provider", None)
            except ModelError:
                # A report assembled from the record is still a report; only
                # the prose is lost.
                summary = None

        markdown = render_markdown(data, self.store, summary=summary)
        return Synthesis(
            markdown=markdown,
            data=data,
            summary=summary,
            invalid_references=invalid,
            model_used=model_used,
        )

    async def _write_summary(self, data: ReportData) -> tuple[str | None, list[str]]:
        response = await self.model.complete(
            [
                Message(role="system", content=SYNTHESIZER_SYSTEM),
                Message(role="user", content=self._record(data)),
            ]
        )
        summary = normalize_whitespace(response.text)
        if not summary:
            return None, []
        invalid = self._invalid_references(summary)
        if invalid:
            # The summary cites things that do not exist. Publishing it would
            # put an untraceable statement in a report whose whole point is
            # traceability.
            return None, invalid
        return truncate(summary, 3000), []

    def _record(self, data: ReportData) -> str:
        lines = [
            f"Question: {data.investigation.question}",
            f"Evidence: {data.counts['documents']} documents from "
            f"{data.counts['independent_sources']} independent sources.",
            f"The investigation stopped because: {data.investigation.stop_reason} "
            f"({data.investigation.stop_detail}).",
            "",
            "Claims and what the evidence says:",
        ]
        for assessment in data.findings():
            lines.append(
                f"- {assessment.claim_id} [{assessment.status}] {assessment.text}"
            )
            lines.append(f"    {assessment.explanation}")
            if assessment.gaps:
                lines.append(f"    gaps: {'; '.join(assessment.gaps[:2])}")
        if data.open_questions:
            lines.append("")
            lines.append("Still open:")
            for question in data.open_questions[:8]:
                lines.append(f"- {question['claim_id']}: {question['gaps'][0]}")
        lines.append("")
        lines.append("Write the executive summary.")
        return "\n".join(lines)

    def _invalid_references(self, summary: str) -> list[str]:
        cited = set(_ID_RE.findall(summary))
        if not cited:
            return []
        known = {
            claim.id for claim in self.store.claims.list(self.investigation_id, hydrate=False)
        } | {
            document.id
            for document in self.store.documents.list(self.investigation_id, limit=2000)
        }
        return sorted(cited - known)
