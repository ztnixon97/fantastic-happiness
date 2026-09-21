"""Asking before researching.

A research question as typed is usually underspecified in ways that change
what gets planned. "Are small modular reactors economic?" hides at least
three questions: economic against what alternative, on whose cost of
capital, and over what horizon. Planning without settling those produces an
investigation that answers a question nobody asked.

So a run may ask first. The rules it works under:

* **Few questions, or none.** Only what would change the plan. A question
  already specific enough gets no questions at all, and saying so is a
  valid answer - an interrogation before every run would be worse than none.
* **Every question says what it would change.** A person deciding whether
  to bother answering needs to know what turns on it.
* **Answers are context, never findings.** They go into the brief the
  planner reads and into the investigation's record. They are not evidence,
  they are not claims, and nothing cites them.
* **Skipping is always allowed.** Unanswered questions are recorded as
  unanswered, which is itself worth knowing when reading the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.budgets import BudgetLedger, Resource
from research.errors import ModelOutputError
from research.llm.base import Message, ModelClient, extract_json
from research.normalize.text import normalize_whitespace

CLARIFIER_SYSTEM = """\
You are about to plan a research investigation. Before you do, decide
whether the question is specific enough to plan against.

Ask only about things that would change the plan: which alternative a
comparison is against, which jurisdiction or market, over what period,
whose perspective, or which of several meanings of an ambiguous term is
intended. Do not ask for information the research itself is supposed to
find, and do not ask the person to do your reasoning for you.

If the question is already clear enough to plan against, ask nothing. That
is the common case for a well-posed question and it is the right answer.
"""

CLARIFIER_FORMAT = """\
Reply with a single JSON object and nothing else:

{"questions": [{"question": "one sentence", "why": "what this changes about
 the plan", "suggestion": "a reasonable default if they do not answer"}]}

At most {limit} questions. An empty list means the question is clear enough.
"""

MAX_QUESTIONS = 3


@dataclass(slots=True)
class ClarifyingQuestion:
    question: str
    why: str = ""
    #: What the run would assume if this goes unanswered.
    suggestion: str = ""
    answer: str | None = None

    @property
    def answered(self) -> bool:
        return bool((self.answer or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "why": self.why,
            "suggestion": self.suggestion,
            "answer": self.answer,
        }


@dataclass(slots=True)
class Clarification:
    questions: list[ClarifyingQuestion] = field(default_factory=list)
    #: Set when the step could not run at all, as opposed to having nothing
    #: to ask. The two are different and a report should not confuse them.
    skipped: str | None = None

    @property
    def answered(self) -> list[ClarifyingQuestion]:
        return [question for question in self.questions if question.answered]

    def brief(self, existing: str | None = None) -> str | None:
        """The planner's brief, with whatever was learned folded in."""
        answered = self.answered
        if not answered:
            return existing
        lines = [existing] if existing else []
        lines.append("The person was asked, before planning, and said:")
        for question in answered:
            lines.append(f"- {question.question}")
            lines.append(f"  {normalize_whitespace(question.answer or '')}")
        unanswered = [q for q in self.questions if not q.answered]
        if unanswered:
            lines.append(
                "They did not answer: "
                + "; ".join(question.question for question in unanswered)
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions": [question.to_dict() for question in self.questions],
            **({"skipped": self.skipped} if self.skipped else {}),
        }


class Clarifier:
    """Asks what it needs to know, and nothing else."""

    def __init__(
        self,
        model: ModelClient,
        *,
        ledger: BudgetLedger | None = None,
        price: tuple[float, float] | None = None,
    ) -> None:
        self.model = model
        self.ledger = ledger
        self.price = price

    async def ask(
        self, question: str, *, brief: str | None = None, limit: int = MAX_QUESTIONS
    ) -> Clarification:
        """Propose clarifying questions, or none.

        A model that fails here does not fail the run: an investigation that
        could not ask is an investigation that proceeds on the question as
        typed, which is what it would have done anyway.
        """
        if self.ledger is not None and not self.ledger.can_spend(Resource.MODEL_CALLS):
            return Clarification(skipped="no model calls left in the budget")

        messages = [
            Message(role="system", content=CLARIFIER_SYSTEM),
            Message(
                role="user",
                content="\n".join(
                    [
                        f"Research question: {question}",
                        *([f"Context already given: {brief}"] if brief else []),
                        "",
                        CLARIFIER_FORMAT.replace("{limit}", str(limit)),
                    ]
                ),
            ),
        ]
        try:
            response = await self.model.complete(messages)
        except Exception as exc:  # any adapter failure, not just ModelError
            return Clarification(skipped=f"the model could not be asked: {exc}")

        if self.ledger is not None:
            self.ledger.charge_model(response.usage, price=self.price)

        try:
            payload = extract_json(response.text)
        except ModelOutputError as exc:
            return Clarification(skipped=f"the model's reply was not usable: {exc}")

        return Clarification(questions=_read_questions(payload, limit=limit))


def _read_questions(payload: Any, *, limit: int) -> list[ClarifyingQuestion]:
    raw = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        return []
    questions: list[ClarifyingQuestion] = []
    for entry in raw[: max(0, limit)]:
        if isinstance(entry, str):
            text = normalize_whitespace(entry)
            why = ""
            suggestion = ""
        elif isinstance(entry, dict):
            text = normalize_whitespace(str(entry.get("question") or ""))
            why = normalize_whitespace(str(entry.get("why") or ""))
            suggestion = normalize_whitespace(str(entry.get("suggestion") or ""))
        else:
            continue
        if len(text) < 5:
            continue
        questions.append(
            ClarifyingQuestion(question=text[:300], why=why[:300], suggestion=suggestion[:300])
        )
    return questions
