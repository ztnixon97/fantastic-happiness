"""A model stand-in that conducts a plausible investigation offline.

It is a decision rule, not a mind: it reads the role it was given and the
last observation, and picks the next action a worker of that role would
reasonably take. That is enough to exercise the whole loop - plan, search,
read, claim, link, delegate, stop - with no network and no credentials, which
is what makes the autonomous path testable and demonstrable.

It quotes real text. When it links evidence to a claim it takes the excerpt
from the document it just read, so the same verbatim check that applies to a
real model applies to it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from research.llm.base import Message, ModelSpec
from research.llm.scripted import RuleBasedModel

_OBSERVATION_RE = re.compile(r"^(?P<action>[a-z_]+) -> (?P<payload>\{.*)$", re.DOTALL)
_CONTENT_RE = re.compile(r"CONTENT:\n(.*?)\n</external_evidence>", re.DOTALL)
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "what",
        "which", "that", "this", "is", "are", "was", "were", "establish",
        "find", "identify", "determine", "whether", "how", "why", "their",
        "its", "with", "from", "about", "into", "look", "search",
    }
)

#: The role each task announces in its system prompt.
_ROLE_RE = re.compile(r"You are the (?P<role>[a-z_]+) researcher")


def _keywords(text: str, *, limit: int = 8) -> str:
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]+", text.lower())
        if word not in _STOPWORDS and len(word) > 2
    ]
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
        if len(seen) >= limit:
            break
    return " ".join(seen)


@dataclass(slots=True)
class _TaskState:
    phase: str = "search"
    documents: list[str] = field(default_factory=list)
    claim_id: str | None = None
    excerpt: str | None = None
    read_document: str | None = None
    spawned: bool = False


class OfflineResearcher:
    """Chooses the next action from the role and the last observation."""

    def __init__(self, *, spawn_followups: bool = True) -> None:
        self.states: dict[str, _TaskState] = {}
        self.spawn_followups = spawn_followups

    # -- entry point ----------------------------------------------------
    def __call__(self, messages: Sequence[Message]) -> dict[str, Any]:
        system = next((m.content for m in messages if m.role == "system"), "")
        first_user = next((m.content for m in messages if m.role == "user"), "")

        if "Plan the investigation" in first_user:
            return self._plan(first_user)

        role_match = _ROLE_RE.search(system)
        role = role_match.group("role") if role_match else "scout"
        objective = _extract(first_user, "Your task: ")
        state = self.states.setdefault(f"{role}:{objective}", _TaskState())
        observation = self._last_observation(messages)
        self._absorb(state, observation)

        last_user = messages[-1].content if messages else ""
        if "last step" in last_user:
            return self._complete(state, objective)
        return self._act(role, objective, state)

    # -- planning -------------------------------------------------------
    def _plan(self, prompt: str) -> dict[str, Any]:
        question = _extract(prompt, "Research question: ")
        topic = _keywords(question, limit=6)
        return {
            "brief": (
                f"Establish what the literature, the recent record and primary "
                f"sources say about: {question} Then attack the result."
            ),
            "entities": [],
            "claims_to_verify": [question.rstrip("?")],
            "tasks": [
                {
                    "role": "academic",
                    "operation": "search_academic",
                    "objective": f"establish what peer-reviewed work says about {topic}",
                    "priority": 1,
                    "rationale": "the literature sets the baseline",
                },
                {
                    "role": "news",
                    "operation": "search_news",
                    "objective": f"establish what has recently been reported about {topic}",
                    "priority": 2,
                    "rationale": "current activity is not yet in the literature",
                },
                {
                    "role": "primary_source",
                    "operation": "find_primary_source",
                    "objective": f"reach the underlying records behind reports about {topic}",
                    "priority": 3,
                    "rationale": "reporting should be traced to filings and regulators",
                },
                {
                    "role": "skeptic",
                    "operation": "find_counterevidence",
                    "objective": f"find evidence that contradicts the emerging picture on {topic}",
                    "priority": 4,
                    "rationale": "a conclusion nobody attacked is not a finding",
                },
            ],
        }

    # -- worker steps ---------------------------------------------------
    def _act(self, role: str, objective: str, state: _TaskState) -> dict[str, Any]:
        query = _keywords(objective)

        if state.phase == "search":
            state.phase = "read"
            return self._search_action(role, query, objective)

        if state.phase == "read":
            if not state.documents:
                return self._complete(state, objective)
            state.phase = "claim"
            state.read_document = state.documents[0]
            return {
                "thought": "read the strongest result before asserting anything",
                "action": "get_evidence",
                "arguments": {"document_id": state.read_document, "characters": 2000},
            }

        if state.phase == "claim":
            state.phase = "link"
            return {
                "thought": "state the proposition this task is about",
                "action": "create_claim",
                "arguments": {"text": _proposition(objective)},
            }

        if state.phase == "link" and state.claim_id and state.read_document:
            state.phase = "extra"
            stance = "contradicts" if role == "skeptic" else "supports"
            arguments: dict[str, Any] = {
                "claim_id": state.claim_id,
                "document_id": state.read_document,
                "stance": stance,
                "analysis": f"{role} worker: this document bears on the task objective",
            }
            if state.excerpt:
                arguments["excerpt"] = state.excerpt
            return {
                "thought": "attach the document, quoting it exactly",
                "action": "link_evidence",
                "arguments": arguments,
            }

        if state.phase in ("link", "extra"):
            extra = self._role_extra(role, state)
            if extra is not None:
                return extra
        return self._complete(state, objective)

    def _search_action(self, role: str, query: str, objective: str) -> dict[str, Any]:
        if role == "academic":
            action, arguments = "search_academic", {"query": query, "limit": 8}
        elif role == "news":
            action, arguments = "search_news", {"query": query, "limit": 8}
        elif role == "skeptic":
            action, arguments = "find_counterevidence", {"proposition": _proposition(objective)}
        else:
            action, arguments = "search_web", {"query": query, "limit": 8}
        arguments["objective"] = objective
        return {"thought": f"{role}: gather material for this objective", "action": action,
                "arguments": arguments}

    def _role_extra(self, role: str, state: _TaskState) -> dict[str, Any] | None:
        """One role-specific follow-through before reporting back."""
        state.phase = "done"
        if role == "academic" and state.documents:
            return {
                "thought": "follow the citation graph from the strongest paper",
                "action": "follow_citations",
                "arguments": {
                    "document_id": state.documents[0],
                    "direction": "backward",
                    "depth": 1,
                    "limit": 6,
                },
            }
        if role == "primary_source" and state.documents:
            return {
                "thought": "trace the assertion to the record behind it",
                "action": "find_primary_source",
                "arguments": {"document_id": state.documents[0]},
            }
        if role == "news" and state.documents and self.spawn_followups and not state.spawned:
            state.spawned = True
            return {
                "thought": "the filings behind this reporting deserve their own task",
                "action": "spawn_research_task",
                "arguments": {
                    "role": "primary_source",
                    "operation": "find_primary_source",
                    "objective": "trace the reported agreements to filings or regulatory records",
                    "priority": 3,
                },
            }
        return None

    def _complete(self, state: _TaskState, objective: str) -> dict[str, Any]:
        state.phase = "finished"
        followups: list[dict[str, Any]] = []
        if self.spawn_followups and state.claim_id and not state.spawned:
            state.spawned = True
            followups.append(
                {
                    "operation": "find_counterevidence",
                    "objective": f"test the claim behind: {objective}",
                    "rationale": "the claim has support but has not been attacked",
                }
            )
        return {
            "thought": "report what this task established",
            "action": "complete_research_task",
            "arguments": {
                "summary": (
                    f"Worked the objective: {objective}. "
                    f"{len(state.documents)} document(s) examined"
                    + (f"; claim {state.claim_id} recorded." if state.claim_id else ".")
                ),
                "claim_ids": [state.claim_id] if state.claim_id else [],
                "evidence_ids": state.documents[:6],
                "open_questions": [],
                "followups": followups,
            },
        }

    # -- observation handling -------------------------------------------
    def _last_observation(self, messages: Sequence[Message]) -> tuple[str, dict[str, Any]] | None:
        for message in reversed(messages):
            if message.role != "user":
                continue
            match = _OBSERVATION_RE.match(message.content.strip().split("\n")[0])
            if not match:
                continue
            try:
                payload = json.loads(match.group("payload"))
            except json.JSONDecodeError:
                return match.group("action"), {}
            return match.group("action"), payload
        return None

    def _absorb(self, state: _TaskState, observation: tuple[str, dict[str, Any]] | None) -> None:
        if observation is None:
            return
        action, payload = observation
        if action in ("search_academic", "search_news", "search_web", "find_counterevidence"):
            for entry in payload.get("results") or payload.get("documents") or []:
                document_id = entry.get("id")
                # Copies add nothing: read the original instead.
                if document_id and not entry.get("not_independent_of"):
                    state.documents.append(document_id)
            if not state.documents:
                state.phase = "finished"
        elif action == "get_evidence":
            state.excerpt = _first_sentence(payload.get("content", ""))
        elif action == "create_claim":
            state.claim_id = payload.get("claim_id")
        elif action in ("follow_citations", "find_primary_source"):
            for entry in payload.get("new_documents") or payload.get("documents") or []:
                if entry.get("id"):
                    state.documents.append(entry["id"])


def _extract(text: str, marker: str) -> str:
    for line in text.split("\n"):
        if line.startswith(marker):
            return line[len(marker):].strip()
    return ""


def _proposition(objective: str) -> str:
    cleaned = objective.strip().rstrip(".")
    for prefix in ("establish what ", "establish ", "find ", "reach ", "identify "):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return cleaned[0].upper() + cleaned[1:] if cleaned else objective


def _first_sentence(content: str, *, minimum: int = 40, maximum: int = 220) -> str | None:
    """Take a quotable sentence out of a rendered evidence block."""
    match = _CONTENT_RE.search(content)
    body = (match.group(1) if match else content).strip()
    for candidate in re.split(r"(?<=[.!?])\s+", body):
        text = candidate.strip()
        if minimum <= len(text) <= maximum:
            return text
    return body[:maximum] or None


def offline_research_model(*, spawn_followups: bool = True) -> RuleBasedModel:
    """A model client that runs an investigation over a fixture corpus."""
    return RuleBasedModel(
        OfflineResearcher(spawn_followups=spawn_followups),
        spec=ModelSpec(provider="offline", model="rule-based-researcher"),
    )
