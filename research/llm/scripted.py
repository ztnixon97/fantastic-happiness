"""Deterministic models for tests and offline runs.

A research system whose behaviour can only be observed by spending money on
a hosted model is one nobody can test. These stand-ins implement the same
interface, so the planner, the workers and the scheduler run unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from research.errors import ModelError
from research.llm.base import Message, ModelResponse, ModelSpec, ModelUsage

Responder = Callable[[Sequence[Message]], str | dict[str, Any]]


def _render(reply: str | dict[str, Any]) -> str:
    if isinstance(reply, str):
        return reply
    return json.dumps(reply, ensure_ascii=False)


class ScriptedModel:
    """Replays a fixed list of replies, in order.

    Also records every prompt it was given, which is how tests assert that
    retrieved material reached the model as labelled external evidence rather
    than as instructions.
    """

    def __init__(
        self,
        replies: Sequence[str | dict[str, Any]],
        *,
        spec: ModelSpec | None = None,
        loop_last: bool = False,
    ) -> None:
        self.spec = spec or ModelSpec(provider="scripted", model="scripted")
        self._replies = list(replies)
        self._index = 0
        self.loop_last = loop_last
        self.calls: list[list[Message]] = []

    @property
    def remaining(self) -> int:
        return max(0, len(self._replies) - self._index)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if self._index >= len(self._replies):
            if not (self.loop_last and self._replies):
                raise ModelError(
                    f"scripted model exhausted after {self._index} calls",
                    provider="scripted",
                )
            reply = self._replies[-1]
        else:
            reply = self._replies[self._index]
            self._index += 1
        text = _render(reply)
        return ModelResponse(
            text=text,
            usage=ModelUsage(
                input_tokens=sum(len(message.content) // 4 for message in messages),
                output_tokens=len(text) // 4,
            ),
            stop_reason="end_turn",
        )

    def prompts(self) -> list[str]:
        """Every message body the model has been shown, flattened."""
        return [message.content for call in self.calls for message in call]


class RuleBasedModel:
    """Chooses a reply by inspecting the prompt.

    Used for the offline demonstration, where the 'model' needs to behave
    plausibly - plan, search, then stop - without any network. It is a stub
    with a decision rule, not an attempt at intelligence.
    """

    def __init__(self, responder: Responder, *, spec: ModelSpec | None = None) -> None:
        self.spec = spec or ModelSpec(provider="rule-based", model="offline")
        self._responder = responder
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        text = _render(self._responder(messages))
        return ModelResponse(
            text=text,
            usage=ModelUsage(
                input_tokens=sum(len(message.content) // 4 for message in messages),
                output_tokens=len(text) // 4,
            ),
            stop_reason="end_turn",
        )
