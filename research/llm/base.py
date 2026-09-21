"""Provider-agnostic access to a language model.

The research system uses a model for one thing: semantic judgement. Which
vendor supplies that judgement is a configuration detail, so the interface
here is the smallest one that supports the job - messages in, text out, token
usage reported - and everything provider-specific lives in an adapter.

Structured output is obtained by asking for JSON and parsing it, rather than
by any vendor's function-calling format. That keeps the same agent code
working against models that have no such format, including local ones.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from research.errors import ModelOutputError

Role = Literal["system", "user", "assistant"]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Which model to call and how."""

    provider: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.0
    base_url: str | None = None

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "ModelUsage") -> "ModelUsage":
        return ModelUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    usage: ModelUsage = field(default_factory=ModelUsage)
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelClient(Protocol):
    """Every adapter satisfies exactly this."""

    spec: ModelSpec

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        ...


def extract_json(text: str) -> Any:
    """Parse a JSON object out of a model's reply.

    Models wrap JSON in prose and code fences however they were trained to.
    That is a formatting quirk, not a failure, so it is handled here rather
    than by asking more insistently in the prompt.
    """
    candidate = text.strip()
    if not candidate:
        raise ModelOutputError("the model returned nothing")

    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost balanced object or array in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ModelOutputError(
        f"could not read JSON from the model's reply: {candidate[:200]!r}"
    )
