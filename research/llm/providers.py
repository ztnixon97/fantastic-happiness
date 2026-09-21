"""Model adapters.

Two are enough to keep the abstraction honest: Anthropic's Messages API, and
any OpenAI-compatible ``/chat/completions`` endpoint - which covers OpenAI
itself, several hosted providers, and local servers such as vLLM, llama.cpp
and Ollama.

Each adapter is handed its own credential explicitly. None of them reads the
environment.
"""

from __future__ import annotations

from typing import Any, Sequence

from research.errors import ModelError
from research.llm.base import Message, ModelResponse, ModelSpec, ModelUsage
from research.llm.transport import ModelTransport

ANTHROPIC_VERSION = "2023-06-01"


def _split_system(messages: Sequence[Message]) -> tuple[str | None, list[Message]]:
    """Anthropic takes the system prompt beside the conversation, not in it."""
    system_parts = [message.content for message in messages if message.role == "system"]
    rest = [message for message in messages if message.role != "system"]
    return ("\n\n".join(system_parts) or None), rest


class AnthropicModel:
    """Anthropic Messages API."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        api_key: str,
        transport: ModelTransport | None = None,
    ) -> None:
        self.spec = spec
        self._api_key = api_key
        self.transport = transport or ModelTransport()
        self.base_url = (spec.base_url or "https://api.anthropic.com").rstrip("/")

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        system, conversation = _split_system(messages)
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "max_tokens": max_tokens or self.spec.max_tokens,
            "temperature": self.spec.temperature if temperature is None else temperature,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in conversation
            ],
        }
        if system:
            payload["system"] = system

        data = await self.transport.post_json(
            f"{self.base_url}/v1/messages",
            provider=self.spec.provider,
            payload=payload,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        blocks = data.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = data.get("usage") or {}
        return ModelResponse(
            text=text,
            usage=ModelUsage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            ),
            stop_reason=data.get("stop_reason"),
            raw=data,
        )


class OpenAICompatibleModel:
    """Any ``/chat/completions`` endpoint, hosted or local."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        api_key: str | None = None,
        transport: ModelTransport | None = None,
    ) -> None:
        self.spec = spec
        self._api_key = api_key
        self.transport = transport or ModelTransport()
        self.base_url = (spec.base_url or "https://api.openai.com/v1").rstrip("/")

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "max_tokens": max_tokens or self.spec.max_tokens,
            "temperature": self.spec.temperature if temperature is None else temperature,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
        }
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        data = await self.transport.post_json(
            f"{self.base_url}/chat/completions",
            provider=self.spec.provider,
            payload=payload,
            headers=headers,
        )
        choices = data.get("choices") or []
        if not choices:
            raise ModelError("no choices in the response", provider=self.spec.provider)
        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        return ModelResponse(
            text=message.get("content") or "",
            usage=ModelUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            ),
            stop_reason=choices[0].get("finish_reason"),
            raw=data,
        )
