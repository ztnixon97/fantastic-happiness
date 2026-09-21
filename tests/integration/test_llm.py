"""Model adapters against recorded API shapes."""

from __future__ import annotations

import httpx
import pytest

from research.config import ModelSettings, ResearchConfig, load_config
from research.errors import ConfigError, ModelError, ModelOutputError
from research.llm.base import Message, ModelSpec, extract_json
from research.llm.factory import build_model
from research.llm.providers import AnthropicModel, OpenAICompatibleModel
from research.llm.transport import ModelTransport

ANTHROPIC_REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": '{"action": "search_academic"}'}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 120, "output_tokens": 18},
}

OPENAI_REPLY = {
    "id": "chatcmpl-1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"action": "search_news"}'},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 90, "completion_tokens": 12},
}

MESSAGES = [
    Message(role="system", content="you are a researcher"),
    Message(role="user", content="take the next step"),
]


def transport_for(reply, *, record: list | None = None, status: int = 200) -> ModelTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        if status >= 400:
            return httpx.Response(status, json={"error": "nope"})
        return httpx.Response(200, json=reply)

    return ModelTransport(transport=httpx.MockTransport(handler), sleep=_no_sleep)


async def _no_sleep(_seconds: float) -> None:
    return None


class TestAnthropicAdapter:
    async def test_reply_and_usage_are_normalised(self) -> None:
        model = AnthropicModel(
            ModelSpec(provider="anthropic", model="claude-sonnet-5"),
            api_key="secret",
            transport=transport_for(ANTHROPIC_REPLY),
        )
        response = await model.complete(MESSAGES)
        assert response.text == '{"action": "search_academic"}'
        assert response.usage.input_tokens == 120
        assert response.usage.total_tokens == 138

    async def test_the_system_prompt_is_sent_separately(self) -> None:
        requests: list[httpx.Request] = []
        model = AnthropicModel(
            ModelSpec(provider="anthropic", model="claude-sonnet-5"),
            api_key="secret",
            transport=transport_for(ANTHROPIC_REPLY, record=requests),
        )
        await model.complete(MESSAGES)
        import json

        body = json.loads(requests[0].content)
        assert body["system"] == "you are a researcher"
        assert [message["role"] for message in body["messages"]] == ["user"]
        assert requests[0].headers["x-api-key"] == "secret"
        assert requests[0].headers["anthropic-version"]

    async def test_api_errors_surface_as_model_errors(self) -> None:
        model = AnthropicModel(
            ModelSpec(provider="anthropic", model="claude-sonnet-5"),
            api_key="secret",
            transport=transport_for(ANTHROPIC_REPLY, status=401),
        )
        with pytest.raises(ModelError) as exc:
            await model.complete(MESSAGES)
        assert not exc.value.retryable

    async def test_transient_failures_are_retried(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) == 1:
                return httpx.Response(529, json={"error": "overloaded"})
            return httpx.Response(200, json=ANTHROPIC_REPLY)

        model = AnthropicModel(
            ModelSpec(provider="anthropic", model="claude-sonnet-5"),
            api_key="secret",
            transport=ModelTransport(
                transport=httpx.MockTransport(handler), sleep=_no_sleep
            ),
        )
        response = await model.complete(MESSAGES)
        assert len(attempts) == 2
        assert response.text


class TestOpenAICompatibleAdapter:
    async def test_reply_and_usage_are_normalised(self) -> None:
        model = OpenAICompatibleModel(
            ModelSpec(provider="openai", model="gpt-x"),
            api_key="secret",
            transport=transport_for(OPENAI_REPLY),
        )
        response = await model.complete(MESSAGES)
        assert response.text == '{"action": "search_news"}'
        assert response.usage.total_tokens == 102

    async def test_a_local_endpoint_needs_no_credential(self) -> None:
        requests: list[httpx.Request] = []
        model = OpenAICompatibleModel(
            ModelSpec(provider="openai", model="local-model", base_url="http://127.0.0.1:8000/v1"),
            transport=transport_for(OPENAI_REPLY, record=requests),
        )
        response = await model.complete(MESSAGES)
        assert response.text
        assert "authorization" not in requests[0].headers
        assert str(requests[0].url).startswith("http://127.0.0.1:8000/v1/chat/completions")


class TestFactory:
    def test_configuration_selects_the_adapter(self) -> None:
        config = load_config(environ={"RESEARCH_ANTHROPIC_API_KEY": "k"})
        assert isinstance(build_model(config), AnthropicModel)

        openai_config = ResearchConfig(model=ModelSettings(provider="openai", model="gpt-x"))
        assert isinstance(build_model(openai_config), OpenAICompatibleModel)

    def test_a_missing_credential_is_refused_not_substituted(self) -> None:
        config = load_config(environ={})
        with pytest.raises(ConfigError) as exc:
            build_model(config)
        assert "RESEARCH_ANTHROPIC_API_KEY" in str(exc.value)

    def test_unknown_providers_are_refused(self) -> None:
        config = ResearchConfig(model=ModelSettings(provider="mystery", model="x"))
        with pytest.raises(ConfigError):
            build_model(config)


class TestStructuredOutput:
    @pytest.mark.parametrize(
        "text",
        [
            '{"action": "search_web"}',
            '```json\n{"action": "search_web"}\n```',
            'Sure, here it is:\n```\n{"action": "search_web"}\n```',
            'I will search. {"action": "search_web"} Let me know.',
        ],
    )
    def test_json_survives_however_the_model_wraps_it(self, text: str) -> None:
        assert extract_json(text)["action"] == "search_web"

    @pytest.mark.parametrize("text", ["", "   ", "no json here at all"])
    def test_unparseable_replies_raise(self, text: str) -> None:
        with pytest.raises(ModelOutputError):
            extract_json(text)
