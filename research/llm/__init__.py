"""Provider-agnostic model access."""

from research.llm.base import (
    Message,
    ModelClient,
    ModelResponse,
    ModelSpec,
    ModelUsage,
    extract_json,
)
from research.llm.factory import build_model
from research.llm.providers import AnthropicModel, OpenAICompatibleModel
from research.llm.scripted import RuleBasedModel, ScriptedModel
from research.llm.transport import ModelTransport, TransportPolicy

__all__ = [
    "AnthropicModel",
    "Message",
    "ModelClient",
    "ModelResponse",
    "ModelSpec",
    "ModelTransport",
    "ModelUsage",
    "OpenAICompatibleModel",
    "RuleBasedModel",
    "ScriptedModel",
    "TransportPolicy",
    "build_model",
    "extract_json",
]
