"""Building the configured model client.

One place decides which adapter an investigation uses, from configuration and
the credential allowlist. Everything above this point takes a
:class:`~research.llm.base.ModelClient` and never asks who supplies it.
"""

from __future__ import annotations

from research.config import ResearchConfig
from research.errors import ConfigError
from research.llm.base import ModelClient, ModelSpec
from research.llm.providers import AnthropicModel, OpenAICompatibleModel
from research.llm.transport import ModelTransport

#: Providers that will not answer without a credential. An OpenAI-compatible
#: endpoint may be a local server, which usually will.
REQUIRES_KEY = {"anthropic": True, "openai": False}


def build_model(
    config: ResearchConfig,
    *,
    transport: ModelTransport | None = None,
) -> ModelClient:
    """Return the model client this configuration calls for.

    Raises :class:`ConfigError` rather than falling back to another vendor: a
    silent substitution would change the reasoning behind an investigation
    without saying so.
    """
    settings = config.model
    spec = ModelSpec(
        provider=settings.provider,
        model=settings.model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        base_url=settings.base_url,
    )
    api_key = config.secret(settings.provider)

    if settings.provider == "anthropic":
        if not api_key:
            raise ConfigError(
                "no Anthropic credential is configured; set RESEARCH_ANTHROPIC_API_KEY, "
                "choose another model provider, or run with --offline"
            )
        return AnthropicModel(spec, api_key=api_key, transport=transport)

    if settings.provider in ("openai", "openai_compatible", "local"):
        if REQUIRES_KEY.get(settings.provider, False) and not api_key:
            raise ConfigError(f"no credential configured for {settings.provider}")
        return OpenAICompatibleModel(spec, api_key=api_key, transport=transport)

    raise ConfigError(
        f"unknown model provider {settings.provider!r}; expected 'anthropic', "
        "'openai' (any OpenAI-compatible endpoint, including a local server)"
    )
