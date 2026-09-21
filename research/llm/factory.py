"""Building the configured model client.

One place decides which adapter an investigation uses, from configuration and
the credential allowlist. Everything above this point takes a
:class:`~research.llm.base.ModelClient` and never asks who supplies it.
"""

from __future__ import annotations

from typing import Any

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
    role: str | None = None,
    transport: ModelTransport | None = None,
) -> ModelClient:
    """Return the model client this configuration calls for.

    ``role`` selects a tier: planning and synthesis decide the shape of a
    whole investigation and are called a handful of times, while the
    gathering roles are called constantly and mostly pick the next search.
    With no tiers configured every role resolves to the same model.

    Raises :class:`ConfigError` rather than falling back to another vendor: a
    silent substitution would change the reasoning behind an investigation
    without saying so.
    """
    settings = config.model
    resolved = settings.resolve(role)
    spec = ModelSpec(
        provider=resolved.provider,
        model=resolved.model,
        max_tokens=resolved.max_tokens,
        temperature=resolved.temperature,
        base_url=resolved.base_url,
    )
    api_key = config.secret(resolved.provider)

    if resolved.provider == "anthropic":
        if not api_key:
            raise ConfigError(
                "no Anthropic credential is configured; set RESEARCH_ANTHROPIC_API_KEY, "
                "choose another model provider, or run with --offline"
            )
        return AnthropicModel(spec, api_key=api_key, transport=transport)

    if resolved.provider in ("openai", "openai_compatible", "local"):
        if REQUIRES_KEY.get(resolved.provider, False) and not api_key:
            raise ConfigError(f"no credential configured for {resolved.provider}")
        return OpenAICompatibleModel(spec, api_key=api_key, transport=transport)

    raise ConfigError(
        f"unknown model provider {resolved.provider!r}; expected 'anthropic', "
        "'openai' (any OpenAI-compatible endpoint, including a local server)"
    )


class ModelPool:
    """One client per tier, built on demand and reused.

    The scheduler asks for a model by role rather than being handed one, so
    the planner can run on a stronger model than the eight gathering tasks
    it spawns. Clients are cached per tier: two roles on the same tier share
    one client, and a role never used costs nothing.

    ``fixed`` short-circuits the whole thing with a single client, which is
    what an offline run and a scripted test need.
    """

    def __init__(
        self,
        config: ResearchConfig | None = None,
        *,
        fixed: ModelClient | None = None,
        transport: ModelTransport | None = None,
    ) -> None:
        if config is None and fixed is None:
            raise ValueError("a ModelPool needs either a config or a fixed client")
        self.config = config
        self.fixed = fixed
        self.transport = transport
        self._clients: dict[str, ModelClient] = {}

    def for_role(self, role: Any = None) -> ModelClient:
        if self.fixed is not None:
            return self.fixed
        assert self.config is not None
        tier = self.config.model.tier_for(_name(role))
        client = self._clients.get(tier)
        if client is None:
            client = build_model(self.config, role=_name(role), transport=self.transport)
            self._clients[tier] = client
        return client

    def price_for(self, role: Any = None) -> tuple[float, float] | None:
        """(input, output) per million tokens, or None if this model is unpriced."""
        if self.config is None:
            return None
        return self.config.model.price(self.config.model.resolve(_name(role)).model)

    def describe(self, role: Any = None) -> str:
        if self.config is None:
            return "fixed"
        return self.config.model.resolve(_name(role)).describe()


def _name(role: Any) -> str | None:
    return str(role) if role is not None else None
