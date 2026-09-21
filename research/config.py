"""Configuration for an investigation run.

Two rules shape this module:

* Research budgets are explicit. An investigation stops because it decided
  to, not because a context window ran out.
* Secrets are resolved from a narrow allowlist of environment variables and
  handed only to the provider that needs them. Research components do not
  inherit the host environment.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from research.errors import ConfigError

#: Only these environment variables are ever read for provider credentials.
#: A provider adapter receives its own key and nothing else.
PROVIDER_KEY_ENV = {
    "anthropic": "RESEARCH_ANTHROPIC_API_KEY",
    "openai": "RESEARCH_OPENAI_API_KEY",
    "brave": "RESEARCH_BRAVE_API_KEY",
    "tavily": "RESEARCH_TAVILY_API_KEY",
    "semantic_scholar": "RESEARCH_SEMANTIC_SCHOLAR_API_KEY",
    "newsapi": "RESEARCH_NEWSAPI_API_KEY",
    "serper": "RESEARCH_SERPER_API_KEY",
    "youtube": "RESEARCH_YOUTUBE_API_KEY",
    "mastodon": "RESEARCH_MASTODON_TOKEN",
}

#: Contact address sent in User-Agent to academic APIs; they ask for one and
#: grant higher rate limits in return.
CONTACT_ENV = "RESEARCH_CONTACT_EMAIL"


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Hard limits for a single investigation."""

    max_depth: int = 4
    max_tasks: int = 30
    max_searches: int = 60
    max_documents: int = 250
    max_academic_documents: int = 100
    max_news_documents: int = 100
    max_citation_depth: int = 2
    max_citation_documents: int = 60
    max_runtime_minutes: int = 30
    max_provider_calls: int = 400
    max_model_calls: int = 200
    max_tokens: int = 2_000_000
    max_failed_source_calls: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetPolicy":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass(frozen=True, slots=True)
class AcquisitionPolicy:
    """Network-facing safety limits for retrieval.

    These are enforced in code, not asked for in a prompt: a prompt cannot
    stop a fetch, and retrieved content is treated as hostile.
    """

    request_timeout_seconds: float = 20.0
    connect_timeout_seconds: float = 10.0
    max_response_bytes: int = 5_000_000
    max_text_characters: int = 400_000
    max_redirects: int = 5
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    per_host_min_interval_seconds: float = 0.2
    max_concurrent_requests: int = 6
    user_agent: str = "research-environment/0.1 (+https://example.invalid/research)"
    #: Only these content types are parsed. Anything else is recorded as a
    #: failed fetch rather than guessed at.
    allowed_content_types: tuple[str, ...] = (
        "text/html",
        "application/xhtml+xml",
        "text/plain",
        # Papers, filings and regulator notices are published as PDFs. Text
        # is read out of them; nothing in the file is executed.
        "application/pdf",
        "application/json",
        "application/xml",
        "text/xml",
        "application/atom+xml",
        "application/rss+xml",
    )
    #: A ``Retry-After`` longer than this means "not now, and not soon": the
    #: request fails immediately instead of sleeping. Providers really do ask
    #: for hours (OpenAlex answers a rate-limited call with ~21 hours), and
    #: waiting even 30 seconds for one provider while others are answering is
    #: never the right trade inside a research run.
    max_retry_after_seconds: float = 5.0
    #: Consecutive failures from one provider before an investigation stops
    #: calling it. A rate-limited or unreachable provider otherwise costs the
    #: full retry budget on every single search.
    provider_failure_threshold: int = 3
    #: Loopback/private/link-local destinations are refused. Only a test or a
    #: deliberate local-mirror deployment should turn this on.
    allow_private_hosts: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AcquisitionPolicy":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        if "allowed_content_types" in known:
            known["allowed_content_types"] = tuple(known["allowed_content_types"])
        return cls(**known)


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    """Per-provider switch and endpoint override."""

    enabled: bool = True
    base_url: str | None = None
    #: Requests per second ceiling applied on top of the global policy.
    rate_limit_per_second: float | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Which model conducts the reasoning, and how far it may go.

    ``provider`` selects an adapter, not a vendor lock: ``openai`` here means
    an OpenAI-compatible endpoint, which a local server also provides.
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-5"
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    #: Steps a single research worker may take before it must return.
    max_steps_per_task: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSettings":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Searching evidence the investigation already holds.

    Lexical and graph retrieval are always on: they need no model, no
    credential and no extra store. Vectors are opt-in because they are only
    worth their cost once lexical retrieval is demonstrably failing.
    """

    embeddings_enabled: bool = False
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_base_url: str = "https://api.openai.com/v1"
    #: Which credential the embedder uses, from the same allowlist.
    embedding_provider: str = "openai"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalSettings":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass(slots=True)
class ResearchConfig:
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    acquisition: AcquisitionPolicy = field(default_factory=AcquisitionPolicy)
    model: ModelSettings = field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    providers: dict[str, ProviderSettings] = field(default_factory=dict)
    database_path: str | None = None
    contact_email: str | None = None
    #: Resolved provider credentials. Populated only from PROVIDER_KEY_ENV.
    _secrets: dict[str, str] = field(default_factory=dict, repr=False)

    def provider(self, name: str) -> ProviderSettings:
        return self.providers.get(name, ProviderSettings())

    def is_enabled(self, name: str) -> bool:
        return self.provider(name).enabled

    def secret(self, provider: str) -> str | None:
        """Credential for one provider, or ``None`` if not configured."""
        return self._secrets.get(provider)

    def with_budget(self, **overrides: Any) -> "ResearchConfig":
        budget = replace(self.budget, **overrides)
        return replace(self, budget=budget)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view. Never includes secrets."""
        return {
            "budget": self.budget.to_dict(),
            "acquisition": self.acquisition.to_dict(),
            "model": self.model.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "providers": {
                name: {
                    "enabled": settings.enabled,
                    "base_url": settings.base_url,
                    "rate_limit_per_second": settings.rate_limit_per_second,
                    "options": settings.options,
                }
                for name, settings in self.providers.items()
            },
            "database_path": self.database_path,
            "contact_email": self.contact_email,
        }


def _read_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency present
            raise ConfigError("PyYAML is required to read YAML config") from exc
        data = yaml.safe_load(text) or {}
    elif path.suffix == ".toml":
        import tomllib

        data = tomllib.loads(text)
    elif path.suffix == ".json":
        import json

        data = json.loads(text)
    else:
        raise ConfigError(f"unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def load_config(
    path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> ResearchConfig:
    """Load configuration from a file plus the credential allowlist.

    ``environ`` is injectable so tests never depend on the host environment.
    """
    data: dict[str, Any] = {}
    if path:
        config_path = Path(path).expanduser()
        if not config_path.exists():
            raise ConfigError(f"config file not found: {config_path}")
        data = _read_config_file(config_path)

    research_block = data.get("research", data)
    budget = BudgetPolicy.from_dict(research_block.get("budget", research_block))
    acquisition = AcquisitionPolicy.from_dict(data.get("acquisition", {}))
    model = ModelSettings.from_dict(data.get("model", {}))
    retrieval = RetrievalSettings.from_dict(data.get("retrieval", {}))

    providers: dict[str, ProviderSettings] = {}
    for name, raw in (data.get("providers") or {}).items():
        if not isinstance(raw, dict):
            raise ConfigError(f"provider {name!r} config must be a mapping")
        providers[name] = ProviderSettings(
            enabled=bool(raw.get("enabled", True)),
            base_url=raw.get("base_url"),
            rate_limit_per_second=raw.get("rate_limit_per_second"),
            options=dict(raw.get("options", {})),
        )

    env = os.environ if environ is None else environ
    secrets = {
        provider: env[var]
        for provider, var in PROVIDER_KEY_ENV.items()
        if env.get(var)
    }
    contact = data.get("contact_email") or env.get(CONTACT_ENV)

    acquisition_overrides: dict[str, Any] = {}
    if contact and "(+https://example.invalid" in acquisition.user_agent:
        acquisition_overrides["user_agent"] = (
            f"research-environment/0.1 (mailto:{contact})"
        )
    if acquisition_overrides:
        acquisition = replace(acquisition, **acquisition_overrides)

    return ResearchConfig(
        budget=budget,
        acquisition=acquisition,
        model=model,
        retrieval=retrieval,
        providers=providers,
        database_path=data.get("database_path"),
        contact_email=contact,
        _secrets=secrets,
    )
