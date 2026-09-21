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
    #: A ceiling in money, counted only over models the configuration has
    #: priced. Unlimited by default, because an unpriced model would make
    #: any other default silently unenforceable.
    max_cost: float = float("inf")
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
class ModelTier:
    """An override of the default model, for roles that need a different one.

    Every field is optional and falls back to the defaults on
    :class:`ModelSettings`, so naming a cheaper model for the gathering roles
    does not mean restating the provider, the token limit and the endpoint.
    """

    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, data: Any) -> "ModelTier":
        # A tier written as a bare string is the common case: `fast: gpt-4.1-mini`.
        if isinstance(data, str):
            return cls(model=data)
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in (data or {})}
        return cls(**known)


#: Which tier each role uses when configuration does not say otherwise.
#: Planning and synthesis decide the shape of the whole investigation and are
#: called a handful of times; gathering is called constantly and mostly picks
#: the next search. Spending the same model on both is the expensive mistake.
DEFAULT_ROLE_TIERS: dict[str, str] = {
    "planner": "strategic",
    "synthesizer": "strategic",
    "skeptic": "default",
    "scout": "default",
    "academic": "fast",
    "news": "fast",
    "primary_source": "fast",
    "social": "fast",
}


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Which model conducts the reasoning, and how far it may go.

    ``provider`` selects an adapter, not a vendor lock: ``openai`` here means
    an OpenAI-compatible endpoint, which a local server also provides.

    The fields here are the default. ``fast`` and ``strategic`` override it
    for the roles mapped to them, and ``roles`` re-maps any role to any tier.
    Leave them alone and every role uses one model, as before.
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-5"
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    #: Steps a single research worker may take before it must return.
    max_steps_per_task: int = 8
    #: Research tasks running at once. Tasks are independent by
    #: construction, so this is throughput; the cost is that the stopping
    #: rules are evaluated between batches rather than between tasks, so a
    #: run can overshoot by up to this many tasks.
    max_concurrent_tasks: int = 4
    #: Cheaper model for the roles that mostly decide which search to run.
    fast: ModelTier = field(default_factory=ModelTier)
    #: Stronger model for the roles that decide the shape of the whole run.
    strategic: ModelTier = field(default_factory=ModelTier)
    #: role name -> tier name, overriding DEFAULT_ROLE_TIERS.
    roles: dict[str, str] = field(default_factory=dict)
    #: Model name -> (input, output) price per million tokens. There is no
    #: built-in table on purpose: published prices change, vary by tier and
    #: by region, and a stale number reported as this run's cost would be a
    #: fabricated figure in a system whose whole point is not fabricating
    #: figures. An unpriced model reports tokens and says cost is unknown.
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSettings":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        for name in ("fast", "strategic"):
            if name in known:
                known[name] = ModelTier.from_dict(known[name])
        if "roles" in known:
            known["roles"] = {str(k): str(v) for k, v in (known["roles"] or {}).items()}
        if "prices" in known:
            known["prices"] = {
                str(name): (float(pair[0]), float(pair[1]))
                for name, pair in (known["prices"] or {}).items()
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            }
        return cls(**known)

    def tier_for(self, role: str | None) -> str:
        if not role:
            return "default"
        return self.roles.get(str(role)) or DEFAULT_ROLE_TIERS.get(str(role), "default")

    def resolve(self, role: str | None = None) -> "ResolvedModel":
        """The provider, model and limits a given role should use."""
        tier_name = self.tier_for(role)
        tier = {"fast": self.fast, "strategic": self.strategic}.get(tier_name, ModelTier())
        return ResolvedModel(
            tier=tier_name,
            provider=tier.provider or self.provider,
            model=tier.model or self.model,
            base_url=tier.base_url if tier.base_url is not None else self.base_url,
            max_tokens=tier.max_tokens or self.max_tokens,
            temperature=(
                tier.temperature if tier.temperature is not None else self.temperature
            ),
        )

    def price(self, model: str) -> tuple[float, float] | None:
        entry = self.prices.get(model)
        if not entry:
            return None
        return (float(entry[0]), float(entry[1]))


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """One role's model, after tiers have been applied."""

    tier: str
    provider: str
    model: str
    base_url: str | None
    max_tokens: int
    temperature: float

    def describe(self) -> str:
        return f"{self.provider}:{self.model} ({self.tier})"



@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Searching evidence the investigation already holds.

    All three retrievers are on. Lexical and graph need nothing installed;
    vectors need a sentence encoder, and the default one runs locally, so
    they cost no credential and no outbound request either. A hosted
    embedder is a configuration change, not a different code path.

    A model that cannot be loaded costs the ranking its third opinion and
    nothing else: the search still runs on lexical and graph results, and
    says that it was degraded.
    """

    embeddings_enabled: bool = True
    #: "local" runs a sentence encoder in this process. Anything else names
    #: a credential in the provider allowlist and calls an /embeddings
    #: endpoint of the OpenAI shape.
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    #: Weights already on disk. Set it to embed without a network at all.
    embedding_model_path: str | None = None
    #: Only read when the provider is a hosted one.
    embedding_base_url: str = "https://api.openai.com/v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalSettings":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)



@dataclass(frozen=True, slots=True)
class IngestSettings:
    """How documents are turned into text.

    Docling does the reading: layout analysis, table structure, the office
    formats, and OCR for scans. The built-in readers remain behind it as the
    fallback for anything it cannot open, and as what runs when it is not
    installed at all.

    Two consequences are worth knowing rather than discovering. Model
    weights download on first use unless ``docling_artifacts_path`` points
    at a copy already on disk. And conversion means model inference - with
    OCR, native image decoders too - over external documents, which is a
    larger attack surface than a regular expression over a content stream;
    ``docling_enabled: false`` falls back to the readers that are not.
    """

    docling_enabled: bool = True
    #: off - never; auto - only when the text layer comes out illegible,
    #: which is what a scan looks like; always - every document.
    ocr: str = "auto"
    ocr_languages: tuple[str, ...] = ("en",)
    #: Stop after this many pages rather than spending minutes on one
    #: document. Conversion is bounded by page count because there is no
    #: honest way to interrupt it partway: a timeout would abandon the
    #: result while the work carried on.
    max_pages: int = 300
    #: Pre-downloaded model artifacts. Set it to convert without a network.
    docling_artifacts_path: str | None = None

    def __post_init__(self) -> None:
        if self.ocr not in ("off", "auto", "always"):
            raise ValueError(f"ocr must be off, auto or always, not {self.ocr!r}")

    @property
    def ocr_when_illegible(self) -> bool:
        return self.ocr in ("auto", "always")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ocr_languages"] = list(self.ocr_languages)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestSettings":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        if "ocr_languages" in known:
            known["ocr_languages"] = tuple(known["ocr_languages"])
        if "ocr" in known and isinstance(known["ocr"], bool):
            # 'ocr: true' in a config file is an understandable thing to write.
            known["ocr"] = "auto" if known["ocr"] else "off"
        return cls(**known)


@dataclass(slots=True)
class ResearchConfig:
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    acquisition: AcquisitionPolicy = field(default_factory=AcquisitionPolicy)
    model: ModelSettings = field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)
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
            "ingest": self.ingest.to_dict(),
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
    ingest = IngestSettings.from_dict(data.get("ingest", {}))

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
        ingest=ingest,
        providers=providers,
        database_path=data.get("database_path"),
        contact_email=contact,
        _secrets=secrets,
    )
