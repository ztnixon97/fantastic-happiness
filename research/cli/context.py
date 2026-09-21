"""Wiring for the command line.

Builds the store, configuration, sources and budget ledger for one command,
and closes what it opened. The CLI is an inspection and execution surface
over the research core; no research logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.agents.offline_model import offline_research_model
from research.budgets import BudgetLedger
from research.config import BudgetPolicy, ResearchConfig, load_config
from research.llm.base import ModelClient
from research.llm.factory import build_model
from research.sources.http import SafeHttpClient
from research.sources.offline import build_offline_registry, load_corpus
from research.sources.registry import SourceRegistry, build_registry
from research.storage.store import DEFAULT_DB_PATH, ResearchStore


@dataclass(slots=True)
class CliContext:
    store: ResearchStore
    config: ResearchConfig
    registry: SourceRegistry
    client: SafeHttpClient
    offline: bool = False

    def model(self, *, prefer_offline: bool = False) -> ModelClient:
        """The model this run reasons with.

        Offline runs use the rule-based researcher, which needs no credential
        and behaves deterministically; ``prefer_offline`` asks for it against
        live sources too, which is how a deployment gets smoke-tested without
        spending anything. Everything else comes from configuration: there is
        no silent fallback between vendors.
        """
        if self.offline or prefer_offline:
            return offline_research_model()
        return build_model(self.config)

    def models(self):
        """Models by role, so the planner need not share the gatherers' model.

        An offline run pins every role to the deterministic local researcher:
        tiers are about spending, and an offline run spends nothing.
        """
        from research.llm.factory import ModelPool

        if self.offline:
            return ModelPool(fixed=offline_research_model())
        return ModelPool(self.config)

    def corpus_search(self, investigation_id: str, *, embeddings: bool | None = None):
        """Retrieval over held evidence: lexical, graph and vectors."""
        from research.retrieval.search import CorpusSearch

        return CorpusSearch(
            self.store, investigation_id, embeddings=self.embedding_index(embeddings)
        )

    def embedding_index(self, enabled: bool | None = None):
        """The vector index, or None when vectors are switched off.

        The default embedder runs in this process, so a search needs no
        credential. Naming any other provider switches to an /embeddings
        endpoint of the OpenAI shape and takes its key from the same
        allowlist as everything else.
        """
        from research.retrieval.embeddings import (
            EmbeddingIndex,
            LocalEmbedder,
            OpenAICompatibleEmbedder,
        )

        settings = self.config.retrieval
        if not (settings.embeddings_enabled if enabled is None else enabled):
            return None
        if self.offline and not settings.embedding_model_path:
            # --offline promises no network. Downloading model weights on
            # first use would break that promise on exactly the machine that
            # asked for it, so vectors sit out unless the weights are
            # already named on disk.
            return None
        if settings.embedding_provider == "local":
            client = LocalEmbedder(
                settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                model_path=settings.embedding_model_path,
            )
        else:
            client = OpenAICompatibleEmbedder(
                settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                api_key=self.config.secret(settings.embedding_provider),
                base_url=settings.embedding_base_url,
            )
        return EmbeddingIndex(self.store.db, client)

    def ledger(self, investigation_id: str) -> BudgetLedger:
        """Budget ledger for an investigation, using its stored policy.

        The policy recorded when the investigation was created wins over the
        current config file, so resuming a run cannot silently widen limits
        that were already partly spent.
        """
        investigation = self.store.investigations.get(investigation_id)
        policy = (
            BudgetPolicy.from_dict(investigation.budget)
            if investigation.budget
            else self.config.budget
        )
        return BudgetLedger(self.store.budget, investigation_id, policy)

    async def aclose(self) -> None:
        await self.client.aclose()
        self.store.close()


def build_context(
    *,
    db_path: str | Path | None = None,
    config_path: str | Path | None = None,
    offline: bool = False,
    corpus_path: str | Path | None = None,
) -> CliContext:
    config = load_config(config_path)
    store = ResearchStore.open(db_path or config.database_path or DEFAULT_DB_PATH)

    if offline or corpus_path:
        corpus = load_corpus(corpus_path)
        registry, client = build_offline_registry(corpus, config)
        return CliContext(store, config, registry, client, offline=True)

    client = SafeHttpClient(config.acquisition)
    registry = build_registry(config, client)
    return CliContext(store, config, registry, client)


def corpus_metadata(corpus_path: str | Path | None = None) -> dict[str, Any]:
    corpus = load_corpus(corpus_path)
    return {
        "name": corpus.get("name"),
        "question": corpus.get("question"),
        "academic": len(corpus.get("academic", [])),
        "web": len(corpus.get("web", [])),
    }
