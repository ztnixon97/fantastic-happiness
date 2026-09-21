"""Shared fixtures.

Tests never touch a live API. Providers are exercised through recorded
payloads served by an httpx mock transport, which keeps the suite
deterministic and fast.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from research.config import AcquisitionPolicy, BudgetPolicy, ResearchConfig
from research.budgets import BudgetLedger
from research.sources.http import SafeHttpClient
from research.sources.offline import build_offline_registry, load_corpus
from research.storage.store import ResearchStore

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch) -> None:
    """No test sees the host's credentials.

    A suite whose result depends on whether the developer happens to have an
    API key exported is not a suite.
    """
    for name in list(os.environ):
        if name.startswith("RESEARCH_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def store() -> ResearchStore:
    with ResearchStore.in_memory() as opened:
        yield opened


@pytest.fixture
def investigation(store: ResearchStore):
    return store.investigations.create("test question")


@pytest.fixture
def config() -> ResearchConfig:
    return ResearchConfig(
        budget=BudgetPolicy(),
        acquisition=AcquisitionPolicy(
            per_host_min_interval_seconds=0.0, retry_backoff_seconds=0.0
        ),
    )


@pytest.fixture
def ledger(store: ResearchStore, investigation, config: ResearchConfig) -> BudgetLedger:
    return BudgetLedger(store.budget, investigation.id, config.budget)


@pytest.fixture
def make_client(config: ResearchConfig) -> Callable[..., SafeHttpClient]:
    """Build a client wired to a caller-supplied request handler."""
    created: list[SafeHttpClient] = []

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> SafeHttpClient:
        client = SafeHttpClient(
            config.acquisition,
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        created.append(client)
        return client

    yield factory
    for client in created:
        # The mock transport holds no sockets; closing is bookkeeping only.
        client._client._transport = None  # type: ignore[attr-defined]


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def corpus() -> dict[str, Any]:
    return load_corpus()


@pytest.fixture
def offline(corpus, config):
    """Registry, client and store backed entirely by the demo corpus."""
    registry, client = build_offline_registry(corpus, config)
    with ResearchStore.in_memory() as opened:
        yield opened, registry, client
