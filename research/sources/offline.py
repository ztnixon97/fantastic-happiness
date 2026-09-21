"""Offline mode.

Builds a registry whose providers serve a fixed corpus, including an HTTP
transport that answers fetches from the corpus's stored pages. The real
fetcher, the real HTML extractor and the real deduplication rules all run;
only the network is replaced.

This is what makes the demonstration reproducible and the test suite
independent of live APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from research.config import ResearchConfig
from research.models.common import SourceFamily
from research.normalize.urls import url_identity_key
from research.sources.fetcher import DirectFetchSource
from research.sources.http import SafeHttpClient
from research.sources.registry import SourceRegistry
from research.sources.static import StaticAcademicSource, StaticWebSource

DEMO_CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_corpus.json"


def load_corpus(path: str | Path | None = None) -> dict[str, Any]:
    corpus_path = Path(path) if path else DEMO_CORPUS_PATH
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"corpus must be a mapping: {corpus_path}")
    data.setdefault("academic", [])
    data.setdefault("web", [])
    return data


def corpus_transport(entries: list[dict[str, Any]]) -> httpx.MockTransport:
    """An HTTP transport that serves the corpus's pages and 404s everything else."""
    pages: dict[str, str] = {}
    for entry in entries:
        key = url_identity_key(entry.get("url"))
        if key and entry.get("html"):
            pages[key] = entry["html"]

    def handler(request: httpx.Request) -> httpx.Response:
        key = url_identity_key(str(request.url))
        html = pages.get(key or "")
        if html is None:
            return httpx.Response(404, text="not found in offline corpus")
        return httpx.Response(
            200, text=html, headers={"content-type": "text/html; charset=utf-8"}
        )

    return httpx.MockTransport(handler)


def build_offline_registry(
    corpus: dict[str, Any],
    config: ResearchConfig | None = None,
) -> tuple[SourceRegistry, SafeHttpClient]:
    """Registry plus the client it uses, both backed by the corpus."""
    settings = config or ResearchConfig()
    client = SafeHttpClient(
        settings.acquisition, transport=corpus_transport(corpus.get("web", []))
    )
    registry = SourceRegistry()
    registry.register(StaticAcademicSource(corpus.get("academic", []), name="offline_academic"))
    registry.register(
        StaticWebSource(corpus.get("web", []), name="offline_web", family=SourceFamily.WEB)
    )
    fetcher = DirectFetchSource(
        client, max_text_characters=settings.acquisition.max_text_characters
    )
    registry.register(fetcher)
    registry.fetcher = fetcher
    return registry, client
