"""Shared fixtures.

Tests never touch a live API. Providers are exercised through recorded
payloads served by an httpx mock transport, which keeps the suite
deterministic and fast.
"""

from __future__ import annotations

import json
import os
import zlib
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


@pytest.fixture(autouse=True)
def no_model_inference(monkeypatch) -> None:
    """The suite never runs a model, whatever the defaults say.

    Docling and the sentence encoder are both on by default now, and both
    download weights on first use and take seconds to tens of seconds per
    document. A suite whose result depends on a network, a cache directory,
    or a machine's spare CPU is not a suite.

    So both are made unavailable here, which exercises exactly the path a
    machine without them takes: the built-in readers, and a ranking without
    its third opinion. Tests that need either behaviour supply a stub and
    assert on what the wiring does with it - see tests/unit/test_ingest.py
    and tests/unit/test_retrieval.py.
    """
    from research.normalize import docling_reader
    from research.retrieval import embeddings

    monkeypatch.setattr(docling_reader, "available", lambda: False)
    monkeypatch.setattr(
        embeddings.LocalEmbedder,
        "_load",
        lambda self: setattr(self, "_failed", "disabled for tests") or None,
    )


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


def build_pdf(
    paragraphs: list[str],
    *,
    compress: bool = True,
    title: str | None = None,
    hex_strings: bool = False,
) -> bytes:
    """A tiny but structurally valid PDF carrying the given lines of text.

    Real PDFs are not checked into the suite: a generated one is
    deterministic, is small enough to read, and can be varied - compressed
    or not, text or unmappable two-byte codes - to exercise the branches
    that matter.
    """
    lines = [b"BT /F1 12 Tf 72 720 Td"]
    for index, paragraph in enumerate(paragraphs):
        if index:
            lines.append(b"0 -16 Td")
        if hex_strings:
            # Two-byte font codes with no CMap: what a CID-encoded PDF looks
            # like to a reader that cannot resolve its encoding.
            codes = "".join(f"{ord(character) + 0x2800:04x}" for character in paragraph)
            lines.append(b"<" + codes.encode("ascii") + b"> Tj")
            continue
        escaped = paragraph.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        lines.append(b"(" + escaped.encode("latin-1") + b") Tj")
    lines.append(b"ET")
    content = b"\n".join(lines)
    stream = zlib.compress(content) if compress else content
    filter_entry = b"/Filter /FlateDecode " if compress else b""

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< " + filter_entry + b"/Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    if title:
        objects.append(b"<< /Title (" + title.encode("latin-1") + b") >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    trailer = b"<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R"
    if title:
        trailer += b" /Info " + str(len(objects)).encode() + b" 0 R"
    out += b"trailer\n" + trailer + b" >>\nstartxref\n" + str(xref_at).encode() + b"\n%%EOF\n"
    return bytes(out)


@pytest.fixture(name="build_pdf")
def build_pdf_fixture():
    return build_pdf
