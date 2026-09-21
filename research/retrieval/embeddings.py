"""Vector retrieval.

Embeddings waited until there was a retrieval problem lexical search could
not solve, which is the bar the brief set. There is one: a query and a
document can be about the same thing in different words, and BM25 cannot see
that. So vectors are now a standing part of the ranking rather than a
switch, and the default embedder runs locally - a small sentence encoder, no
credential, no request leaving the machine.

Two things keep that from becoming an architecture. There is no vector
database: vectors are blobs in the same SQLite file as everything else, so
there is no second store that can disagree with the first. And similarity is
computed only over candidates the other retrievers already surfaced, so the
work is proportional to a shortlist rather than to the corpus.

A hosted embedder is still available for anyone who wants one, behind the
same protocol. Nothing above this module knows which is in use.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from research.models.common import utcnow
from research.storage.database import Database, encode_dt


@runtime_checkable
class EmbeddingClient(Protocol):
    """Whatever produces vectors. Provider-agnostic, like the model client."""

    model: str
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True, slots=True)
class VectorHit:
    document_id: str
    similarity: float


def pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class LocalEmbedder:
    """A sentence encoder running in this process.

    This is the default, and the reason vectors can be on by default at all:
    it needs no API key, makes no request once its weights are present, and
    costs milliseconds per document on a CPU. The model is small - a few
    hundred megabytes at most - and is loaded once, lazily, so a run that
    never searches never pays for it.

    Weights come from the Hugging Face cache, downloaded on first use unless
    ``model_path`` points at a copy already on disk. When they cannot be
    loaded at all, :class:`EmbeddingIndex` degrades to lexical and graph
    retrieval rather than failing the search.
    """

    #: 384 dimensions, ~90MB, and good enough at the distinction that
    #: matters here: two documents about the same thing in different words.
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        dimensions: int = 384,
        model_path: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.model_path = model_path
        self.max_tokens = max_tokens
        self._loaded: Any = None
        self._failed: str | None = None

    def _load(self) -> Any:
        if self._loaded is not None or self._failed is not None:
            return self._loaded
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            source = self.model_path or self.model
            tokenizer = AutoTokenizer.from_pretrained(source)
            model = AutoModel.from_pretrained(source)
            model.eval()
            self.dimensions = int(getattr(model.config, "hidden_size", self.dimensions))
            self._loaded = (torch, tokenizer, model)
        except Exception as exc:  # not installed, not downloaded, no network
            self._failed = f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"
        return self._loaded

    @property
    def unavailable(self) -> str | None:
        """Why this embedder cannot run, once something has tried to use it."""
        return self._failed

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        import asyncio

        return await asyncio.to_thread(self._embed, list(texts))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        loaded = self._load()
        if loaded is None:
            return []
        torch, tokenizer, model = loaded
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        # Mean pooling over real tokens only: padding must not dilute a
        # short abstract towards the middle of the space.
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return [[float(value) for value in row] for row in pooled]


class OpenAICompatibleEmbedder:
    """Any ``/embeddings`` endpoint of the OpenAI shape.

    That covers OpenAI, most hosted providers, and local servers - which is
    the point: enabling vectors should not mean committing to a vendor.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimensions: int = 1536,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        transport: Any = None,
    ) -> None:
        from research.llm.transport import ModelTransport

        self.model = model
        self.dimensions = dimensions
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.transport = transport or ModelTransport()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        payload = await self.transport.post_json(
            f"{self.base_url}/embeddings",
            provider="embeddings",
            payload={"model": self.model, "input": list(texts)},
            headers=headers,
        )
        data = payload.get("data") or []
        return [list(entry.get("embedding") or []) for entry in data]


class HashingEmbedder:
    """A deterministic stand-in, for tests only.

    It is not semantic and makes no claim to be: it hashes tokens into a
    fixed number of buckets, so two texts sharing words come out close and two
    texts sharing meaning do not. It exists so the vector path can be
    exercised without a credential, and it should never be enabled for real
    research - which is why nothing selects it from configuration.
    """

    model = "hashing-stub"

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in (text or "").lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimensions
            vector[bucket] += 1.0
        return vector


class EmbeddingIndex:
    """Stores and queries document vectors."""

    def __init__(self, db: Database, client: EmbeddingClient) -> None:
        self.db = db
        self.client = client
        self._failure: str | None = None

    async def index(self, documents: Sequence[Any], *, batch: int = 32) -> int:
        """Embed and store documents that do not already have a vector."""
        pending = [
            document
            for document in documents
            if not self.has(document.id) and (document.best_text or document.title)
        ]
        written = 0
        for start in range(0, len(pending), batch):
            chunk = pending[start : start + batch]
            try:
                vectors = await self.client.embed([_embeddable(document) for document in chunk])
            except Exception as exc:
                # An embedder that cannot run costs the ranking a retriever,
                # not the search. Lexical and graph results still stand.
                self._failure = f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"
                return written
            if not vectors:
                return written
            rows = [
                (
                    document.id,
                    document.investigation_id,
                    self.client.model,
                    len(vector),
                    pack(vector),
                    encode_dt(utcnow()),
                )
                for document, vector in zip(chunk, vectors, strict=False)
                if vector
            ]
            with self.db.transaction() as connection:
                connection.executemany(
                    "INSERT INTO document_embeddings"
                    "(document_id, investigation_id, model, dimensions, vector, created_at) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET "
                    "model = excluded.model, dimensions = excluded.dimensions, "
                    "vector = excluded.vector, created_at = excluded.created_at",
                    rows,
                )
            written += len(rows)
        return written

    def has(self, document_id: str) -> bool:
        return bool(
            self.db.scalar(
                "SELECT 1 FROM document_embeddings WHERE document_id = ? AND model = ?",
                (document_id, self.client.model),
            )
        )

    def count(self, investigation_id: str | None = None) -> int:
        if investigation_id is None:
            return int(self.db.scalar("SELECT COUNT(*) FROM document_embeddings") or 0)
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM document_embeddings WHERE investigation_id = ?",
                (investigation_id,),
            )
            or 0
        )

    async def ensure(self, documents: Sequence[Any]) -> int:
        """Embed anything in ``documents`` that has no vector yet."""
        return await self.index(documents)

    def missing(self, investigation_id: str, *, limit: int) -> list[str]:
        """Held documents with no vector for the current model.

        This is what lets vectors be on without a separate indexing step: a
        search tops up the index by a bounded amount, and converges on a
        fully embedded corpus over the first few searches instead of
        stalling on the first one.
        """
        rows = self.db.query(
            "SELECT documents.id AS id FROM documents "
            "LEFT JOIN document_embeddings AS vectors "
            "  ON vectors.document_id = documents.id AND vectors.model = ? "
            "WHERE documents.investigation_id = ? AND vectors.document_id IS NULL "
            "ORDER BY documents.id LIMIT ?",
            (self.client.model, investigation_id, limit),
        )
        return [row["id"] for row in rows]

    @property
    def unavailable(self) -> str | None:
        """Why vectors are not usable right now, if they are not.

        A missing model is a degraded ranking, not a failed search, so this
        is reported rather than raised.
        """
        return self._failure or getattr(self.client, "unavailable", None)

    async def search(
        self,
        query: str,
        *,
        investigation_id: str | None = None,
        limit: int = 20,
        candidates: Sequence[str] | None = None,
    ) -> list[VectorHit]:
        """Rank stored vectors against the query.

        ``candidates`` restricts the comparison to documents the other
        retrievers already surfaced, which keeps this linear in the size of
        the shortlist rather than the corpus.
        """
        try:
            vectors = await self.client.embed([query])
        except Exception as exc:
            # A hosted embedder that is down, or a model that will not load,
            # costs this ranking its third opinion. Lexical and graph
            # results still stand, and the caller reports the degradation.
            self._failure = f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"
            return []
        if not vectors or not vectors[0]:
            return []
        query_vector = vectors[0]

        sql = "SELECT document_id, vector FROM document_embeddings WHERE model = ?"
        params: list[Any] = [self.client.model]
        if investigation_id is not None:
            sql += " AND investigation_id = ?"
            params.append(investigation_id)
        if candidates:
            placeholders = ",".join("?" for _ in candidates)
            sql += f" AND document_id IN ({placeholders})"
            params.extend(candidates)

        rows = self.db.query(sql, tuple(params))
        scored = _similarities(query_vector, rows)
        scored.sort(key=lambda hit: (-hit.similarity, hit.document_id))
        return [hit for hit in scored if hit.similarity > 0][:limit]



def _similarities(query_vector: Sequence[float], rows: Sequence[Any]) -> list[VectorHit]:
    """Cosine against every stored vector.

    Searching the whole corpus is the point - vectors have to be able to
    find a document that shares no words with the query, which is the one
    thing lexical retrieval cannot do. A corpus bounded by a document budget
    is small enough that this is a matrix multiply, not an index server.
    """
    if not rows:
        return []
    try:
        import numpy

        matrix = numpy.array(
            [unpack(row["vector"]) for row in rows], dtype=numpy.float32
        )
        query = numpy.array(query_vector, dtype=numpy.float32)
        norms = numpy.linalg.norm(matrix, axis=1) * float(numpy.linalg.norm(query))
        with numpy.errstate(divide="ignore", invalid="ignore"):
            scores = numpy.where(norms > 0, matrix @ query / norms, 0.0)
        return [
            VectorHit(row["document_id"], float(score))
            for row, score in zip(rows, scores, strict=True)
        ]
    except (ImportError, ValueError):
        # ValueError covers a ragged matrix: vectors written by a model with
        # different dimensions, which cosine() rejects one at a time anyway.
        return [
            VectorHit(row["document_id"], cosine(query_vector, unpack(row["vector"])))
            for row in rows
        ]

def _embeddable(document: Any) -> str:
    """What of a document to embed.

    Title and abstract carry the subject; a whole article body dilutes it and
    most embedding models would truncate it anyway.
    """
    parts = [document.title or "", document.abstract or "", (document.text or "")[:2000]]
    return "\n\n".join(part for part in parts if part).strip()
