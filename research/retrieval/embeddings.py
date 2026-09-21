"""Optional vector retrieval.

Off by default, and deliberately so: the brief's rule is that embeddings wait
until there is a retrieval problem lexical search cannot solve, and on a
corpus of a few hundred documents BM25 plus graph expansion generally is the
solution. What this module provides is the seam - when a corpus does outgrow
lexical matching, vectors slot into the same fusion step as another ranking,
rather than becoming the architecture.

There is no vector database. Vectors live in the same SQLite file as
everything else, and similarity is computed over candidates the other
retrievers already surfaced. A corpus bounded by a document budget does not
need an index server, and adding one would mean a second store that can
disagree with the first.
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
            vectors = await self.client.embed([_embeddable(document) for document in chunk])
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
        vectors = await self.client.embed([query])
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

        scored = [
            VectorHit(row["document_id"], cosine(query_vector, unpack(row["vector"])))
            for row in self.db.query(sql, tuple(params))
        ]
        scored.sort(key=lambda hit: (-hit.similarity, hit.document_id))
        return [hit for hit in scored if hit.similarity > 0][:limit]


def _embeddable(document: Any) -> str:
    """What of a document to embed.

    Title and abstract carry the subject; a whole article body dilutes it and
    most embedding models would truncate it anyway.
    """
    parts = [document.title or "", document.abstract or "", (document.text or "")[:2000]]
    return "\n\n".join(part for part in parts if part).strip()
