"""Retrieval over evidence the investigation already holds."""

from research.retrieval.embeddings import (
    EmbeddingClient,
    EmbeddingIndex,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
)
from research.retrieval.fusion import FusedHit, reciprocal_rank_fusion
from research.retrieval.graph import GraphExpansion, GraphHit
from research.retrieval.lexical import LexicalHit, LexicalIndex, prepare_query
from research.retrieval.search import CorpusHit, CorpusSearch

__all__ = [
    "CorpusHit",
    "CorpusSearch",
    "EmbeddingClient",
    "EmbeddingIndex",
    "FusedHit",
    "GraphExpansion",
    "GraphHit",
    "HashingEmbedder",
    "LexicalHit",
    "LexicalIndex",
    "OpenAICompatibleEmbedder",
    "prepare_query",
    "reciprocal_rank_fusion",
]
