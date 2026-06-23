"""
This module provides a collection of classes that represent and manage embedding functions in the application.
"""

from typing import Dict, Type

from allin_llmflow.assets.embedding_functions._base_embedding_function import _EmbeddingFunction as EmbeddingFunction
from allin_llmflow.assets.embedding_functions.milvus_bm25 import BM25EmbeddingFunction

__all__ = [
    "EmbeddingFunction",
    "BM25EmbeddingFunction",
]

SUPPORTED_EMBEDDING_FUNCTIONS: Dict[str, Type[EmbeddingFunction]] = {
    v.EMBEDDING_STRATEGY: v
    for v in globals().values()
    if isinstance(v, type)
    and issubclass(v, EmbeddingFunction)
    and not v == EmbeddingFunction
    and not bool(v.__abstractmethods__)
}
"""A dictionary mapping embedding strategies to their corresponding EmbeddingFunction classes."""
