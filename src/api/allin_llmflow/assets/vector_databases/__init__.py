"""
This module provides a collection of classes for interacting with different vector databases.

A vector database is a database that stores vector representations of data. Vector databases are widely used as the
backend for retrieval-based applications that require fast and efficient similarity search to find the most relevant
data points, playing a crucial role in Retrieval Augmented Generation (RAG) applications.

Vector databases include:

- VectorDatabase:
    An abstract base class for managing and interacting with generic vector databases.
- MilvusVectorDatabase:
    A class specifically designed for managing and interacting with Milvus vector databases.
"""

from typing import Dict, Type

from allin_llmflow.assets.vector_databases._base_vector_database import _VectorDatabase as VectorDatabase
from allin_llmflow.assets.vector_databases.milvus import MilvusVectorDatabase

__all__ = [
    "VectorDatabase",
    "MilvusVectorDatabase",
]

# Register all supported vector database types
SUPPORTED_VECTORDB_TYPES: Dict[str, Type[VectorDatabase]] = {
    v.VECTOR_DB_TYPE: v
    for v in globals().values()
    if isinstance(v, type) and issubclass(v, VectorDatabase) and not bool(v.__abstractmethods__)
}
"""A dictionary mapping vector database types to their corresponding VectorDatabase classes."""
