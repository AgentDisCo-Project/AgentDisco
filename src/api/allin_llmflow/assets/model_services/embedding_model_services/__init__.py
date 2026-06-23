"""
This module provides a collection of classes that represent and manage embedding model services in the application.

An embedding model service is a deployed model (or a selection of models) that can be used for generating embeddings
from input data, such as text or images. Embeddings are vector representations of the input data that can be used for
various downstream tasks, such as similarity search or input to other models.

Embedding model services include:

- EmbeddingModelService:
    An abstract base class of a generic service for managing and interacting with embedding models.
- HttpxEmbeddingModelService:
    An abstract base class of a generic service for managing and interacting with embedding models using HTTPX.
- JinaEmbeddingModelService:
    A service specifically designed for managing and interacting with Jina embedding models.
"""

from allin_llmflow.assets.model_services.embedding_model_services._base_embedding_model_service import (
    _EmbeddingModelService as EmbeddingModelService,
    _HttpxEmbeddingModelService as HttpxEmbeddingModelService,
)
from allin_llmflow.assets.model_services.embedding_model_services.jina import JinaEmbeddingModelService
from allin_llmflow.assets.model_services.embedding_model_services.openai import OpenAIEmbeddingModelService

__all__ = [
    "EmbeddingModelService",
    "HttpxEmbeddingModelService",
    "OpenAIEmbeddingModelService",
    "JinaEmbeddingModelService",
]
