"""
This module defines the base class for vector databases. A vector database is a database that stores vector
representations of data. Vector databases are widely used as the backend for retrieval-based applications that require
fast and efficient similarity search to find the most relevant data points, playing a crucial role in Retrieval
Augmented Generation (RAG) applications.
"""

import abc
from typing import Any, Dict, Iterable, List, Optional, Union

from haystack import Document

from allin_llmflow.assets._base_asset import _Asset
from allin_llmflow.dataclasses import EmbeddingField


class _VectorDatabase(_Asset, metaclass=abc.ABCMeta):
    """
    A VectorDatabase asset refers to a database that stores vector representations of data. Either a uri or host
    and port must be provided to connect to the database.

    :param name: The name of the vector database.
    :param collection_name: The name of the collection in the database.
    :param content_field: The field in the database that contains the raw content from which embeddings are generated.
    :param embedding_fields: A dictionary mapping field names to embedding configurations.
    :param meta_fields: The fields in the database that contain metadata.
    :param uri: The URI of the vector database server. This can be in the form of a URL or a connection string, e.g.,
        "http://localhost:19530", "tcp:localhost:19530", "https://ok.s3.south.com:19530".
    :param host: The host of the vector database server.
    :param port: The port of the vector database server.
    """

    ASSET_TYPE = "vector-database"
    VECTOR_DB_TYPE: str = NotImplemented
    """The type of the vector database. This should be overridden by subclasses."""

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        collection_name: str,
        content_field: str,
        embedding_fields: Dict[str, Dict[str, Any]],
        meta_fields: Optional[Iterable[str]] = None,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        group_by: Optional[Dict[str, Any]] = None,
    ):
        if not uri and not (host and port):
            raise ValueError("Either a uri or host and port must be provided to connect to the vector database.")
        configs: Dict[str, Any] = {
            "vector_db_type": self.VECTOR_DB_TYPE,
            "collection_name": collection_name,
            "content_field": content_field,
            "embedding_fields": embedding_fields,
            "meta_fields": list(meta_fields) if meta_fields else [],
            "uri": uri,
            "host": host,
            "port": port,
            "group_by": group_by,
        }
        secrets: Dict[str, Any] = {}

        self.embedding_fields = {key: EmbeddingField.from_dict(value) for key, value in embedding_fields.items()}
        self.content_field = content_field
        self.meta_fields = set(meta_fields if meta_fields else ())
        self.group_by = group_by or {}
        super().__init__(name, configs=configs, secrets=secrets)

    def validate_meta_fields(self, meta_fields: Iterable[str]) -> None:
        """
        Validate that the meta fields are a subset of the available meta fields in the database.

        :param meta_fields: The meta fields to validate.
        :raises ValueError: If the meta fields are not a subset of the available meta fields in the database.
        """
        if not set(meta_fields).issubset(self.meta_fields):
            raise ValueError(
                f"meta_fields must be a subset of the available meta fields ({self.meta_fields}) in the database, "
                f"got: {meta_fields}"
            )

    @abc.abstractmethod
    def search(
        self,
        query_embeddings: Any,
        *,
        embedding_fields: List[str],
        meta_fields: Optional[List[str]] = None,
        top_k: int,
        top_k_by_embedding_field: Optional[Union[int, Dict[str, int]]] = None,
        timeout: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        reranker: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Search the database using the given vector query and return the top-k documents.

        :param query_embeddings: The vector query.
        :param embedding_fields: The fields in the database to search.
        :param meta_fields: The fields to output in the search results. If None, all meta fields will be returned.
        :param top_k: The number of results to return.
        :param top_k_by_embedding_field: The number of results to return for each embedding field.
        :param timeout: The timeout for the search operation.
        :param filters: Filters to apply to the search results.
        :param reranker: The reranker to use in hybrid search.
        :param kwargs: Additional keyword arguments to pass to the search method.
        :return: The top-k documents from the search.
        """
        raise NotImplementedError("search method must be implemented")
