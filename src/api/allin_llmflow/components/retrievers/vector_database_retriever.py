"""
This module contains a retriever that uses a VectorDatabase to retrieve relevant documents based on a query.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from haystack import component, Document, default_to_dict, default_from_dict
from typing_extensions import Self

from allin_llmflow.assets.asset_factory import VectorDatabaseFactory
from allin_llmflow.assets.vector_databases import VectorDatabase
from allin_llmflow.dataclasses.asset_reference import AssetReference
from allin_llmflow.utils.embedders import load_embedder


@component
class VectorDatabaseRetriever:
    """
    A retriever that uses a VectorDatabase to retrieve relevant documents based on a query.

    The retriever first embeds the query using the specified embedder from associated embedding fields in the
    VectorDatabase metadata. It then performs a vector-based search on the VectorDatabase and returns the top-k results.

    :param vector_database: The VectorDatabase instance to use for retrieving documents.
    :param meta_fields: A list of fields to include in the output documents, defaults to None.
    :param embedding_fields: A list of fields to use for retrieving documents, defaults to None.
    :param top_k: The number of documents to retrieve, defaults to 10.
    :param top_k_by_embedding_field: The number of documents to retrieve per embedding field in hybrid search. This can
        be a dict mapping different embedding fields to the number of documents to retrieve to allow for different
        top-k values per field. Defaults to None.
    :param timeout: The maximum time to wait for the search to complete, defaults to None.
    :param filters: Additional filters to apply to the search results. The actual format of the filters is dependent on
        the specific VectorDatabase on which the retriever is based. Defaults to None.
    :param reranker: A serializable dictionary containing the configuration for a reranker to use for hybrid search
        during retrieval. This reranker is different from the independent reranker component, and it is only used for
        hybrid search. The actual format of the reranker configuration is dependent on the specific VectorDatabase on
        which the retriever is based. Defaults to None.
    :param retrieval_kwargs: Additional parameters for vector search. The options are dependent on the specific
        VectorDatabase on which the retriever is based. Defaults to None.
    """

    def __init__(
        self,
        vector_database: VectorDatabase,
        meta_fields: Optional[Iterable[str]] = None,
        embedding_fields: Optional[List[str]] = None,
        top_k: int = 10,
        top_k_by_embedding_field: Optional[Union[int, Dict[str, int]]] = None,
        timeout: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        reranker: Optional[Dict[str, Any]] = None,
        retrieval_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.vector_database = vector_database
        self.embedding_fields = embedding_fields or list(self.vector_database.embedding_fields.keys())
        self.top_k = top_k
        self.top_k_by_embedding_field = top_k_by_embedding_field
        self.timeout = timeout
        self.filters = filters or {}
        self.reranker = reranker
        self.retrieval_kwargs = retrieval_kwargs or {}
        if meta_fields is not None:
            self.vector_database.validate_meta_fields(meta_fields)
            self.meta_fields = list(meta_fields)
        else:
            self.meta_fields = list(self.vector_database.meta_fields)

        # Initialize embedders based on the embedding configuration of the field
        self.embedders: Dict[str, Callable[[str], List[List[float]]]] = {}
        for embedding_field in self.embedding_fields:
            if embedding_field not in self.vector_database.embedding_fields:
                raise ValueError(f"embedding field {embedding_field} not found in embedding_fields.")
            # Initialize embedders based on the embedding configuration of the field
            embedding_configuration = self.vector_database.embedding_fields[embedding_field]
            self.embedders[embedding_field] = load_embedder(
                embedding_configuration.from_asset, embedding_configuration.embedder_kwargs
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a serialized dictionary representation of the component.

        :return: A dictionary representation of the component.
        """
        vector_database_reference = self.vector_database.reference
        return default_to_dict(
            self,
            vector_database=vector_database_reference.to_dict(),
            meta_fields=self.meta_fields,
            embedding_fields=self.embedding_fields,
            top_k=self.top_k,
            top_k_by_embedding_field=self.top_k_by_embedding_field,
            timeout=self.timeout,
            filters=self.filters,
            reranker=self.reranker,
            retrieval_kwargs=self.retrieval_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Loads a VectorDatabaseRetriever component from its serialized dictionary representation.

        :param data: The serialized dictionary representation of the component.
        :return: The deserialized VectorDatabaseRetriever component.
        """
        # Make a shallow copy of init_parameters to avoid modifying the original data
        init_parameters_data = data["init_parameters"].copy()
        # Load the chat model service from reference
        vector_database_reference = AssetReference.from_dict(init_parameters_data["vector_database"])
        init_parameters_data["vector_database"] = VectorDatabaseFactory.load_from_reference(vector_database_reference)

        # Load the component
        return default_from_dict(cls, {"type": data["type"], "init_parameters": init_parameters_data})

    @component.output_types(documents=List[Document])
    def run(
        self, query: str, filters: Optional[Dict[str, Any]] = None, retrieval_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Retrieve documents based on the given query.

        :param query: The search query to retrieve documents for, in natural language.
        :param filters: Additional filters to apply to the search results, defaults to None.
        :param retrieval_kwargs: Additional parameters for vector search. The options are dependent on the specific
            VectorDatabase on which the retriever is based. Defaults to None.
        :returns: A dictionary containing the retrieved documents.
        """
        # Merge the filters and retrieval_kwargs with the default values
        filters = {**self.filters, **(filters or {})}
        retrieval_kwargs = {**self.retrieval_kwargs, **(retrieval_kwargs or {})}
        query_embeddings = {
            embedding_field: self.embedders[embedding_field](query) for embedding_field in self.embedding_fields
        }
        documents = self.vector_database.search(
            query_embeddings=query_embeddings,
            embedding_fields=self.embedding_fields,
            meta_fields=self.meta_fields,
            top_k=self.top_k,
            top_k_by_embedding_field=self.top_k_by_embedding_field,
            timeout=self.timeout,
            filters=filters,
            reranker=self.reranker,
            **retrieval_kwargs,
        )
        return {"documents": documents}
