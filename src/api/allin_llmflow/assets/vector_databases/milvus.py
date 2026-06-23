"""
This module defines MilvusVectorDatabase, a class specifically designed for managing and interacting with Milvus vector
databases.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Iterable

from haystack import Document

from allin_llmflow.assets.vector_databases._base_vector_database import _VectorDatabase
from allin_llmflow.utils.lazy_imports import LazyImport

with LazyImport("Run 'pip install pymilvus'") as pymilvus_import:
    from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker, WeightedRanker, SearchResult
    from pymilvus.client.abstract import BaseRanker
    from pymilvus.client.constants import RANKER_TYPE_RRF, RANKER_TYPE_WEIGHTED


class MilvusVectorDatabase(_VectorDatabase):
    """
    A VectorDatabase asset that refers to a Milvus database. Either a uri or host and port must be provided.

    :param name: The name of the vector database.
    :param collection_name: The name of the collection in the database.
    :param content_field: The field in the database that contains the raw content from which embeddings are generated.
    :param embedding_fields: A dictionary mapping field names to embedding configurations.
    :param meta_fields: The fields in the database that contain metadata.
    :param uri: The URI of the milvus server. This can be in the form of a URL or a connection string, e.g.,
        "http://localhost:19530", "tcp:localhost:19530", "https://ok.s3.south.com:19530".
    :param host: The host of the milvus server.
    :param port: The port of the milvus server.
    """

    VECTOR_DB_TYPE = "milvus"

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
        pymilvus_import.check()

        # Initialize connection to Milvus
        connections.connect(
            alias="default",
            uri=uri or "",
            host=host or "",
            port=port or "",
        )
        self.collection = Collection(collection_name)

        super().__init__(
            name=name,
            collection_name=collection_name,
            content_field=content_field,
            embedding_fields=embedding_fields,
            meta_fields=meta_fields,
            uri=uri,
            host=host,
            port=port,
            group_by=group_by,
        )

    def _normalize_results(self, result: "SearchResult") -> List[Document]:
        """
        Normalize the search result.

        :param result: A SearchResult object from Milvus search.
        :return: A list of Document objects containing the search results.
        """
        documents = []
        for hit in result[0]:
            doc = Document(
                content=hit.get(self.content_field),
                meta={key: value for key, value in hit.fields.items() if key != self.content_field},
            )
            # Always include distance and score information from search
            doc.meta["distance"] = hit.distance
            doc.meta["score"] = hit.score
            documents.append(doc)
        return documents

    def _search(
        self,
        query_embedding: Any,
        anns_field: str,
        output_fields: List[str],
        top_k: int,
        timeout: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "SearchResult":
        """
        Search the database using the given vector query.

        :param query_embedding: The vector query.
        :param anns_field: The field in the database to search.
        :param output_fields: The fields to output in the search results.
        :param top_k: The number of results to return.
        :param timeout: The timeout for the search operation.
        :param filters: Filters to apply to the search results.
        :param kwargs: Additional keyword arguments to pass to the search method.
        :returns: The search results.
        :raises ValueError: If the anns_field is not found in the embedding_fields.
        """
        if anns_field not in self.embedding_fields:
            raise ValueError(f"embedding_field {anns_field} not found in embedding_fields.")
        results = self.collection.search(
            data=query_embedding,
            anns_field=anns_field,
            param=self.embedding_fields[anns_field].search_kwargs or {},
            limit=top_k,
            expr=filters.get("expr") if filters else None,
            partition_names=kwargs.get("partition_names"),
            output_fields=output_fields,
            timeout=timeout,
            round_decimal=kwargs.get("round_decimal", -1),
            group_by_field=self.group_by.get("field"),
        )
        return results

    def _build_search_request_by_field(
        self,
        query_embedding: Any,
        anns_field: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> "AnnSearchRequest":
        """
        Build a search request for a single field.

        :param query_embedding: The vector query.
        :param anns_field: The field in the database to search.
        :param top_k: The number of results to return.
        :param filters: Filters to apply to the search results.
        :returns: A AnnSearchRequest request object which can be used to perform the search.
        :raises ValueError: If the anns_field is not found in the embedding_fields.
        """
        if anns_field not in self.embedding_fields:
            raise ValueError(f"embedding_field {anns_field} not found in embedding_fields.")
        return AnnSearchRequest(
            data=query_embedding,
            anns_field=anns_field,
            param=self.embedding_fields[anns_field].search_kwargs or {},
            limit=top_k,
            expr=filters.get("expr") if filters else None,
        )

    def _load_reranker(self, reranker_config: Optional[Dict[str, Any]]) -> "BaseRanker":
        """
        Load the reranker object based on the provided configuration. As per the current Milvus implementation, only
        RRFRanker and WeightedRanker are supported. If no reranker is provided, RRFRanker is used by default.

        :param reranker_config: The reranker configuration.
        :returns: The reranker object.
        :raises ValueError: If an invalid reranker strategy is provided.
        """
        if not reranker_config:
            logging.info("No reranker provided. Using RRFRanker by default for hybrid search.")
            return RRFRanker()

        strategy = reranker_config.get("strategy")
        if strategy == RANKER_TYPE_RRF:
            return RRFRanker(reranker_config.get("params", {}).get("k", 60))
        if strategy == RANKER_TYPE_WEIGHTED:
            return WeightedRanker(*reranker_config.get("params", {}).get("weights", []))
        raise ValueError(f"Invalid reranker strategy: {strategy}")

    def search(
        self,
        query_embeddings: Dict[str, Any],
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
        :param meta_fields: The fields to output in the search results.
        :param top_k: The number of results to return.
        :param top_k_by_embedding_field: The number of results to return for each embedding field.
        :param timeout: The timeout for the search operation.
        :param filters: Filters to apply to the search results. For MilvusVectorDatabase, this is a dictionary with a
            single key "expr" containing the filter expression. The syntax information of the expression can be found on
            https://milvus.io/docs/boolean.md.
        :param reranker: The reranker to use in hybrid search. For MilvusVectorDatabase, this is a dictionary
            representation of an RRFRanker or WeightedRanker. More information can be found on
            https://github.com/milvus-io/pymilvus/blob/master/pymilvus/client/abstract.py#L306-L339.
        :param kwargs: Additional keyword arguments to pass to the search method. Currently, only the following options
            are supported for MilvusVectorDatabase:
            `partition_names` -- A list of partition names.
            `round_decimal` -- The number of decimal places to round the search results to. -1 indicates no rounding.
            `group_by_field` -- Groups search results by a specified field to ensure diversity and avoid returning
                multiple results from the same group.
        :return: The top-k documents from the search.
        """
        output_fields = [self.content_field] + (meta_fields or [])  # Always include content field in output

        if len(embedding_fields) == 0:
            raise ValueError("embedding_fields must not be empty.")
        if len(embedding_fields) == 1:
            # Run a single field search if only one embedding field is used
            if top_k_by_embedding_field is not None:
                logging.warning(
                    "top_k_by_embedding_field is ignored when only one embedding field is used for the search."
                )
            results = self._search(
                query_embedding=query_embeddings[embedding_fields[0]],
                anns_field=embedding_fields[0],
                top_k=top_k,
                output_fields=output_fields,
                timeout=timeout,
                filters=filters,
                **kwargs,
            )
        else:
            reranker = self._load_reranker(reranker)
            # If top_k_by_embedding_field is not provided, use top_k
            top_k_by_embedding_field = top_k_by_embedding_field or top_k
            if isinstance(top_k_by_embedding_field, int):
                top_k_by_embedding_field = {anns_field: top_k_by_embedding_field for anns_field in embedding_fields}

            # Build search requests for each field and perform hybrid search
            # temporarily set top_k to be 2x larger in hybrid search to make the number of the filtered results
            # if group by field is set
            temp_scale = 2 if self.group_by.get("field", "") else 1
            requests = [
                self._build_search_request_by_field(
                    query_embedding=query_embeddings[embedding_field],
                    anns_field=embedding_field,
                    top_k=top_k_by_embedding_field.get(embedding_field, top_k) * temp_scale,
                    filters=filters,
                )
                for embedding_field in embedding_fields
            ]
            results = self.collection.hybrid_search(
                reqs=requests,
                rerank=reranker,
                limit=top_k * temp_scale,
                partition_names=kwargs.get("partition_names"),
                output_fields=output_fields,
                timeout=timeout,
                round_decimal=kwargs.get("round_decimal", -1),
                # group_by field is not supported in hybrid search. Need a temporary solution to get around this.
                # To make the number of the filtered results equal to the top_k, we can temporarily set the
                # top_k to be larger in hybrid search.
                # group_by_field=self.group_by.get("field"),
            )
            # temp solution to get around the group_by field issue
            normalized_results = self._normalize_results(results)
            grouped_results = self._group_by_results(normalized_results, top_k)
            return grouped_results

        return self._normalize_results(results)

    def _group_by_results(self, results: List[Document], top_k: int) -> List[Document]:
        """
        Group the search results by a specified field to ensure diversity and avoid returning multiple results from the
        same group, limiting the number of results to top_k.

        :param results: The search results to group.
        :param top_k: The number of results to return.
        :return: The grouped search results.
        """
        group_by_field = self.group_by.get("field")
        if not group_by_field:
            return results

        # Group the search results by the specified field
        grouped_results: List[Document] = []
        grouped_values = set()
        for doc in results:
            if len(grouped_results) >= top_k:
                break
            if doc.meta.get(group_by_field) not in grouped_values:
                grouped_results.append(doc)
                grouped_values.add(doc.meta.get(group_by_field))
        return grouped_results
