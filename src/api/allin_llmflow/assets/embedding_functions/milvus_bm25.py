"""
This module defines the BM25EmbeddingFunction asset, which refers to BM25 embedding functions.
"""

import logging
import subprocess
from typing import Optional, Dict, Any

from scipy.sparse import csr_array

from allin_llmflow.assets.embedding_functions._base_embedding_function import _EmbeddingFunction
from allin_llmflow.utils.lazy_imports import LazyImport

with LazyImport("Run 'pip install milvus-model'") as milvus_model_import:
    from milvus_model.sparse import BM25EmbeddingFunction as MilvusBM25EmbeddingFunction, bm25


class BM25EmbeddingFunction(_EmbeddingFunction):
    """
    BM25EmbeddingFunction assets refer to BM25 embedding functions.
    """

    EMBEDDING_STRATEGY = "BM25"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        idf_path: str,
        language: str = "zh",
        token: Optional[str] = None,
        **kwargs,
    ):
        milvus_model_import.check()
        self.idf_remote_path = idf_path
        self.language = language
        self.bm25_embedding_function: Optional[MilvusBM25EmbeddingFunction] = None
        self.kwargs = kwargs
        super().__init__(name=name, token=token, idf_path=idf_path, **kwargs)

    def load_and_init_embedder(self, local_path: str, embedder_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Load the BM25 embedding function from the remote path and initialize the embedder.

        :param local_path: The local path to save the BM25 idf file.
        :param embedder_kwargs: The keyword arguments to pass to the BM25 embedding function. (Not used)
        """
        cmd = ["redcast", "sync", "-t", "1000", "-s", "cos", "-o", local_path, "-u", self.idf_remote_path]
        logging.info("Syncing idf file to local: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        analyzer = bm25.build_default_analyzer(language=self.language)
        self.bm25_embedding_function = MilvusBM25EmbeddingFunction(analyzer, **self.kwargs)
        self.bm25_embedding_function.load(local_path)

    def encode_query(self, query: str) -> csr_array:
        """
        Encode the query using the BM25 embedding function.
        :param query: The query to encode.
        :return: The encoded query.
        :raises ValueError: If the BM25 embedding function is not loaded.
        """
        if self.bm25_embedding_function is None:
            raise ValueError("BM25 embedding function is not loaded. Please load the embedding function first.")
        return self.bm25_embedding_function.encode_queries([query])
