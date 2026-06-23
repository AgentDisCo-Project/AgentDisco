"""
This module defines the base class for all embedding functions. An embedding function is an offline function that
encodes a query into a vector representation. Most embedding functions are based on pre-trained models and require
external files to be loaded before encoding queries.
"""

import abc
from typing import Optional, Dict, Any

from allin_llmflow.assets._base_asset import _Asset


class _EmbeddingFunction(_Asset, metaclass=abc.ABCMeta):
    """
    Base class for all embedding functions.
    """

    ASSET_TYPE = "embedding-function"
    EMBEDDING_STRATEGY: str = NotImplemented
    """The embedding strategy of the EmbeddingFunction. This should be overridden by subclasses."""

    def __init__(self, *, name: Optional[str] = None, token: Optional[str] = None, **kwargs) -> None:
        configs = {
            "embedding_strategy": self.EMBEDDING_STRATEGY,
            **kwargs,
        }
        if token:
            secrets = {"token": token}
        else:
            secrets = {}

        super().__init__(name, configs=configs, secrets=secrets)

    @abc.abstractmethod
    def load_and_init_embedder(self, local_path: str, embedder_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Load the embedding function from the remote path and initialize the embedder.

        :param local_path: The local path to save the embedding function.
        :param embedder_kwargs: The keyword arguments to pass to the embedding function.
        """
        raise NotImplementedError("load_and_init_embedder method must be implemented in the subclass")

    @abc.abstractmethod
    def encode_query(self, query: str) -> Any:
        """
        Encode the query using the embedding function.

        :param query: The query to encode.
        :return: The encoded query.
        """
        raise NotImplementedError("encode_query method must be implemented in the subclass")
