"""
This module defines the base class for embedding model services. An embedding model service is a deployed model (or a
selection of models) that can be used for generating embeddings from input data, such as text or images. Embeddings are
vector representations of the input data that can be used for various downstream tasks, such as similarity search or
input to other models.
"""

import abc
from typing import Any, Dict, List, Optional, Union

import httpx

from allin_llmflow.assets.model_services._base_model_service import _ModelService


class _EmbeddingModelService(_ModelService, metaclass=abc.ABCMeta):
    """
    An abstract base class of a generic service for managing and interacting with embedding models.
    """

    @abc.abstractmethod
    def infer(
        self,
        query: str,
        *,
        model: Optional[str],
        timeout: Optional[float],
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Generate embeddings for the given query using the model service.

        :param query: The query to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param kwargs: Additional keyword arguments to pass to the client.
        :return: The list of embeddings.
        """
        raise NotImplementedError("infer method must be implemented")


class _HttpxEmbeddingModelService(_EmbeddingModelService, metaclass=abc.ABCMeta):
    """
    An abstract base class of a generic service for managing and interacting with embedding models using HTTPX.
    """

    @abc.abstractmethod
    def build_request(
        self,
        client: httpx.Client,
        texts: Union[str, List[str], List[Dict[str, str]]],
        *,
        model: Optional[str] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Request:
        """
        Build a request according to the corresponding embedding API.

        :param client: The httpx client to use for the request.
        :param texts: The texts to send to the model.
        :param model: The model to use for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param kwargs: Additional keyword arguments to pass to the httpx client.
        :return: The httpx request object.
        """
        raise NotImplementedError("build_request method must be implemented")

    @abc.abstractmethod
    def parse_response(
        self,
        response: httpx.Response,
    ) -> List[List[float]]:
        """
        Parse the response from the corresponding embedding API to a list of embeddings.

        :param response: The httpx response from the model service.
        :return: The list of embeddings.
        """
        raise NotImplementedError("parse_response method must be implemented")

    def infer(
        self,
        query: str,
        *,
        model: Optional[str],
        timeout: Optional[float],
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Generate embeddings for the given query using the model service.

        :param query: The query to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param kwargs: Additional keyword arguments to pass to the client.
        :return: The list of embeddings.
        """
        with httpx.Client(timeout=timeout) as client:
            request = self.build_request(
                client,
                [query],
                model=model,
                inference_kwargs=inference_kwargs,
            )
            response = client.send(request).raise_for_status()
        return self.parse_response(response)
