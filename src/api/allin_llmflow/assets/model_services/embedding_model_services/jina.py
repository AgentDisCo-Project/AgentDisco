"""
This module defines JinaEmbeddingModelService, a ModelService asset that refers to a Jina embedding model.
"""

from typing import Optional, Union, List, Dict, Any

import httpx

from allin_llmflow.assets.model_services.embedding_model_services._base_embedding_model_service import (
    _HttpxEmbeddingModelService,
)


class JinaEmbeddingModelService(_HttpxEmbeddingModelService):
    """
    A ModelService asset that refers to a Jina embedding model.
    """

    INFERENCE_API_FORMAT = "jina-embedding"

    def build_request(
        self,
        client: httpx.Client,
        texts: Union[str, List[str], List[Dict[str, str]]],
        model: Optional[str] = None,
        *,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Request:
        """
        Build a request according to the Jina API format.

        :param client: The httpx client to use for the request.
        :param texts: The texts to send to the model.
        :param model: The model to use for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param kwargs: Additional keyword arguments to pass to the httpx client.
        :return: The httpx request object.
        """
        model = model or "jina-embeddings-v2-base-zh"
        inference_kwargs = inference_kwargs or {}
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": model,
            "input": texts,
            **inference_kwargs,
        }

        # Build the request
        request = client.build_request(
            "POST",
            url=self.inference_uri,
            json=body,
            headers=headers,
            **kwargs,
        )

        return request

    def parse_response(
        self,
        response: httpx.Response,
    ) -> List[List[float]]:
        """
        Parse the response from the Jina API to a list of embeddings.

        :param response: The response body from the Jina API.
        :return: The list of embeddings.
        """
        response_body = response.json()
        data = response_body.get("data", [])
        return [entry.get("embedding") for entry in data]
