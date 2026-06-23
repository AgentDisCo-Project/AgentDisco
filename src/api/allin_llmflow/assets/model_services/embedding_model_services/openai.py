"""
This module defines the OpenAIEmbeddingModelService class for handling the OpenAI embedding model service or any model
supporting the OpenAI Embedding API format.
"""

import logging
from typing import Optional, List, Dict, Any, Union

import httpx

from allin_llmflow.assets.model_services.embedding_model_services._base_embedding_model_service import (
    _HttpxEmbeddingModelService,
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingModelService(_HttpxEmbeddingModelService):
    """
    A ModelService asset that refers to a deployed OpenAI model, or a deployed model supporting OpenAI Chat Completion
    API format.
    """

    INFERENCE_API_FORMAT = "openai-embedding"

    def build_request(
        self,
        client: httpx.Client,
        texts: Union[str, List[str], List[Dict[str, str]]],
        *,
        model: Optional[str] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Request:
        model = model or "text-embedding-3-large"
        inference_kwargs = inference_kwargs or {}

        body = {
            "input": texts,
            "model": model,
            **inference_kwargs,
        }

        # Create the headers for the request
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["api-key"] = self.api_key  # For Azure OpenAI API compatibility
            if self.organization:
                headers["OpenAI-Organization"] = self.organization

        # Build the request
        request = client.build_request(
            "POST",
            url=self.inference_uri,
            json=body,
            headers=headers,
            **kwargs,
        )
        return request

    def parse_response(self, response: httpx.Response) -> List[List[float]]:
        response_body = response.json()
        return [entry.get("embedding", []) for entry in response_body]
