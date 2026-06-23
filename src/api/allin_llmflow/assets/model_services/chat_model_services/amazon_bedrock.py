"""
This module provides a collection of classes for handling the chat model services hosted on Amazon Bedrock. Currently,
only the Anthropic chat model service is supported.
"""

import base64
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

import httpx

from allin_llmflow.assets.model_services.chat_model_services.anthropic import AnthropicChatModelService
from allin_llmflow.dataclasses import ChatMessage, ChatRole, StreamingChunk, Tool
from allin_llmflow.dataclasses.chat_message import TextContent

logger = logging.getLogger(__name__)


class BedrockAnthropicChatModelService(AnthropicChatModelService):
    """
    A ModelService asset that refers to an Anthropic chat model hosted on AWS Bedrock.

    :param name: The name of the model service.
    :param model: The model hosted on the model service, defaults to None.
    :param api_key: The API key to use for the model service, defaults to None.
    :param uri: The base URI of the model service. Bedrock Anthropic chat model services are expected to have an
        inference endpoint at {uri}/invoke and a streaming endpoint at {uri}/invoke-with-response-stream.
    :param mesh_uri: The mesh URI of the model service in prod environment, defaults to None.
    :param organization: The organization of the model service, defaults to None.
    """

    INFERENCE_API_FORMAT = "bedrock-anthropic-chat"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        uri: Optional[str] = None,
        organization: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, api_key=api_key, uri=uri, organization=organization, client_kwargs=client_kwargs)
        self.configs["model"] = model
        self.model = model or "claude-3-5-sonnet-latest"

        if not self.inference_uri.endswith("/"):
            self.inference_uri += "/"
        self.invoke_uri = self.inference_uri + "invoke"
        self.stream_uri = self.inference_uri + "invoke-with-response-stream"

    def _decode_data(self, stream_bytes: str | bytes, bedrock_content_type: str = "application/json") -> Dict[str, Any]:
        """
        Decode the data from the stream of bytes.

        :param stream_bytes: The stream of bytes.
        :param bedrock_content_type: The actual content type of the data.
        :returns: The decoded data.
        :raises NotImplementedError: If the content type is not supported.
        :raises RuntimeError: If the response format is invalid.
        """
        if bedrock_content_type != "application/json":
            raise NotImplementedError(f"Unsupported content type: {bedrock_content_type}")
        content = json.loads(stream_bytes)
        if encoded_data := content.get("chunk", {}).get("bytes"):
            return json.loads(base64.b64decode(encoded_data, validate=True))

        # Error handling if encoded_data is not present
        if msg := content.get("internalServerException", {}).get("message"):
            raise RuntimeError(f"Internal server exception from {self.stream_uri}: {msg}")
        if model_err_info := content.get("modelStreamErrorException"):
            raise RuntimeError(f"Model stream error from {self.stream_uri}: {model_err_info}")
        if msg := content.get("validationException", {}).get("message"):
            raise RuntimeError(f"Validation exception from {self.stream_uri}: {msg}")
        if msg := content.get("throttlingException", {}).get("message"):
            raise RuntimeError(f"Throttling exception from {self.stream_uri}: {msg}")
        if msg := content.get("modelTimeoutException", {}).get("message"):
            raise RuntimeError(f"Model timeout exception from {self.stream_uri}: {msg}")
        if msg := content.get("serviceUnavailableException", {}).get("message"):
            raise RuntimeError(f"Service unavailable exception from {self.stream_uri}: {msg}")
        raise RuntimeError(f"Invalid response format from {self.stream_uri}: {content}")

    def build_request(
        self,
        client: httpx.Client | httpx.AsyncClient,
        messages: List[ChatMessage],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> httpx.Request:
        """
        Build a request according to the Bedrock-Anthropic Chat Completion API format.

        :param client: The httpx client to use for the request.
        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param stream: Whether to stream the response.
        :param tools: A list of tools to include in the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param kwargs: Additional keyword arguments to pass to the httpx client.
        :return: The httpx request object.
        """
        inference_kwargs = inference_kwargs or {}

        # Extract the system prompt if it exists
        if messages[0].role == ChatRole.SYSTEM:
            if len(messages[0]) != 1 or not isinstance(messages[0].content[0], TextContent):
                raise ValueError("The system prompt must be a single text message.")
            system_prompt = messages[0].text
            messages = messages[1:]
        else:
            system_prompt = None

        # Format the rest of the messages
        anthropic_formatted_messages = [self.format_chat_message(message) for message in messages]

        max_tokens = inference_kwargs.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.default_max_tokens(self.model)
            logger.info(
                "max_tokens is required for AWS Bedrock Anthropic Chat API. "
                "Using default value inferred from model: %s",
                max_tokens,
            )

        # Create the body of the request
        body = {
            "messages": anthropic_formatted_messages,
            "max_tokens": max_tokens,
            "anthropic_version": "bedrock-2023-05-31",
            **inference_kwargs,
        }

        if system_prompt:
            body["system"] = system_prompt

        # Create the headers for the request
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["token"] = self.api_key

        # Include tool definitions if provided
        if tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]

        # Build the request
        request = client.build_request(
            "POST",
            url=self.stream_uri if stream else self.invoke_uri,
            json=body,
            headers=headers,
        )
        return request

    def parse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
    ) -> List[ChatMessage]:
        """
        Parse the response from the Anthropic Chat Completion API to a list of ChatMessage instances. If streaming is
        enabled, the response is parsed as a stream of ServerSentEvents, and each event is captured as a StreamingChunk.
        Callback functions can be provided to process each StreamingChunk as soon as it is received.

        :param response: The response from the model service.
        :param stream: Whether the response is streamed.
        :param streaming_callbacks: A list of callback functions to call for each StreamingChunk when streaming.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        if stream:
            chunks: List[StreamingChunk] = []
            _first_token = True  # For tracking first token completion time
            streaming_callbacks = streaming_callbacks or []
            for event_stream in response.iter_lines():
                data = self._decode_data(
                    event_stream,
                    bedrock_content_type=response.headers.get("x-amzn-bedrock-content-type", "application/json"),
                )
                chunk = self._build_chunk_from_data(data)
                if chunk is None:
                    continue
                if _first_token:
                    _first_token = False
                    chunk.meta["completion_start_time"] = datetime.now().isoformat()
                for callback in streaming_callbacks:
                    callback(chunk)
                chunks.append(chunk)
            chat_messages = [self._connect_chunks(chunks)]
        else:
            response_json = response.json()
            chat_messages = [
                self._build_message(
                    content=response_json.get("content", []),
                    model=response_json.get("model"),
                    msg_id=response_json.get("id"),
                    stop_reason=response_json.get("stop_reason"),
                    stop_sequence=response_json.get("stop_sequence"),
                    usage=response_json.get("usage"),
                )
            ]
        for message in chat_messages:
            self._check_stop_reason(message)
        return chat_messages

    async def aparse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
    ) -> List[ChatMessage]:
        """
        Asynchronously parse the response from the Bedrock Anthropic Chat Completion API to a list of ChatMessage
        instances. If streaming is enabled, the response is parsed as a stream of ServerSentEvents, and each event is
        captured as a StreamingChunk. Callback functions can be provided to process each StreamingChunk as soon as it
        is received.

        :param response: The response from the model service.
        :param stream: Whether the response is streamed.
        :param streaming_callbacks: A list of callback functions to call for each StreamingChunk when streaming.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        if stream:
            chunks: List[StreamingChunk] = []
            _first_token = True  # For tracking first token completion time
            streaming_callbacks = streaming_callbacks or []
            async for event_stream in response.aiter_lines():
                data = self._decode_data(
                    event_stream,
                    bedrock_content_type=response.headers.get("x-amzn-bedrock-content-type", "application/json"),
                )
                chunk = self._build_chunk_from_data(data)
                if chunk is None:
                    continue
                if _first_token:
                    _first_token = False
                    chunk.meta["completion_start_time"] = datetime.now().isoformat()
                for callback in streaming_callbacks:
                    callback(chunk)
                chunks.append(chunk)
            chat_messages = [self._connect_chunks(chunks)]
        else:
            response_json = response.json()
            chat_messages = [
                self._build_message(
                    content=response_json.get("content", []),
                    model=response_json.get("model"),
                    msg_id=response_json.get("id"),
                    stop_reason=response_json.get("stop_reason"),
                    stop_sequence=response_json.get("stop_sequence"),
                    usage=response_json.get("usage"),
                )
            ]
        for message in chat_messages:
            self._check_stop_reason(message)
        return chat_messages
