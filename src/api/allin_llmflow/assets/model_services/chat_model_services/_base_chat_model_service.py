"""
This module defines the base class for chat model services. A chat model service is a deployed model (or a selection of
models) that can be used for generating replies to chat message inputs.
"""

import abc
import asyncio
from typing import Any, Dict, List, Optional, Callable

import httpx

from allin_llmflow.assets.model_services._base_model_service import _ModelService
from allin_llmflow.dataclasses import ChatMessage, StreamingChunk, Tool


class _ChatModelService(_ModelService, metaclass=abc.ABCMeta):
    """
    An abstract base class of a generic service for managing and interacting with chat models.
    """

    @abc.abstractmethod
    def infer(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str],
        timeout: Optional[float],
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
        tools: Optional[List[Tool]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[ChatMessage]:
        """
        Infer responses to the given messages using the model service.

        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param stream: Whether to stream the response.
        :param streaming_callbacks: A list of callback functions to call for each StreamingChunk.
        :param tools: A list of tools to include in the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        raise NotImplementedError("infer method must be implemented")

    async def ainfer(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str],
        timeout: Optional[float],
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
        tools: Optional[List[Tool]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[ChatMessage]:
        """
        Asynchronously infer responses to the given messages using the model service.

        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param stream: Whether to stream the response.
        :param streaming_callbacks: A list of callback functions to call for each StreamingChunk.
        :param tools: A list of tools to include in the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        # By default, we run the synchronous method in a separate thread
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.infer(
                messages,
                model=model,
                timeout=timeout,
                stream=stream,
                streaming_callbacks=streaming_callbacks,
                tools=tools,
                inference_kwargs=inference_kwargs,
            ),
        )


class _HttpxChatModelService(_ChatModelService, metaclass=abc.ABCMeta):
    """
    An abstract base class of a generic service for managing and interacting with chat models using HTTPX.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        uri: Optional[str] = None,
        organization: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, api_key=api_key, uri=uri, organization=organization)
        if client_kwargs:
            self.configs["client_kwargs"] = client_kwargs

    @property
    def client_kwargs(self) -> Dict[str, Any]:
        """
        Get the client keyword arguments for the HTTPX client.

        :return: The client keyword arguments.
        """
        return self.configs.get("client_kwargs", {})

    @abc.abstractmethod
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
        Build a request according to the OpenAI Chat Completion API format.

        :param client: The httpx client to use for the request.
        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param stream: Whether to stream the response.
        :param tools: A list of tools to include in the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :return: The httpx request object.
        """
        raise NotImplementedError("build_request method must be implemented")

    @abc.abstractmethod
    def parse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
    ) -> List[ChatMessage]:
        """
        Parse the response from the model service to a list of ChatMessage instances.

        :param response: The response from the model service.
        :param stream: Whether the response is streamed.
        :param streaming_callbacks: A list of callback functions to call for each StreamingChunk
            if the response is streamed.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        raise NotImplementedError("parse_response method must be implemented")

    async def aparse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
    ) -> List[ChatMessage]:
        """
        Asynchronously parse the response from the model service to a list of ChatMessage instances.

        Default implementation runs the synchronous parse_response in a thread.
        Subclasses can override this with a truly asynchronous implementation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.parse_response(
                response,
                stream=stream,
                streaming_callbacks=streaming_callbacks,
            ),
        )

    def infer(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str],
        timeout: Optional[float],
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
        tools: Optional[List[Tool]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[ChatMessage]:
        """
        Infer responses to the given messages using the model service.

        :param messages: The list of ChatMessage instances to send to the model.
        :param timeout: The timeout for the request.
        :param stream: Whether to stream the response.
        :param streaming_callbacks: A list of callback functions to call for each StreamingChunk.
        :param model: The model to use for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param tools: A list of tools to include in the request.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        with httpx.Client(timeout=timeout, **self.client_kwargs) as client:
            request = self.build_request(
                client,
                messages,
                model=model,
                stream=stream,
                tools=tools,
                inference_kwargs=inference_kwargs,
            )
            try:
                response = client.send(request, stream=stream).raise_for_status()
            except httpx.HTTPStatusError as e:
                if stream:
                    response_text = e.response.read().decode("utf-8")
                else:
                    response_text = e.response.text
                raise httpx.HTTPStatusError(
                    str(e) + f"\nModel Service: {self.name}\nDetails: {response_text}",
                    request=e.request,
                    response=e.response,
                )
            chat_messages = self.parse_response(response, stream=stream, streaming_callbacks=streaming_callbacks)
        return chat_messages

    async def ainfer(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str],
        timeout: Optional[float],
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
        tools: Optional[List[Tool]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[ChatMessage]:
        """
        Asynchronously infer responses to the given messages using the model service.

        :param messages: The list of ChatMessage instances to send to the model.
        :param timeout: The timeout for the request.
        :param stream: Whether to stream the response.
        :param streaming_callbacks: A list of callback functions to call for each StreamingChunk.
        :param model: The model to use for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param tools: A list of tools to include in the request.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        async with httpx.AsyncClient(timeout=timeout, **self.client_kwargs) as client:
            request = self.build_request(
                client,
                messages,
                model=model,
                stream=stream,
                tools=tools,
                inference_kwargs=inference_kwargs,
            )
            try:
                response = await client.send(request, stream=stream)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if stream:
                    response_bytes = await e.response.aread()
                    response_text = response_bytes.decode("utf-8")
                else:
                    response_text = e.response.text
                raise httpx.HTTPStatusError(
                    str(e) + f"\nModel Service: {self.name}\nDetails: {response_text}",
                    request=e.request,
                    response=e.response,
                )
            chat_messages = await self.aparse_response(response, stream=stream, streaming_callbacks=streaming_callbacks)
        return chat_messages
