"""
This module defines the OpenAIChatModelService class, which is a ModelService asset that refers to a deployed OpenAI
model, or any deployed model supporting the OpenAI Chat Completion API format.
"""

import json
import logging
from base64 import b64encode
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Iterable

import httpx

from allin_llmflow.assets.model_services.chat_model_services._base_chat_model_service import _HttpxChatModelService
from allin_llmflow.dataclasses import ChatMessage, ChatRole, Tool, StreamingChunk
from allin_llmflow.dataclasses.chat_message import TextContent, MediaContent, ToolCall
from allin_llmflow.utils.streaming import ServerSentEvent, ServerSentEventStream, AsyncServerSentEventStream

logger = logging.getLogger(__name__)


def _convert_message_to_openai_format(message: ChatMessage) -> Dict[str, str]:
    """
    Convert a message to the format expected by OpenAI's Chat API.

    See the [API reference](https://platform.openai.com/docs/api-reference/chat/create) for details.

    :returns: A dictionary with the following key:
        - `role`
        - `content`
        - `name` (optional)
    """

    openai_msg: Dict[str, Any] = {"role": message.role.value}

    if len(message) == 1 and isinstance(message.content[0], TextContent):
        openai_msg["content"] = message.content[0].text
    elif message.tool_call_result:
        # Tool call results should only be included for ChatRole.TOOL messages and should not include any other content
        if message.role != ChatRole.TOOL:
            raise ValueError("Tool call results should only be included for tool messages.")
        if len(message) > 1:
            raise ValueError("Tool call results should not be included with other content.")
        if message.tool_call_result.origin.id is None:
            raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI.")
        openai_msg["content"] = message.tool_call_result.result
        openai_msg["tool_call_id"] = message.tool_call_result.origin.id
    else:
        openai_msg["content"] = []
        for item in message.content:
            if isinstance(item, TextContent):
                openai_msg["content"].append({"type": "text", "text": item.text})
            elif isinstance(item, MediaContent):
                match item.media.type:
                    case "image":
                        base64_data = b64encode(item.media.data).decode("utf-8")
                        url = f"data:{item.media.mime_type};base64,{base64_data}"
                        openai_msg["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": url,
                                    "detail": item.media.meta.get("detail", "auto"),
                                },
                            }
                        )
                    case _:
                        raise ValueError(f"Unsupported media type '{item.media.mime_type}' for OpenAI completions.")
            elif isinstance(item, ToolCall):
                if message.role != ChatRole.ASSISTANT:
                    raise ValueError("Tool calls should only be included for assistant messages.")
                if item.id is None:
                    raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI.")
                openai_msg.setdefault("tool_calls", []).append(
                    {
                        "id": item.id,
                        "type": "function",
                        "function": {
                            "name": item.tool_name,
                            "arguments": json.dumps(item.arguments, ensure_ascii=False),
                        },
                    }
                )
            else:
                raise ValueError(f"Unsupported content type '{type(item).__name__}' for OpenAI completions.")

    if message.name:
        openai_msg["name"] = message.name

    return openai_msg


class OpenAIChatModelService(_HttpxChatModelService):
    """
    A ModelService asset that refers to a deployed OpenAI model, or a deployed model supporting OpenAI Chat Completion
    API format.
    """

    INFERENCE_API_FORMAT = "openai-chat"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        uri: Optional[str] = None,
        organization: Optional[str] = None,
        support_stream_options: bool = True,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, api_key=api_key, uri=uri, organization=organization, client_kwargs=client_kwargs)
        self.configs["support_stream_options"] = self.support_stream_options = support_stream_options

    @staticmethod
    def _build_message(
        choice: Dict[str, Any],
        model: str,
        usage: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """
        Converts the non-streaming response from the OpenAI API to a ChatMessage.

        :param choice: A single choice from the OpenAI API response.
        :param model: The model used for the request.
        :param usage: The usage information for the request.
        :return: The ChatMessage.
        """
        message = choice.get("message", {})
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        tool_calls = (
            OpenAIChatModelService._build_tool_calls(message.get("tool_calls")) if message.get("tool_calls") else None
        )

        chat_message = ChatMessage.from_assistant(content, reasoning_content=reasoning_content, tool_calls=tool_calls)
        chat_message.meta.update(
            {
                "model": model,
                "index": choice.get("index"),
                "finish_reason": choice.get("finish_reason"),
                "usage": dict(usage or {}),
            }
        )
        return chat_message

    @staticmethod
    def _build_tool_calls(raw_tool_calls: Iterable[Dict[str, Any]]) -> List[ToolCall]:
        """
        Build a list of ToolCall instances from the raw tool calls returned by the OpenAI API.

        :param raw_tool_calls: The raw tool calls returned by the OpenAI API.
        :return: A list of ToolCall instances.
        :raises ValueError: If the tool calls are not in the expected format.
        """
        tool_calls = []
        for raw_tc in raw_tool_calls:
            try:
                arguments_str = raw_tc["function"]["arguments"]
                tool_call_id = raw_tc["id"]
                tool_name = raw_tc["function"]["name"]
            except KeyError as e:
                raise ValueError(f"Invalid tool call format: {raw_tc}") from e
            try:
                arguments = json.loads(arguments_str)
                tool_calls.append(
                    ToolCall(
                        id=tool_call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                )
            except json.JSONDecodeError:
                logger.warning(
                    "OpenAI returned a malformed JSON string for tool call arguments. This tool call "
                    "will be skipped. To always generate a valid JSON, set `tools_strict` to `True`. "
                    "Tool call ID: %s, Tool name: %s, Arguments: %s",
                    tool_call_id,
                    tool_name,
                    arguments_str,
                )
        return tool_calls

    @staticmethod
    def _build_chunk(event: ServerSentEvent) -> StreamingChunk:
        """
        Build a StreamingChunk from a ServerSentEvent, following the OpenAI API format.

        :param event: The ServerSentEvent.
        :return: The StreamingChunk.
        """
        data = event.json()
        try:
            choices = data["choices"]
        except KeyError as err:
            raise ValueError(f"Could not find 'choices' in the response data: {data}") from err

        choice = choices[0] if len(choices) > 0 else {}  # to handle the usage stats chunk where choices are empty
        content = choice.get("delta", {}).get("content") or ""
        reasoning_content = choice.get("delta", {}).get("reasoning_content")
        tool_calls = choice.get("delta", {}).get("tool_calls")
        meta = {
            "model": data.get("model"),
            "index": choice.get("index"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
        }
        return StreamingChunk(content=content, reasoning_content=reasoning_content, tool_calls=tool_calls, meta=meta)

    def _connect_chunks(self, chunks: List[StreamingChunk]) -> ChatMessage:
        """
        Connect the streaming chunks to a single ChatMessage.

        :param chunks: The list of StreamingChunks.
        :return: The ChatMessage.
        """
        if len(chunks) == 0:
            raise ValueError(f"The model service '{self.name}' returned an empty stream.")
        total_content = ""
        total_reasoning_content = None
        total_meta = {}
        content_chunks = (
            chunks[:-1] if self.support_stream_options else chunks
        )  # We exclude the last chunk as it only includes usage stats
        raw_tool_calls: Dict[str, Dict[str, Any]] = {}
        for chunk in content_chunks:
            if chunk.tool_calls is not None:
                for raw_tc in chunk.tool_calls:
                    if raw_tc["index"] not in raw_tool_calls:
                        raw_tool_calls[raw_tc["index"]] = raw_tc
                    elif delta_args := raw_tc.get("function", {}).get("arguments", ""):
                        # Append the arguments from the delta to the existing tool call
                        arguments = raw_tool_calls[raw_tc["index"]].setdefault("function", {}).get("arguments", "")
                        arguments += delta_args
                        raw_tool_calls[raw_tc["index"]]["function"]["arguments"] = arguments
            if chunk.reasoning_content:
                if total_reasoning_content is None:
                    total_reasoning_content = chunk.reasoning_content
                else:
                    total_reasoning_content += chunk.reasoning_content
            total_content += chunk.content
            total_meta.update(chunk.meta)
        tool_calls = (
            OpenAIChatModelService._build_tool_calls(raw_tool_calls.values()) if len(raw_tool_calls) > 0 else None
        )
        chat_message = ChatMessage.from_assistant(
            total_content, reasoning_content=total_reasoning_content, tool_calls=tool_calls, meta=total_meta
        )
        if self.support_stream_options:
            chat_message.meta["usage"] = chunks[-1].meta.get("usage", {})

        return chat_message

    @staticmethod
    def _check_finish_reason(message: ChatMessage) -> None:
        """
        Check the `finish_reason` returned with the OpenAI completions. This handles all edge cases described in
        [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling#edge-cases).

        If the `finish_reason` is `length` or `content_filter`, log a warning.
        :param message: The message returned by the LLM.
        """
        finish_reason = message.meta.get("finish_reason")
        match finish_reason:
            case "length":
                logger.warning(
                    "The completion for index %s has been truncated before reaching a natural stopping point. "
                    "Increase the max_tokens parameter to allow for longer completions.",
                    message.meta.get("index", 0),
                )
            case "content_filter":
                logger.warning(
                    "The completion for index %s has been truncated due to the content filter.",
                    message.meta.get("index", 0),
                )
            case "tool_calls":
                if message.tool_call is None:
                    logger.warning(
                        "The completion for index %s claimed to have tool calls, but no valid tool call was found in "
                        "the response. This could be due to invalid tool call formatting or model errors.",
                        message.meta.get("index", 0),
                    )
            case _:
                pass

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
        :param kwargs: Additional keyword arguments to pass to the httpx client.
        :return: The httpx request object.
        """
        model = model or ""
        inference_kwargs = inference_kwargs or {}
        # Convert ChatMessage instances to the format expected by the OpenAI API
        openai_formatted_messages = [_convert_message_to_openai_format(message) for message in messages]

        # Create the body of the request
        body = {
            "model": model,
            "messages": openai_formatted_messages,
            "stream": stream,
            **inference_kwargs,
        }

        # Automatically include usage data tracking if streaming is enabled and stream_options are supported
        if stream and self.support_stream_options:
            if "stream_options" not in body:
                body["stream_options"] = {}
            body["stream_options"]["include_usage"] = True

        # Create the headers for the request
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["api-key"] = self.api_key  # For Azure OpenAI API compatibility
            if self.organization:
                headers["OpenAI-Organization"] = self.organization

        # Include tool definitions if provided
        if tools:
            body["tools"] = [{"type": "function", "function": {**t.tool_spec}} for t in tools]

        # Build the request
        request = client.build_request(
            "POST",
            url=self.inference_uri,
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
        Parse the response from the OpenAI Chat Completion API to a list of ChatMessage instances. If streaming is
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
            for sse in ServerSentEventStream(response=response):
                chunk = self._build_chunk(sse)
                if _first_token:
                    _first_token = False
                    chunk.meta["completion_start_time"] = datetime.now().isoformat()
                for callback in streaming_callbacks:
                    callback(chunk)
                chunks.append(chunk)
            chat_messages = [self._connect_chunks(chunks)]
        else:
            try:
                response_json = response.json()
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"The model service '{self.name}' returned an invalid JSON response: {response.text}"
                ) from err
            chat_messages = [
                self._build_message(
                    choice,
                    model=response_json.get("model", ""),
                    usage=response_json.get("usage"),
                )
                for choice in response_json.get("choices", [])
            ]
        for message in chat_messages:
            self._check_finish_reason(message)
        return chat_messages

    async def aparse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
    ) -> List[ChatMessage]:
        """
        Asynchronously parse the response from the OpenAI Chat Completion API to a list of ChatMessage instances. If
        streaming is enabled, the response is parsed as a stream of ServerSentEvents, and each event is captured as a
        StreamingChunk. Callback functions can be provided to process each StreamingChunk as soon as it is received.

        :param response: The response from the model service.
        :param stream: Whether the response is streamed.
        :param streaming_callbacks: A list of callback functions to call for each StreamingChunk when streaming.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        if stream:
            chunks: List[StreamingChunk] = []
            _first_token = True  # For tracking first token completion time
            streaming_callbacks = streaming_callbacks or []
            async for sse in AsyncServerSentEventStream(response=response):
                chunk = self._build_chunk(sse)
                if _first_token:
                    _first_token = False
                    chunk.meta["completion_start_time"] = datetime.now().isoformat()
                for callback in streaming_callbacks:
                    callback(chunk)
                chunks.append(chunk)
            chat_messages = [self._connect_chunks(chunks)]
        else:
            try:
                response_json = response.json()
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"The model service '{self.name}' returned an invalid JSON response: {response.text}"
                ) from err
            chat_messages = [
                self._build_message(
                    choice,
                    model=response_json.get("model", ""),
                    usage=response_json.get("usage"),
                )
                for choice in response_json.get("choices", [])
            ]
        for message in chat_messages:
            self._check_finish_reason(message)
        return chat_messages

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
        Infer responses to the given messages using the openai model service.

        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param stream: Whether to stream the response.
        :param streaming_callbacks: A list of callback functions to call for each StreamingChunk.
        :param tools: A list of Tool to use for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        inference_kwargs = inference_kwargs or {}
        if stream and inference_kwargs.get("n", 1) > 1:
            raise ValueError("Streaming with multiple completions is not yet supported for OpenAI models.")
        return super().infer(
            messages,
            model=model,
            timeout=timeout,
            stream=stream,
            streaming_callbacks=streaming_callbacks,
            tools=tools,
            inference_kwargs=inference_kwargs,
        )
