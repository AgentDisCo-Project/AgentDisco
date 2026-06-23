"""
This module defines the AnthropicChatModelService class for handling the Anthropic chat model service or any model
supporting the Anthropic Chat API format.
"""

import json
import logging
from base64 import b64encode
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable

import httpx

from allin_llmflow.assets.model_services.chat_model_services._base_chat_model_service import _HttpxChatModelService
from allin_llmflow.dataclasses import ChatMessage, ChatRole, StreamingChunk, Tool
from allin_llmflow.dataclasses.chat_message import TextContent, MediaContent, ToolCall, ToolCallResult
from allin_llmflow.utils.streaming import ServerSentEvent, ServerSentEventStream, AsyncServerSentEventStream

logger = logging.getLogger(__name__)

REDACTED_REASONING_CONTENT = "\a[REDACTED]"


class AnthropicChatModelService(_HttpxChatModelService):
    """
    A ModelService asset that refers to a deployed OpenAI model, or a deployed model supporting OpenAI Chat Completion
    API format.
    """

    INFERENCE_API_FORMAT = "anthropic-chat"

    @staticmethod
    def default_max_tokens(model_name: str) -> int:
        """
        Get the default maximum tokens for the given model, This is used when the `max_tokens` parameter is not provided
        in the request. Source: https://docs.anthropic.com/en/docs/about-claude/models

        :param model_name: The name of the model.
        :return: The default maximum tokens.
        """
        if any(model in model_name for model in ("claude-3-opus", "claude-3-sonnet", "claude-3-haiku")):
            return 4096
        return 8192

    @staticmethod
    def format_chat_message(message: ChatMessage) -> Dict[str, Any]:
        """
        Convert a ChatMessage to the Anthropic API format.

        :param message: The ChatMessage to convert.
        :return: The Anthropic API format of the ChatMessage.
        """
        anthropic_msg: Dict[str, Any] = {}
        if "thinking_blocks" not in message.meta and message.reasoning_content is not None:
            logger.warning(
                "Reasoning content found without structured thinking blocks with signature. This will be omitted "
                "from the Anthropic completion for safety reasons."
            )
        if message.role not in (ChatRole.USER, ChatRole.ASSISTANT):
            if message.role == ChatRole.TOOL and message.tool_call_result is not None:
                logger.debug("Assuming user role for tool call results as per Anthropic guidelines.")
            else:
                logger.warning("The role %s is not supported by the Anthropic API.", message.role)
            anthropic_msg["role"] = ChatRole.USER.value
        else:
            anthropic_msg["role"] = message.role.value
        if len(message) == 1 and isinstance(message.content[0], TextContent) and "thinking_blocks" not in message.meta:
            anthropic_msg["content"] = message.content[0].text
        else:
            anthropic_msg["content"] = message.meta.get("thinking_blocks", [])
            for item in message.content:
                if isinstance(item, TextContent):
                    anthropic_msg["content"].append({"type": "text", "text": item.text})
                elif isinstance(item, MediaContent):
                    match item.media.type:
                        case "image":
                            anthropic_msg["content"].append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": item.media.mime_type,
                                        "data": b64encode(item.media.data).decode("utf-8"),
                                    },
                                }
                            )
                        case "application":
                            if item.media.subtype != "pdf":
                                raise ValueError(
                                    f"Unsupported media type '{item.media.mime_type}' for Anthropic completions."
                                )
                            anthropic_msg["content"].append(
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": item.media.mime_type,
                                        "data": b64encode(item.media.data).decode("utf-8"),
                                    },
                                }
                            )
                        case _:
                            raise ValueError(
                                f"Unsupported media type '{item.media.mime_type}' for Anthropic completions."
                            )
                elif isinstance(item, ToolCall):
                    if item.id is None:
                        raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with Anthropic.")
                    anthropic_msg["content"].append(
                        {
                            "type": "tool_use",
                            "id": item.id,
                            "name": item.tool_name,
                            "input": item.arguments,
                        }
                    )
                elif isinstance(item, ToolCallResult):
                    if item.origin.id is None:
                        raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with Anthropic.")
                    anthropic_msg["content"].append(
                        {
                            "type": "tool_result",
                            "tool_use_id": item.origin.id,
                            "is_error": item.error,
                            "content": item.result,
                        }
                    )
                else:
                    raise ValueError(f"Unsupported content type '{type(item).__name__}' for Anthropic completions.")
        return anthropic_msg

    @staticmethod
    def _build_message(
        content: List[Dict[str, Any]],
        model: str,
        msg_id: Optional[str] = None,
        stop_reason: Optional[str] = None,
        stop_sequence: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """
        Converts the non-streaming response from the Anthropic API to a ChatMessage.

        :param content: The content of the response.
        :param model: The model used for the request.
        :param msg_id: The message ID.
        :param stop_reason: The reason the model stopped generating completions.
        :param stop_sequence: Which custom stop sequence was generated in the completion, if any.
        :param usage: The usage information for the request.
        :return: The ChatMessage.
        """
        text = ""
        reasoning_content = None
        thinking_blocks: List[Dict[str, Any]] = []
        tool_calls: List[ToolCall] = []

        for block in content:
            if block["type"] == "tool_use":
                tool_call = AnthropicChatModelService._build_tool_call(block)
                if tool_call:
                    tool_calls.append(tool_call)
            elif block["type"] == "text":
                # Normally there should be only one text block from Anthropic API
                text += block["text"]
            elif block["type"] == "thinking":
                if reasoning_content is None:
                    reasoning_content = ""
                reasoning_content += block["thinking"]
                thinking_blocks.append(block)
            elif block["type"] == "redacted_thinking":
                if reasoning_content is None:
                    reasoning_content = ""
                reasoning_content += REDACTED_REASONING_CONTENT
                thinking_blocks.append(block)

        chat_message = ChatMessage.from_assistant(
            text,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            meta={
                "id": msg_id,
                "model": model,
                "stop_reason": stop_reason,
                "stop_sequence": stop_sequence,
                "usage": dict(usage or {}),
            },
        )
        if reasoning_content:
            chat_message.meta["thinking_blocks"] = thinking_blocks
        return chat_message

    @staticmethod
    def _build_tool_call(raw_tool_call: Dict[str, Any], stream: bool = False) -> Optional[ToolCall]:
        """
        Build a ToolCall instance from a raw tool call returned by the Anthropic API.

        :param raw_tool_call: The raw tool call data.
        :param stream: Whether the tool call is part of a streaming response.
        :return: The ToolCall instance, or None if the tool call arguments cannot be parsed.
        :raises ValueError: If the tool call is invalid.
        """
        try:
            tool_call_id = raw_tool_call["id"]
            tool_name = raw_tool_call["name"]
            if stream:
                arguments_str = raw_tool_call.get("raw_input", "")
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    logger.warning(
                        "Anthropic returned a malformed JSON string for tool call arguments as a part of its streaming"
                        " response. This tool call will be skipped. Tool call ID: %s, Tool name: %s, Arguments: %s",
                        tool_call_id,
                        tool_name,
                        arguments_str,
                    )
                    return None
            else:
                arguments = raw_tool_call["input"]
        except KeyError as e:
            raise ValueError(f"Invalid tool call format: {raw_tool_call}") from e

        return ToolCall(
            id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
        )

    @staticmethod
    def _build_chunk(sse: ServerSentEvent) -> StreamingChunk | None:
        """
        Build a StreamingChunk from a ServerSentEvent, following the Anthropic API format.

        :param sse: The ServerSentEvent.
        :return: The StreamingChunk.
        """
        data = sse.json()
        return AnthropicChatModelService._build_chunk_from_data(data=data, event=sse.event)

    # pylint: disable=too-many-return-statements
    @staticmethod
    def _build_chunk_from_data(data: Dict[str, Any], event: Optional[str] = None) -> Optional[StreamingChunk]:
        """
        Build a StreamingChunk from the data returned by the model service. It follows the Anthropic API specification
        in https://docs.anthropic.com/en/api/messages-streaming.

        :param data: The data returned by the model service.
        :param event: The event type of the data. If not specified, it will be extracted from the data.
        :return: The StreamingChunk, or None if the data does not correspond to a StreamingChunk.
        """
        if event is None:
            try:
                event = data["type"]
            except KeyError as err:
                raise ValueError(f"Invalid data received during model inference streaming: {data}") from err

        if event == "message_start":
            message = data.get("message", {})
            return StreamingChunk(
                "",
                meta={
                    "id": message.get("id"),
                    "model": message.get("model"),
                    "stop_reason": message.get("stop_reason"),
                    "stop_sequence": message.get("stop_sequence"),
                    "usage": message.get("usage"),
                },
            )
        if event == "content_block_start":
            content_block = data.get("content_block", {})
            match content_block.get("type"):
                case "text":
                    return StreamingChunk(content_block.get("text", ""), meta={"index": data.get("index")})
                case "thinking":
                    return StreamingChunk("", reasoning_content="", meta={"index": data.get("index")})
                case "redacted_thinking":
                    return StreamingChunk(
                        "",
                        reasoning_content=REDACTED_REASONING_CONTENT,
                        meta={"index": data.get("index"), "redacted_thinking_block": content_block},
                    )
                case "tool_use":
                    return StreamingChunk("", tool_calls=[content_block], meta={"index": data.get("index")})
                case _:
                    raise ValueError(f"Invalid content block type: {content_block.get('type')}")
        if event == "content_block_delta":
            delta = data.get("delta", {})
            match delta.get("type"):
                case "text_delta":
                    return StreamingChunk(delta.get("text", ""), meta={"index": data.get("index")})
                case "thinking_delta":
                    return StreamingChunk(
                        "", reasoning_content=delta.get("thinking", ""), meta={"index": data.get("index")}
                    )
                case "input_json_delta":
                    return StreamingChunk(
                        "",
                        tool_calls=[{"partial_json": delta.get("partial_json", "")}],
                        meta={"index": data.get("index")},
                    )
                case "signature_delta":
                    return StreamingChunk(
                        "",
                        reasoning_content="",
                        meta={"index": data.get("index"), "signature": delta.get("signature", "")},
                    )
                case _:
                    raise ValueError(f"Invalid content block delta type: {delta.get('type')}")
        if event == "content_block_stop":
            return StreamingChunk("", meta={"index": data.get("index")})
        if event == "message_delta":
            delta = data.get("delta", {})
            return StreamingChunk(
                "",
                meta={
                    "stop_reason": delta.get("stop_reason"),
                    "stop_sequence": delta.get("stop_sequence"),
                    "usage": data.get("usage"),
                },
            )
        if event == "error":
            raise RuntimeError(f"Error received during model inference streaming: {data.get('error')}")
        if event in ("ping", "message_stop"):
            return None
        # Gracefully handle unknown event types as warnings
        logger.warning("Unknown event type: %s. Skipping the event with data: %s", event, data)
        return None

    def _connect_chunks(self, chunks: List[StreamingChunk]) -> ChatMessage:
        """
        Connect the streaming chunks to a single ChatMessage.

        :param chunks: The list of StreamingChunks.
        :return: The ChatMessage.
        """
        content = ""
        reasoning_content = None
        thinking_blocks: Dict[str, Dict[str, Any]] = {}
        raw_tool_calls: Dict[str, Dict[str, Any]] = {}
        meta: Dict[str, Any] = {}
        for chunk in chunks:
            if chunk.tool_calls is not None:
                if len(chunk.tool_calls) != 1:
                    raise ValueError("Anthropic tool call chunks should contain only one tool call.")
                raw_tc = chunk.tool_calls[0]
                if chunk.meta["index"] not in raw_tool_calls:
                    raw_tool_calls[chunk.meta["index"]] = raw_tc
                elif "partial_json" in raw_tc:
                    # Append the partial JSON to the existing tool call
                    raw_input = raw_tool_calls[chunk.meta["index"]].get("raw_input", "")
                    raw_input += raw_tc["partial_json"]
                    raw_tool_calls[chunk.meta["index"]]["raw_input"] = raw_input
            if chunk.reasoning_content is not None:
                # Append the reasoning content to the existing reasoning content
                if reasoning_content is None:
                    reasoning_content = ""
                reasoning_content += chunk.reasoning_content

                # Preserve thinking data for future reconstruction
                if "redacted_thinking_block" in chunk.meta:
                    thinking_blocks[chunk.meta["index"]] = chunk.meta["redacted_thinking_block"]
                    chunk.meta.pop("redacted_thinking_block")
                else:
                    if chunk.meta["index"] not in thinking_blocks:
                        thinking_blocks[chunk.meta["index"]] = {
                            "type": "thinking",
                            "thinking": chunk.reasoning_content,
                        }
                    else:
                        thinking_blocks[chunk.meta["index"]]["thinking"] += chunk.reasoning_content
                    if "signature" in chunk.meta:
                        thinking_blocks[chunk.meta["index"]]["signature"] = chunk.meta["signature"]
                        chunk.meta.pop("signature")

            content += chunk.content
            # We need to merge the usage information from all the chunks
            meta.update({**chunk.meta, "usage": {**meta.get("usage", {}), **chunk.meta.get("usage", {})}})

        # Build the tool calls
        tool_calls: List[ToolCall] = []
        for raw_tc in raw_tool_calls.values():
            tc = AnthropicChatModelService._build_tool_call(raw_tc, stream=True)
            if tc:
                tool_calls.append(tc)

        # Add thinking data to the meta
        if reasoning_content:
            meta["thinking_blocks"] = list(thinking_blocks.values())

        return ChatMessage.from_assistant(
            content, reasoning_content=reasoning_content, tool_calls=tool_calls, meta=meta
        )

    @staticmethod
    def _check_stop_reason(message: ChatMessage) -> None:
        """
        Check the `stop_reason` returned with the Anthropic completions.

        If the `stop_reason` is `max_tokens`, log a warning.
        :param message: The message returned by the LLM.
        """
        stop_reason = message.meta.get("stop_reason")
        match stop_reason:
            case "max_tokens":
                logger.warning(
                    "The assistant completion %s has been truncated before reaching a natural stopping point. "
                    "Increase the max_tokens parameter or pick another model to allow for longer completions.",
                    message.meta.get("id"),
                )
            case "stop_sequence":
                logger.warning(
                    "The assistant completion %s stopped due to a custom stop sequence: '%s'",
                    message.meta.get("id"),
                    message.meta.get("stop_sequence"),
                )
            case "tool_use":
                if message.tool_call is None:
                    logger.warning(
                        "The assistant completion %s stopped due to a tool call, but no valid tool call was found in "
                        "the response. This could be due to invalid tool call formatting or model errors.",
                        message.meta.get("id"),
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
        Build a request according to the Anthropic Chat Completion API format.

        :param client: The httpx client to use for the request.
        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param stream: Whether to stream the response.
        :param tools: A list of tools to include in the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param kwargs: Additional keyword arguments to pass to the httpx client.
        :return: The httpx request object.
        """
        model = model or "claude-3-5-sonnet"
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
            max_tokens = self.default_max_tokens(model)
            logger.info(
                "max_tokens is required for Anthropic Chat API. Using default value inferred from model: %s",
                max_tokens,
            )

        # Create the body of the request
        body = {
            "model": model,
            "messages": anthropic_formatted_messages,
            "max_tokens": max_tokens,
            "stream": stream,
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
            headers["x-api-key"] = self.api_key

        # Include tool definitions if provided
        if tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]

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
            for sse in ServerSentEventStream(response=response):
                chunk = self._build_chunk(sse)
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
        Asynchronously parse the response from the Anthropic Chat Completion API to a list of ChatMessage instances. If
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
