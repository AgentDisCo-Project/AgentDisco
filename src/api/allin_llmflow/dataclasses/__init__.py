"""
The 'dataclasses' module is a package that contains all the dataclasses used in the SDK.
"""

from allin_llmflow.dataclasses.asset_reference import AssetReference
from allin_llmflow.dataclasses.byte_stream import ByteStream
from allin_llmflow.dataclasses.chat_message import ChatMessage, ChatRole, ToolCall, ToolCallResult
from allin_llmflow.dataclasses.embedding_fields import EmbeddingField
from allin_llmflow.dataclasses.streaming_chunk import StreamingChunk
from allin_llmflow.dataclasses.tool import Tool

__all__ = [
    "AssetReference",
    "ByteStream",
    "ChatMessage",
    "ChatRole",
    "EmbeddingField",
    "StreamingChunk",
    "Tool",
    "ToolCall",
    "ToolCallResult",
]
