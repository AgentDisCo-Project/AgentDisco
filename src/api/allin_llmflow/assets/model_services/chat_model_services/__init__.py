"""
This module provides a collection of classes that represent and manage chat model services in the application.

A chat model service is a deployed model (or a selection of models) that can be used for generating replies to chat
message inputs.

Chat model services include:

- ChatModelService:
    An abstract base class of a generic service for managing and interacting with chat models.
- HttpxChatModelService:
    An abstract base class of a generic service for managing and interacting with chat models using HTTPX.
- OpenAIChatModelService:
    A service specifically designed for managing and interacting with OpenAI-API style chat models.
"""

from allin_llmflow.assets.model_services.chat_model_services._base_chat_model_service import (
    _ChatModelService as ChatModelService,
    _HttpxChatModelService as HttpxChatModelService,
)
from allin_llmflow.assets.model_services.chat_model_services.amazon_bedrock import BedrockAnthropicChatModelService
from allin_llmflow.assets.model_services.chat_model_services.anthropic import AnthropicChatModelService
from allin_llmflow.assets.model_services.chat_model_services.openai import OpenAIChatModelService

__all__ = [
    "ChatModelService",
    "HttpxChatModelService",
    "OpenAIChatModelService",
    "BedrockAnthropicChatModelService",
    "AnthropicChatModelService",
]
