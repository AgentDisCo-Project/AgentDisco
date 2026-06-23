"""
This module provides a collection of classes that represent and manage websearch services in the application.

A websearch service is a deployed tool (or a selection of tools) that can be used for performing web searches.

Web search services include:

- WebSearchService:
    An abstract base class of a generic service for managing and interacting with websearch tools.
- HttpxWebSearchService:
    An abstract base class of a generic service for managing and interacting with websearch tools using HTTPX.
- BochaWebSearchService:
    A service specifically designed for managing and interacting with Bocha websearch tools.
"""

from allin_llmflow.assets.tool_services.websearch_services._base_websearch_service import (
    _WebSearchService as WebSearchService,
    _HttpxWebSearchService as HttpxWebSearchService,
)
from allin_llmflow.assets.tool_services.websearch_services.bocha import BochaWebSearchService
from allin_llmflow.assets.tool_services.websearch_services.rednote import RednoteWebSearchService

__all__ = [
    "WebSearchService",
    "HttpxWebSearchService",
    "BochaWebSearchService",
    "RednoteWebSearchService",
]
