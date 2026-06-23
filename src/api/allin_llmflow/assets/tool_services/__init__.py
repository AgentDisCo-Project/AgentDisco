"""
This module provides a collection of classes that represent and manage tool services in the application.

A tool service is a deployed tool (or a selection of tools) that can be used for various downstream tasks.
Tool services can be used to perform tasks such as web searches, data processing, and often provide an auxiliary
role in the application.
"""

from typing import Dict, Type

from allin_llmflow.assets.tool_services._base_tool_service import _ToolService as ToolService
from allin_llmflow.assets.tool_services.websearch_services import *  # noqa: F403
from allin_llmflow.assets.tool_services.webscraper_services import *  # noqa: F403

# Register all supported inference API formats
SUPPORTED_CALL_API_FORMATS: Dict[str, Type[ToolService]] = {
    v.CALL_API_FORMAT: v
    for v in globals().values()
    if isinstance(v, type) and issubclass(v, ToolService) and not v == ToolService and not bool(v.__abstractmethods__)
}
"""A dictionary mapping call API formats to their corresponding ToolService classes."""
