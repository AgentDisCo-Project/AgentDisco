"""
This module provides a collection of classes that represent and manage webscraper services in the application.

A webscraper service is a deployed tool (or a selection of tools) that can be used for scraping web pages.

Webscraper services include:

- WebScraperService:
    An abstract base class of a generic service for managing and interacting with webscraper tools.
- HttpxWebScraperService:
    An abstract base class of a generic service for managing and interacting with webscraper tools using HTTPX.
- FirecrawlWebScraperService:
    A service specifically designed for managing and interacting with Firecrawl webscraper tools.
"""

from allin_llmflow.assets.tool_services.webscraper_services._base_webscraper_service import (
    _WebScraperService as WebScraperService,
    _HttpxWebScraperService as HttpxWebScraperService,
)
from allin_llmflow.assets.tool_services.webscraper_services.firecrawl import FirecrawlWebScraperService

__all__ = [
    "WebScraperService",
    "HttpxWebScraperService",
    "FirecrawlWebScraperService",
]
