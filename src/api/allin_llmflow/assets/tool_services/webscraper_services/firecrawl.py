"""
This module contains a WebScraperService asset that refers to a deployed web scraper service following Firecrawl API
 format.
"""

import json
from typing import Any, Dict, List, Optional

import httpx
from haystack import Document

from allin_llmflow.assets.tool_services.webscraper_services._base_webscraper_service import _HttpxWebScraperService

FIRECRAWL_BASE_URI = "https://api.firecrawl.dev/"
FIRECRAWL_SCRAPE_ENDPOINT_SUFFIX = "v1/scrape"


class FirecrawlWebScraperService(_HttpxWebScraperService):
    """
    A WebScraperService asset that refers to a deployed web scraper service following Firecrawl API format.
    """

    CALL_API_FORMAT = "firecrawl-webscraper"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        uri: str = FIRECRAWL_BASE_URI,
        organization: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, api_key=api_key, uri=uri, organization=organization, client_kwargs=client_kwargs)
        self.scrape_endpoint = f"{self.uri.rstrip('/')}/{FIRECRAWL_SCRAPE_ENDPOINT_SUFFIX}"

    def build_request(
        self,
        client: httpx.Client,
        url: str,
        *,
        timeout: Optional[float] = None,
        output_formats: Optional[List[str]] = None,
        scraper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> httpx.Request:
        body: Dict[str, Any] = {"url": url}
        if timeout:
            body["timeout"] = int(timeout * 1000)  # Convert to milliseconds
        if output_formats:
            body["formats"] = output_formats
        if scraper_kwargs:
            body.update(scraper_kwargs)

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        request = client.build_request("POST", self.scrape_endpoint, headers=headers, json=body)
        return request

    def parse_response(self, response: httpx.Response, output_formats: Optional[List[str]] = None) -> List[Document]:
        try:
            response_json = response.json()
        except json.JSONDecodeError as err:
            raise ValueError(
                f"The websearch service '{self.name}' returned an invalid JSON response: {response.text}"
            ) from err

        data = response_json["data"]
        metadata = data["metadata"]

        # Find the content and content type
        content = None
        meta: Dict[str, Any] = {}
        output_formats = output_formats or ["markdown"]  # Default output format
        for key in output_formats:
            if key in data and data[key]:
                if content is None:
                    content = data[key]
                    meta["content_type"] = key
                else:
                    meta[key] = data[key]

        # Update metadata
        meta.update(
            {
                "status_code": metadata.get("statusCode"),
                "url": metadata.get("sourceURL"),
                "title": metadata.get("title"),
                "description": metadata.get("description"),
            }
        )
        if "error" in metadata and metadata["error"]:
            meta["error"] = metadata["error"]

        document = Document(content=content, meta=meta)
        return [document]

    def __call__(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
        output_formats: Optional[List[str]] = None,
        scraper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform web scraping using the firecrawl web scraper service.

        :param url: The URL of the web page to scrape.
        :param timeout: The timeout for the request.
        :param output_formats: The output formats of the scraped content. If left as None, the content will be returned
            in Markdown format by default.
        :param scraper_kwargs: Additional crawler keyword arguments to include in the request.
            See the API documentation at https://docs.firecrawl.dev/api-reference/endpoint/scrape for the full list
            of supported parameters.
        :return: The search results as a list of Document instances. Depending on the method of scraping, the list may
            include multiple Document instances for different subpages. The Document metadata usually contains the
            following fields:
            - `title`: The title of the web page.
            - `url`: The URL of the web page.
            - `description`: The description of the web page.
            - `content_type`: The content type of the web page (e.g. "markdown", "html").
            - `status_code`: The status code of the response.
            - `error`: The error message if the scrape failed.
        """
        # pylint: disable=useless-parent-delegation
        return super().__call__(url, timeout=timeout, output_formats=output_formats, scraper_kwargs=scraper_kwargs)
