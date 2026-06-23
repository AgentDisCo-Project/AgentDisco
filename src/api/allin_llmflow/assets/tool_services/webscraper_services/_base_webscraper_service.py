"""
This module defines the base class for a WebScraperService asset. A WebScraperService asset refers to a deployed web
scraper service that can be used for scraping and parsing web pages. The WebScraperService asset provides a common
interface for calling the web scraper service and converting it to a list of Document instances.

Moreover, this module also provides HttpxWebScraperService, a base class for web scraper services that use HTTPX for
making requests. This class provides abstract methods for building the request and parsing the response from the web
scraper service.
"""

import abc
from typing import Any, Dict, List, Optional

import httpx
from haystack import Document

from allin_llmflow.assets.tool_services._base_tool_service import _ToolService


class _WebScraperService(_ToolService, metaclass=abc.ABCMeta):
    """
    A WebScraperService asset refers to a deployed web scraper service that can be used for scraping and parsing web
    pages.
    """

    @abc.abstractmethod
    def __call__(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
        output_formats: Optional[List[str]] = None,
        scraper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform web scraping using the web scraper service.

        :param url: The URL of the web page to scrape.
        :param timeout: The timeout for the request.
        :param output_formats: The output formats of the scraped content. The default format varies depending on the
            specific type of web scraper service.
        :param scraper_kwargs: Additional crawler keyword arguments to include in the request.
        :return: The search results as a list of Document instances. Depending on the method of scraping, the list may
            include multiple Document instances for different subpages. Normally, the Document metadata should at least
            contain the following fields:
            - `title`: The title of the web page.
            - `url`: The URL of the web page.
            - `content_type`: The content type of the web page (e.g. "markdown", "html").
            - `status_code`: The status code of the response.
        """
        raise NotImplementedError("The __call__ method must be implemented by subclasses.")


class _HttpxWebScraperService(_WebScraperService):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        uri: str,
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
        client: httpx.Client,
        url: str,
        *,
        timeout: Optional[float] = None,
        output_formats: Optional[List[str]] = None,
        scraper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> httpx.Request:
        """
        Build the HTTPX request for the web scraper service.

        :param client: The HTTPX client to use for the request.
        :param url: The URL of the web page to scrape.
        :param timeout: The timeout for the crawler.
        :param output_formats: The output formats of the scraped content.
        :param scraper_kwargs: Additional crawler keyword arguments to include in the request.
        :return: The HTTPX request.
        """
        raise NotImplementedError("build_request method must be implemented")

    @abc.abstractmethod
    def parse_response(self, response: httpx.Response, output_formats: Optional[List[str]] = None) -> List[Document]:
        """
        Parse the response from the web search service.

        :param response: The response from the web scrapper service.
        :param output_formats: The output formats of the scraped content.
        :return: The search results as a list of Document instances.
        """
        raise NotImplementedError("parse_response method must be implemented")

    def __call__(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
        output_formats: Optional[List[str]] = None,
        scraper_kwargs: Optional[Dict[str, Any]] = None,
    ):
        with httpx.Client(timeout=timeout, **self.client_kwargs) as client:
            request = self.build_request(
                client, url, timeout=timeout, output_formats=output_formats, scraper_kwargs=scraper_kwargs
            )
            try:
                response = client.send(request).raise_for_status()
            except httpx.HTTPStatusError as e:
                raise httpx.HTTPStatusError(
                    str(e) + f"\nWebScrapper Service: {self.name}\nDetails: {e.response.text}",
                    request=e.request,
                    response=e.response,
                )
            documents = self.parse_response(response, output_formats=output_formats)
        return documents
