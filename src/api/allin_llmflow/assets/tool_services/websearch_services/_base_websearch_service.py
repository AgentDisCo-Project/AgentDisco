"""
This module defines the base class for a WebSearchService asset. A WebSearchService asset refers to a deployed web
search service that can be used for searching the web, taking search queries and returning webpage results. The
WebSearchService asset provides a common interface for calling the web search service and converting it to a list of
Document instances.

Moreover, this module also provides HttpxWebSearchService, a base class for web search services that use HTTPX for
making requests. This class provides abstract methods for building the request and parsing the response from the web
search service.
"""

import abc
from typing import Any, Dict, List, Optional

import httpx
from haystack import Document

from allin_llmflow.assets.tool_services._base_tool_service import _ToolService


class _WebSearchService(_ToolService, metaclass=abc.ABCMeta):
    """
    A WebSearchService asset refers to a deployed web search service that can be used for searching the web.
    """

    @abc.abstractmethod
    def __call__(
        self,
        query: str,
        *,
        num: Optional[int] = None,
        page: Optional[int] = None,
        timeout: Optional[float] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform a web search using the web search service.

        :param query: The search query.
        :param num: The number of search results to return.
        :param page: The page number of search results to return.
        :param timeout: The timeout for the request.
        :param search_kwargs: Additional search keyword arguments to include in the request.
        :return: The search results as a list of Document instances. Normally, the Document metadata should at least
            contain the following fields:
            - `position`: The position of the search result in the search results.
            - `title`: The title of the search result.
            - `snippet`: A snippet of the search result content.
            - `url`: The URL of the search result.
        """
        raise NotImplementedError("The __call__ method must be implemented by subclasses.")


class _HttpxWebSearchService(_WebSearchService):
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
        query: str,
        *,
        num: Optional[int] = None,
        page: Optional[int] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> httpx.Request:
        """
        Build the HTTPX request for the web search service.

        :param client: The HTTPX client to use for the request.
        :param query: The search query.
        :param num: The number of search results to return.
        :param page: The page number of search results to return.
        :param search_kwargs: Additional search keyword arguments to include in the request.
        :return: The HTTPX request.
        """
        raise NotImplementedError("build_request method must be implemented")

    @abc.abstractmethod
    def parse_response(self, response: httpx.Response) -> List[Document]:
        """
        Parse the response from the web search service.

        :param response: The response from the web search service.
        :return: The search results as a list of Document instances.
        """
        raise NotImplementedError("parse_response method must be implemented")

    def __call__(
        self,
        query: str,
        *,
        num: Optional[int] = None,
        page: Optional[int] = None,
        timeout: Optional[float] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        with httpx.Client(timeout=timeout, **self.client_kwargs) as client:
            request = self.build_request(client, query, num=num, page=page, search_kwargs=search_kwargs)
            try:
                response = client.send(request).raise_for_status()
            except httpx.HTTPStatusError as e:
                raise httpx.HTTPStatusError(
                    str(e) + f"\nWebSearch Service: {self.name}\nDetails: {e.response.text}",
                    request=e.request,
                    response=e.response,
                )
            documents = self.parse_response(response)
        return documents
