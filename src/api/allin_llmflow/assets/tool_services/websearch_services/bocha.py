"""
This module contains a WebSearchService asset that refers to a deployed web search service following the Bocha WebSearch
 API format.
"""

import json
from typing import Any, Dict, List, Optional

import httpx
from haystack import Document

from allin_llmflow.assets.tool_services.websearch_services._base_websearch_service import _HttpxWebSearchService

BOCHA_BASE_URI = "https://api.bochaai.com/"
BOCHA_WEBSEARCH_ENDPOINT_SUFFIX = "v1/web-search"


class BochaWebSearchService(_HttpxWebSearchService):
    """
    A WebSearchService asset that refers to a deployed web search service following the Bocha WebSearch API format.
    """

    CALL_API_FORMAT = "bocha-websearch"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        uri: str = BOCHA_BASE_URI,
        organization: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, api_key=api_key, uri=uri, organization=organization, client_kwargs=client_kwargs)
        self.search_endpoint = f"{self.uri.rstrip('/')}/{BOCHA_WEBSEARCH_ENDPOINT_SUFFIX}"

    def build_request(
        self,
        client: httpx.Client,
        query: str,
        *,
        num: Optional[int] = None,
        page: Optional[int] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> httpx.Request:

        body: Dict[str, Any] = {"query": query}
        if num:
            body["num"] = num
        if page:
            body["page"] = page
        if search_kwargs:
            body.update(search_kwargs)

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        request = client.build_request("POST", self.search_endpoint, headers=headers, json=body)
        return request

    def parse_response(self, response: httpx.Response) -> List[Document]:
        try:
            response_json = response.json()
        except json.JSONDecodeError as err:
            raise ValueError(
                f"The websearch service '{self.name}' returned an invalid JSON response: {response.text}"
            ) from err
        documents = []
        for idx, webpage_data in enumerate(response_json["data"]["webPages"].get("value", [])):
            content = webpage_data.get("summary") or webpage_data.get("snippet")
            document = Document(
                content=content,
                meta={
                    "position": idx,
                    "search_id": webpage_data.get("id"),
                    "title": webpage_data.get("name"),
                    "url": webpage_data.get("url"),
                    "date": webpage_data.get("dateLastCrawled"),
                    "snippet": webpage_data.get("snippet"),
                    "site_info": {"name": webpage_data.get("siteName"), "icon": webpage_data.get("siteIcon")},
                    "language": webpage_data.get("language"),
                },
            )
            documents.append(document)
        return documents
