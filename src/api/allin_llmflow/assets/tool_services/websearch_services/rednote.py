"""
This module contains a WebSearchService asset that refers to a deployed web search service following the Bocha WebSearch
 API format.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from haystack import Document

from allin_llmflow.assets.tool_services.websearch_services._base_websearch_service import _HttpxWebSearchService


logger = logging.getLogger(__name__)

REDNOTE_EXPLORE_URL_TEMPLATE = "https://www.xiaohongshu.com/explore/{note_id}"


class RednoteWebSearchService(_HttpxWebSearchService):
    """
    A WebSearchService asset that refers to a deployed web search service following the Bocha WebSearch API format.
    """

    CALL_API_FORMAT = "rednote-websearch"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        uri: str,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, uri=uri, client_kwargs=client_kwargs)
        self.search_endpoint = uri
        self.user_id = self.configs["user_id"] = user_id

    def build_request(
        self,
        client: httpx.Client,
        query: str,
        *,
        num: Optional[int] = None,
        page: Optional[int] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> httpx.Request:

        body: Dict[str, Any] = {
            "query": query,
            "requestId": str(uuid.uuid4()),
            "userId": self.user_id,
            "business_type": "POI_OFFLINE",
        }
        if num:
            if num > 20:
                logger.warning("RednoteWebSearchService only supports up to 20 results, but got %s. Truncating.", num)
                num = 20
            body["pageSize"] = num
        if page:
            body["pageIndex"] = page - 1
        if search_kwargs:
            body.update(search_kwargs)

        headers = {"Content-Type": "application/json"}
        request = client.build_request("POST", self.search_endpoint, headers=headers, json=body)
        return request

    def parse_response(self, response: httpx.Response) -> List[Document]:
        try:
            response_json = response.json()
        except json.JSONDecodeError as err:
            raise ValueError(
                f"The websearch service '{self.name}' returned an invalid JSON response: {response.text}"
            ) from err
        if not response_json.get("success", False):
            raise ValueError(f"The websearch service '{self.name}' returned an error: {response_json.get('message')}")
        documents = []

        for idx, note_data in enumerate(response_json["data"]["notes"]):
            detail = note_data.get("detail", {})
            snippet = note_data.get("searchMeta", {}).get("abstractShow")
            content = detail.get("content") or snippet
            document = Document(
                content=content,
                meta={
                    "position": idx,
                    "search_id": response_json["data"].get("meta", {}).get("searchId"),
                    "title": detail.get("title"),
                    "url": REDNOTE_EXPLORE_URL_TEMPLATE.format(note_id=note_data.get("id")),
                    "date": datetime.utcfromtimestamp(detail.get("time", {}).get("updateTime") / 1000).isoformat(),
                    "snippet": snippet,
                    "search_meta": note_data.get("searchMeta"),
                    "image_list": detail.get("imagesList"),
                },
            )
            documents.append(document)
        return documents
