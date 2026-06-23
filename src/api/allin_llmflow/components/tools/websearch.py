"""
This module contains the WebSearch component. The WebSearch component is used to perform a web search using a
WebSearchService and return the search results as a list of Document objects.
"""

from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.utils import serialize_callable, deserialize_callable
from typing_extensions import Self

from allin_llmflow.assets.asset_factory import ToolServiceFactory
from allin_llmflow.assets.tool_services import WebSearchService
from allin_llmflow.dataclasses import AssetReference
from allin_llmflow.utils.callbacks import StreamEventCallback


@component
class WebSearch:
    """
    Performs a web search using a WebSearchService and returns the search results as a list of Document objects.

    :param websearch_service: An instance of the WebSearchService to use for the web search.
    :param num: The number of search results to return, defaults to None (i.e., the service default).
    :param page: The page number of search results to return, defaults to None (i.e., the service default).
    :param timeout: The timeout duration for the HTTP client, defaults to 20.0.
    :param search_kwargs: Additional keyword arguments to pass to the web search service.
        The keyword arguments are specific to the web search service being used. Defaults to None.
    :param stream: A flag to enable streaming search results, defaults to False.
    :param streaming_callbacks: A list of callback functions to handle streaming search results, defaults to None.
    """

    STREAMING_EVENT_TYPE = "websearch"

    def __init__(
        self,
        websearch_service: WebSearchService,
        num: Optional[int] = None,
        page: Optional[int] = None,
        timeout: Optional[float] = 20.0,
        search_kwargs: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        streaming_callbacks: Optional[List[StreamEventCallback]] = None,
    ):
        self.websearch_service = websearch_service
        self.num = num
        self.page = page
        self.timeout = timeout
        self.search_kwargs = search_kwargs or {}
        self.stream = stream
        self.streaming_callbacks = streaming_callbacks or []

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a serialized dictionary representation of the component.

        :return: A dictionary representation of the component.
        """
        websearch_service_reference = self.websearch_service.reference
        # Serialize the streaming callbacks
        callback_names = (
            [serialize_callable(callback) for callback in self.streaming_callbacks]
            if self.streaming_callbacks
            else None
        )

        return default_to_dict(
            self,
            websearch_service=websearch_service_reference.to_dict(),
            num=self.num,
            page=self.page,
            timeout=self.timeout,
            search_kwargs=self.search_kwargs,
            stream=self.stream,
            streaming_callbacks=callback_names,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Load a WebSearch component from its serialized dictionary representation.

        :param data: The serialized dictionary representation of the component.
        :return: The deserialized WebSearch component.
        """
        # Make a shallow copy of init_parameters to avoid modifying the original data
        init_parameters_data = data.get("init_parameters", {}).copy()
        # Load the chat model service from reference
        websearch_service_reference = AssetReference.from_dict(init_parameters_data["websearch_service"])
        websearch_service = ToolServiceFactory.load_from_reference(websearch_service_reference)
        if not isinstance(websearch_service, WebSearchService):
            raise TypeError(
                f"The websearch service must be an instance of WebSearchService, got {type(websearch_service)}"
            )
        init_parameters_data["websearch_service"] = websearch_service

        # Deserialize the streaming callbacks
        serialized_callbacks = init_parameters_data.get("streaming_callbacks")
        if serialized_callbacks and isinstance(serialized_callbacks, list):
            init_parameters_data["streaming_callbacks"] = [
                deserialize_callable(callback) for callback in serialized_callbacks
            ]

        return default_from_dict(cls, {"type": data["type"], "init_parameters": init_parameters_data})

    @component.output_types(documents=List[Document])
    def run(self, query: str, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Run the WebSearch component to perform a web search using the configured WebSearchService.

        :param query: The search query.
        :param search_kwargs: Additional keyword arguments to pass to the web search service at runtime.
            These arguments will merge with the search_kwargs provided during initialization.
        :return:
            A dictionary with the following keys:
                - `documents`: The search results as a list of Document objects.
        """
        if self.stream:
            for callback in self.streaming_callbacks:
                callback(query, event_type="search_query")

        # update search_kwargs with any additional search_kwargs passed to the run method
        search_kwargs = {**self.search_kwargs, **(search_kwargs or {})}

        documents = self.websearch_service(
            query,
            num=self.num,
            page=self.page,
            timeout=self.timeout,
            search_kwargs=search_kwargs,
        )

        if self.stream:
            for callback in self.streaming_callbacks:
                callback([doc.to_dict(flatten=False) for doc in documents], event_type="search_results")

        return {"documents": documents}
