"""
This module contains a generator that uses a chat model to generate chat responses from a given list of messages.
"""

from typing import Any, Callable, Dict, List, Optional

from haystack import component, default_to_dict, default_from_dict
from haystack.utils import serialize_callable, deserialize_callable
from typing_extensions import Self

from allin_llmflow.assets.asset_factory import ModelServiceFactory
from allin_llmflow.assets.model_services import ChatModelService
from allin_llmflow.dataclasses import AssetReference, ChatMessage, Tool, StreamingChunk
from allin_llmflow.dataclasses.tool import deserialize_tools_inplace


@component
class ChatGenerator:
    """
    A class used to generate chat responses from a given chat model service.

    :param model_service: An instance of the ChatModelService to interact with the model.
    :param model: The name of the model to use for generating responses, defaults to "gpt-3.5-turbo"
    :param stream: A flag to enable streaming responses, defaults to False
    :param streaming_callbacks: A list of callback functions to handle streaming responses, defaults to None
    :param timeout: The timeout duration for the HTTP client, defaults to 10.0
    :param generation_kwargs: Additional keyword arguments to pass to the model for generation, defaults to None
    """

    def __init__(
        self,
        model_service: ChatModelService,
        model: Optional[str] = None,
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
        timeout: Optional[float] = 20.0,
        tools: Optional[List[Tool]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_service = model_service
        self.model = model or ""
        self.stream = stream
        self.streaming_callbacks = streaming_callbacks
        self.tools = tools
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a serialized dictionary representation of the component.

        :return: A dictionary representation of the component.
        """
        model_service_reference = self.model_service.reference
        # Serialize the streaming callbacks
        callback_names = (
            [serialize_callable(callback) for callback in self.streaming_callbacks]
            if self.streaming_callbacks
            else None
        )
        return default_to_dict(
            self,
            model_service=model_service_reference.to_dict(),
            model=self.model,
            stream=self.stream,
            streaming_callbacks=callback_names,
            timeout=self.timeout,
            tools=[tool.to_dict() for tool in self.tools] if self.tools else None,
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Loads a ChatGenerator component from its serialized dictionary representation.

        :param data: The serialized dictionary representation of the component.
        :return: The deserialized ChatGenerator component.
        """
        # Make a shallow copy of init_parameters to avoid modifying the original data
        init_parameters_data = data.get("init_parameters", {}).copy()
        # Load the chat model service from reference
        model_service_reference = AssetReference.from_dict(init_parameters_data["model_service"])
        model_service = ModelServiceFactory.load_from_reference(model_service_reference)
        if not isinstance(model_service, ChatModelService):
            raise TypeError(f"The model service must be an instance of ChatModelService, got {type(model_service)}")
        init_parameters_data["model_service"] = model_service

        # Deserialize the streaming callbacks
        serialized_callbacks = init_parameters_data.get("streaming_callbacks")
        if serialized_callbacks and isinstance(serialized_callbacks, list):
            init_parameters_data["streaming_callbacks"] = [
                deserialize_callable(callback) for callback in serialized_callbacks
            ]
        deserialize_tools_inplace(init_parameters_data, key="tools")

        # Load the component
        return default_from_dict(cls, {"type": data["type"], "init_parameters": init_parameters_data})

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Generates responses for the given list of messages.

        :param messages: The list of messages to generate responses for.
        :param generation_kwargs: Additional keyword arguments to pass to the model for generation, defaults to None.
        :return: A dictionary containing the generated responses.
        """
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        completions = self.model_service.infer(
            messages,
            model=self.model,
            timeout=self.timeout,
            stream=self.stream,
            streaming_callbacks=self.streaming_callbacks,
            tools=self.tools,
            inference_kwargs=generation_kwargs,
        )

        return {"replies": completions}

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Asynchronously generates responses for the given list of messages.

        :param messages: The list of messages to generate responses for.
        :param generation_kwargs: Additional keyword arguments to pass to the model for generation, defaults to None.
        :return: A dictionary containing the generated responses.
        """
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        completions = await self.model_service.ainfer(
            messages,
            model=self.model,
            timeout=self.timeout,
            stream=self.stream,
            streaming_callbacks=self.streaming_callbacks,
            tools=self.tools,
            inference_kwargs=generation_kwargs,
        )

        return {"replies": completions}
