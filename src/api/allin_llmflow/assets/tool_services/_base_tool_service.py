"""
This module defines the base class for a ToolService asset. A ToolService asset refers to a deployed tool (or a
selection of tools) that can be used for various tasks. The ToolService asset provides a common interface for calling
the tool service and converting it to a Tool instance.
"""

import abc
import inspect
import re
from typing import Any, Optional

from allin_llmflow.assets._base_asset import _Asset
from allin_llmflow.dataclasses import Tool


class _ToolService(_Asset, metaclass=abc.ABCMeta):
    """
    A ToolService asset refers to a deployed tool (or a selection of tools) that can be used for various tasks.

    :param name: The name of the tool service.
    :param api_key: The API key to use for the tool service, defaults to None.
    :param uri: The URI of the tool service.
    :param organization: The organization of the tool service, defaults to None.
    """

    ASSET_TYPE = "tool-service"
    CALL_API_FORMAT: str = NotImplemented
    """The expected format of the call API. This should be overridden by subclasses."""

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        uri: str,
        organization: Optional[str] = None,
    ):
        self.uri = uri
        self.organization = organization
        configs = {
            "uri": uri,
            "organization": organization,
            "call_api_format": self.CALL_API_FORMAT,
        }
        if api_key:
            secrets = {"api_key": api_key}
        else:
            secrets = {}

        super().__init__(name, configs=configs, secrets=secrets)

    @property
    def api_key(self) -> Optional[str]:
        """Get the api key."""
        return self.secrets.get("api_key")

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        """Set the api key."""
        self.secrets["api_key"] = value

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """
        Perform a call to the tool service.

        :param args: positional arguments to pass to the tool service.
        :param kwargs: keyword arguments to pass to the tool service.
        :return: the result of the call.
        """
        raise NotImplementedError("The __call__ method must be implemented by subclasses.")

    def as_tool(self, name: Optional[str] = None, description: Optional[str] = None) -> Tool:
        """
        Convert the tool service to a Tool instance. This method is especially useful when the tool service is used
        directly by a Large Language Model.

        :param name: The name of the tool, defaults to the name of the tool service.
        :param description: The description of the tool, defaults to the inherited docstring of the __call__ method.
        :returns: A Tool instance representing the tool service.
        :raises ValueError: If the __call__ method accepts *args or **kwargs, and therefore cannot be described for
            tool use scenarios.
        """
        call_method = self.__call__
        sig = inspect.signature(call_method)
        for param in sig.parameters.values():
            # We need to disallow *args and **kwargs in the __call__ method so its parameters can be correctly
            # cast to a JSON schema for tool use scenarios.
            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                raise ValueError(
                    f"The __call__ method of {self.__class__.__name__} must not accept *args or **kwargs"
                    "to be used as a tool."
                )

        # Handle the name
        if name is None:
            class_name = self.__class__.__name__
            # Convert CamelCase to snake_case and remove "ToolService" suffix if present
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
            if name.endswith("_tool_service"):
                name = name[:-13]

        # Handle the description by searching the class hierarchy for docstrings
        if description is None:
            # Start with the current class's __call__ docstring
            description = call_method.__doc__

            # If that's empty, search up the inheritance chain for first non-empty docstring
            if not description or not description.strip():
                for cls in self.__class__.__mro__:
                    if hasattr(cls, "__call__") and cls.__call__.__doc__:
                        description = cls.__call__.__doc__
                        break

                # If still no docstring found, use the class docstring
                if not description or not description.strip():
                    description = self.__class__.__doc__ or f"Tool service using {self.__class__.__name__}"

        # Create a wrapper function with the right name and docstring
        def wrapped_call(*args, **kwargs):
            return call_method(*args, **kwargs)

        wrapped_call.__name__ = name
        wrapped_call.__doc__ = description
        setattr(wrapped_call, "__signature__", sig)
        wrapped_call.__annotations__ = getattr(call_method, "__annotations__", {})

        return Tool.from_function(wrapped_call)
