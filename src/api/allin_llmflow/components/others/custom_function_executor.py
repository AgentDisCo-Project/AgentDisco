"""
This module contains the CustomFunctionExecutor component, which can be used to execute custom functions at runtime.
"""

from importlib import import_module
from inspect import signature
from typing import Any, Callable, Dict, Optional

from haystack import component, default_to_dict, default_from_dict, DeserializationError
from haystack.core.component.types import _empty
from haystack.utils import serialize_callable, serialize_type, deserialize_type  # , deserialize_callable
from typing_extensions import Self


# Temporary fix for the issue with the serialization of the Callable type
# This will be removed once https://github.com/deepset-ai/haystack/pull/8648 is merged and released.
def deserialize_callable(callable_handle: str) -> Optional[Callable]:
    """
    Deserializes a callable given its full import path as a string.

    :param callable_handle: The full path of the callable_handle
    :return: The callable
    :raises DeserializationError: If the callable cannot be found
    """
    parts = callable_handle.split(".")
    module_name = ".".join(parts[:-1])
    function_name = parts[-1]
    try:
        module = import_module(module_name, None)
    except Exception as e:
        raise DeserializationError(f"Could not locate the module of the callable: {module_name}") from e
    deserialized_callable = getattr(module, function_name, None)
    if not deserialized_callable:
        raise DeserializationError(f"Could not locate the callable: {function_name}")
    return deserialized_callable


@component
class CustomFunctionExecutor:
    """
    A component that can be used to execute custom functions at runtime. It is particularly useful for integrating
    simple custom logic without the need to define a new component.

    When the `run` method is called, the component executes the custom function with keyword arguments directly from
    the inputs of the component. The custom function is expected to return a dictionary that matches the output types
    specified in the `output_types` parameter. For example, if the output types are {"result": str}, the custom
    function needs to return a dictionary with a key "result" mapping to a string value.

    The custom function needs be 'findable' by the callable serialization functions, i.e., the custom function needs
    to be directly importable by the serialization/deserialization machinery. This is necessary for the component to
    be correctly serialized and deserialized when saving and loading the pipeline.

    :param custom_function: The custom function to execute.
    :param output_types: The definition of the output parameters of the component.
    """

    def __init__(self, custom_function: Callable[..., Dict[str, Any]], output_types: Dict[str, type]):
        self.custom_function = custom_function
        self.output_types = output_types
        component.set_output_types(self, **self.output_types)
        for input_name, input_type in signature(custom_function).parameters.items():
            if input_type.kind not in (input_type.KEYWORD_ONLY, input_type.POSITIONAL_OR_KEYWORD):
                raise ValueError(f"Custom function input '{input_name}' must be a keyword argument.")
            component.set_input_type(
                self,
                input_name,
                input_type.annotation if input_type.annotation != input_type.empty else Any,
                default=input_type.default if input_type.default != input_type.empty else _empty,
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: The serialized dictionary representation of the component.
        """
        return default_to_dict(
            self,
            custom_function=serialize_callable(self.custom_function),
            output_types={k: serialize_type(v) for k, v in self.output_types.items()},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Loads a CustomFunctionExecutor component from its serialized dictionary representation.

        :param data: The serialized dictionary representation of the component.
        :return: The loaded CustomFunctionExecutor component.
        """
        init_parameters = data.get("init_parameters", {}).copy()
        init_parameters["custom_function"] = deserialize_callable(init_parameters["custom_function"])
        init_parameters["output_types"] = {k: deserialize_type(v) for k, v in init_parameters["output_types"].items()}
        return default_from_dict(cls, {"type": data["type"], "init_parameters": init_parameters})

    def run(self, **kwargs):
        """
        Execute the custom function with the provided keyword arguments.

        :param kwargs: The keyword arguments to pass to the custom function.
        :return: The output of the custom function.
        """
        try:
            return self.custom_function(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Error executing custom function: {e}") from e
