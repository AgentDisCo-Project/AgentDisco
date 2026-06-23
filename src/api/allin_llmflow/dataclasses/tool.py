"""
This module defines the Tool data class, which represents a tool that can be invoked by a Language Model.
This is a direct copy of
https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/dataclasses/tool.py
"""

# pylint: skip-file
# ruff: noqa
# pragma: exclude file

# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import re
import textwrap
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict

from haystack.lazy_imports import LazyImport
from haystack.utils import deserialize_callable, serialize_callable
from pydantic import create_model

with LazyImport(message="Run 'pip install jsonschema'") as jsonschema_import:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

# Define constants for ReST directives
REST_DIRECTIVES = r"param|return|returns|raise|raises"


class ToolInvocationError(Exception):
    """
    Exception raised when a Tool invocation fails.
    """

    pass


class SchemaGenerationError(Exception):
    """
    Exception raised when automatic schema generation fails.
    """

    pass


@dataclass
class Tool:
    """
    Data class representing a tool for which Language Models can prepare a call.

    Accurate definitions of the textual attributes such as `name` and `description`
    are important for the Language Model to correctly prepare the call.

    :param name:
        Name of the tool.
    :param description:
        Description of the tool.
    :param parameters:
        A JSON schema defining the parameters expected by the tool.
    :param function:
        The function that will be invoked when the tool is called.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

    def __post_init__(self):
        jsonschema_import.check()
        # Check that the parameters define a valid JSON schema
        try:
            Draft202012Validator.check_schema(self.parameters)
        except SchemaError as e:
            raise ValueError("The provided parameters do not define a valid JSON schema") from e

    @property
    def tool_spec(self) -> Dict[str, Any]:
        """
        Return the tool specification to be used by the Language Model.
        """
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def invoke(self, **kwargs) -> Any:
        """
        Invoke the tool with the provided keyword arguments.
        """

        try:
            result = self.function(**kwargs)
        except Exception as e:
            raise ToolInvocationError(f"Failed to invoke tool `{self.name}` with parameters {kwargs}") from e
        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Tool to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        serialized = asdict(self)
        serialized["function"] = serialize_callable(self.function)
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        """
        Deserializes the Tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized Tool.
        """
        data["function"] = deserialize_callable(data["function"])
        return cls(**data)

    @classmethod
    def from_function(cls, function: Callable, docstring_as_desc: bool = True) -> "Tool":
        """
        Create a Tool instance from a function.

        Usage example:
        ```python
        from typing import Annotated, Literal
        from haystack_experimental.dataclasses import Tool

        def get_weather(
            city: Annotated[str, "the city for which to get the weather"] = "Munich",
            unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius"):
            '''A simple function to get the current weather for a location.'''
            return f"Weather report for {city}: 20 {unit}, sunny"

        tool = Tool.from_function(get_weather)

        print(tool)
        >>> Tool(name='get_weather', description='A simple function to get the current weather for a location.',
        >>> parameters={
        >>> 'type': 'object',
        >>> 'properties': {
        >>>     'city': {'type': 'string', 'description': 'the city for which to get the weather', 'default': 'Munich'},
        >>>     'unit': {
        >>>         'type': 'string',
        >>>         'enum': ['Celsius', 'Fahrenheit'],
        >>>         'description': 'the unit for the temperature',
        >>>         'default': 'Celsius',
        >>>     },
        >>>     }
        >>> },
        >>> function=<function get_weather at 0x7f7b3a8a9b80>)
        ```

        :param function:
            The function to be converted into a Tool.
            The function must include type hints for all parameters.
            If a parameter is annotated using `typing.Annotated`, its metadata will be used as parameter description.
        :param docstring_as_desc:
            Whether to use the function's docstring as the tool description.

        :returns:
            The Tool created from the function.

        :raises ValueError:
            If any parameter of the function lacks a type hint.
        :raises SchemaGenerationError:
            If there is an error generating the JSON schema for the Tool.
        """
        tool_description = ""
        param_descriptions_from_rest: Dict[str, str] = {}
        return_description = ""
        raises_descriptions = []

        # Process docstring if available
        if docstring_as_desc and function.__doc__:
            docstring = textwrap.dedent(function.__doc__).strip()

            # Check if this is a ReST-style docstring
            if re.search(rf":({REST_DIRECTIVES})\s+", docstring):
                # Extract main description (everything before first directive)
                main_parts = re.split(rf":({REST_DIRECTIVES})\s+", docstring, 1)
                tool_description = main_parts[0].strip()

                # Parse parameter descriptions (handling both :param name: and :param type name: formats)
                param_pattern = re.compile(rf":param\s+(\w+)\s*:(.*?)(?=:(?:{REST_DIRECTIVES})|$)", re.DOTALL)
                param_descriptions_from_rest = {name: desc.strip() for name, desc in param_pattern.findall(docstring)}

                # Parse return descriptions
                return_pattern = re.compile(rf":return:\s*(.*?)(?=:(?:{REST_DIRECTIVES})|$)", re.DOTALL)
                return_matches = return_pattern.findall(docstring)
                if return_matches:
                    return_description = return_matches[0].strip()

                # Parse raises descriptions
                raises_pattern = re.compile(
                    rf":raises?\s+(\w+(?:,\s*\w+)*)\s*:\s*(.*?)(?=:(?:{REST_DIRECTIVES})|$)", re.DOTALL
                )
                for exc_types, desc in raises_pattern.findall(docstring):
                    for exc_type in re.split(r",\s*", exc_types):
                        raises_descriptions.append(f"{exc_type}: {desc.strip()}")
            else:
                # Not a ReST-style docstring, use the whole thing
                tool_description = docstring.strip()

        # Build a comprehensive description including return values and exceptions
        full_description = tool_description

        if return_description:
            full_description += f"\n\nReturns: {return_description}"

        if raises_descriptions:
            full_description += "\n\nRaises:\n" + "\n".join(f"- {r}" for r in raises_descriptions)

        signature = inspect.signature(function)

        # collect fields (types and defaults) and descriptions from function parameters
        fields: Dict[str, Any] = {}
        descriptions = {}

        for name, param in signature.parameters.items():
            if param.annotation is param.empty:
                raise ValueError(f"Function '{function.__name__}': parameter '{name}' does not have a type hint.")

            # Default handling - required parameters use ...
            default = param.default if param.default is not param.empty else ...
            fields[name] = (param.annotation, default)

            # Priority 1: Get descriptions from Annotated type hints
            if hasattr(param.annotation, "__metadata__"):
                descriptions[name] = param.annotation.__metadata__[0]
            # Priority 2: Get descriptions from ReST docstring
            elif name in param_descriptions_from_rest:
                descriptions[name] = param_descriptions_from_rest[name]

        # create Pydantic model and generate JSON schema
        try:
            model = create_model(function.__name__, **fields)
            schema = model.model_json_schema()
        except Exception as e:
            raise SchemaGenerationError(f"Failed to create JSON schema for function '{function.__name__}'") from e

        # we don't want to include title keywords in the schema, as they contain redundant information
        # there is no programmatic way to prevent Pydantic from adding them, so we remove them later
        # see https://github.com/pydantic/pydantic/discussions/8504
        _remove_title_from_schema(schema)

        # add parameters descriptions to the schema
        for name, description in descriptions.items():
            if name in schema["properties"]:
                schema["properties"][name]["description"] = description

        return Tool(name=function.__name__, description=full_description, parameters=schema, function=function)


def _remove_title_from_schema(schema: Dict[str, Any]):
    """
    Remove the 'title' keyword from JSON schema and contained property schemas.

    :param schema:
        The JSON schema to remove the 'title' keyword from.
    """
    schema.pop("title", None)

    for property_schema in schema["properties"].values():
        for key in list(property_schema.keys()):
            if key == "title":
                del property_schema[key]


def deserialize_tools_inplace(data: Dict[str, Any], key: str = "tools"):
    """
    Deserialize tools in a dictionary inplace.

    :param data:
        The dictionary with the serialized data.
    :param key:
        The key in the dictionary where the tools are stored.
    """
    if key in data:
        serialized_tools = data[key]

        if serialized_tools is None:
            return

        if not isinstance(serialized_tools, list):
            raise TypeError(f"The value of '{key}' is not a list")

        deserialized_tools = []
        for tool in serialized_tools:
            if not isinstance(tool, dict):
                raise TypeError(f"Serialized tool '{tool}' is not a dictionary")
            deserialized_tools.append(Tool.from_dict(tool))

        data[key] = deserialized_tools
