"""
This module contains a builder that renders prompt templates with the provided variables and assembles chat messages.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Set

from haystack import component, default_to_dict, default_from_dict
from haystack.core.component.types import _empty
from jinja2 import Template, meta
from typing_extensions import Self

from allin_llmflow.dataclasses import ChatMessage, ByteStream
from allin_llmflow.utils.asset_storage import get_local_asset_storage_path


@component
class ChatPromptBuilder:
    """
    ChatPromptBuilder is a component that renders prompt templates with the provided variables and assemble chat
    messages.

    The template variables found in the template string are used as input types for the component and are all optional,
    unless explicitly specified. If an optional template variable is not provided as an input, it will be replaced with
    an empty string in the rendered prompt.

    Usage example::

        system_template = "You are a helpful AI bot. Your name is {name}."
        user_template = "Tell me a {adjective} joke about {content}."
        builder = ChatPromptBuilder(
            system_template=system_template,
            user_template=user_template,
            required_variables=["name", "adjective"]
        )
        builder.run(name="Bob", adjective="funny", history=[])

    :param system_template: A template string for the system message.
    :param user_template: A template string for the user message.
    :param required_variables: An optional list of input variables that must be provided at all times.
    """

    def __init__(
        self,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
        required_variables: Optional[List[str]] = None,
    ):
        self._uuid: Optional[str] = None
        if not system_template and not user_template:
            logging.warning("No prompt templates provided to ChatPromptBuilder.")
        self._system_template_string = system_template
        self._user_template_string = user_template
        self.required_variables: List[str] = required_variables or []

        self.template_variables: Set[str] = set()
        self.system_template = self._process_template(system_template)
        self.user_template = self._process_template(user_template)

        for var in self.template_variables:
            if var == "_history":
                logging.warning(
                    "The parameter key '_history' is reserved for parsing chat history. Please avoid using '_history' "
                    "as a template variable to prevent conflicts."
                )
            if var == "_media":
                logging.warning(
                    "The parameter key '_media' is reserved for parsing media contents. Please avoid using '_media' "
                    "as a template variable to prevent conflicts."
                )
            component.set_input_type(self, var, Any, "" if var not in self.required_variables else _empty)

    @property
    def _name(self) -> str:
        """
        Generate a name for the component. If the component is added to a pipeline, the name will be inferred from
        the pipeline. Otherwise, a random name will be generated.

        :return: The name of the component.
        """
        if pipeline := getattr(self, "__haystack_added_to_pipeline__", None):
            return pipeline.get_component_name(self)
        if not self._uuid:
            self._uuid = self.__class__.__name__.lower() + "_" + str(uuid.uuid4())
        return self._uuid

    @classmethod
    def _load_prompt_template_from_local(cls, template_path: str) -> str:
        """
        Load the prompt template from the local asset storage.

        :param template_path: The path of the prompt template.
        :returns: The prompt template string.
        """
        prompt_template_path = get_local_asset_storage_path() / f"{template_path}"
        with open(prompt_template_path, "r", encoding="utf-8") as fr:
            return fr.read()

    @classmethod
    def _save_prompt_template_to_local(cls, template_string: str, template_path: str) -> None:
        """
        Save the prompt template to the local asset storage.

        :param template_string: The prompt template string.
        :param template_path: The path of the prompt template.
        """
        prompt_template_path = get_local_asset_storage_path() / f"{template_path}"
        with open(prompt_template_path, "w", encoding="utf-8") as fw:
            fw.write(template_string)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Loads the component from a dictionary.

        :param data: The serialized dictionary representation of the component.
        :returns: The component object.
        """
        # Make a shallow copy of init_parameters to avoid modifying the original data
        init_parameters_data = data.get("init_parameters", {}).copy()
        # Load prompt templates
        if system_template_path := init_parameters_data.get("system_template"):
            init_parameters_data["system_template"] = cls._load_prompt_template_from_local(system_template_path)
        if user_template_path := init_parameters_data.get("user_template"):
            init_parameters_data["user_template"] = cls._load_prompt_template_from_local(user_template_path)

        # Load the component
        return default_from_dict(cls, {"type": data["type"], "init_parameters": init_parameters_data})

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the component.

        :returns: Serialized dictionary representation of the component.
        """
        if self._system_template_string:
            system_template_path = f"prompt_templates/{self._name}.system_prompt.jinja2"
            self._save_prompt_template_to_local(self._system_template_string, system_template_path)
        else:
            system_template_path = None

        if self._user_template_string:
            user_template_path = f"prompt_templates/{self._name}.user_prompt.jinja2"
            self._save_prompt_template_to_local(self._user_template_string, user_template_path)
        else:
            user_template_path = None

        return default_to_dict(
            self,
            system_template=system_template_path,
            user_template=user_template_path,
            required_variables=self.required_variables,
        )

    def _process_template(self, template_string: Optional[str]) -> Optional[Template]:
        """
        Processes a template string and extracts the template variables.

        :param template_string: A template string to be processed.
        :return: A Jinja2 Template object or None if the template string is None.
        """
        if not template_string:
            return None
        template = Template(template_string)
        ast = template.environment.parse(template_string)
        self.template_variables.update(meta.find_undeclared_variables(ast))
        return template

    @component.output_types(messages=List[ChatMessage])
    def run(
        self,
        _history: Optional[List[ChatMessage]] = None,
        _media: Optional[List[ByteStream]] = None,
        **kwargs,
    ):
        """
        Renders the prompt template with the provided variables.

        :param _history: A list of previous chat messages to be included in the builder.
        :param _media: A list of media contents to be included in the builder.
        :param kwargs: The variables that will be used to render the prompt template.
        :returns: A dictionary with the following keys:
            `messages` -- The updated message list after rendering the prompt template.
        """
        messages: List[ChatMessage] = []
        missing_variables = [var for var in self.required_variables if var not in kwargs]
        if missing_variables:
            missing_vars_str = ", ".join(missing_variables)
            raise ValueError(f"Missing required input variables in ChatPromptBuilder: {missing_vars_str}")

        if self.system_template:
            messages.append(ChatMessage.from_system(self.system_template.render(kwargs)))
        if _history:
            messages.extend(_history)
        if self.user_template:
            messages.append(ChatMessage.from_user(self.user_template.render(kwargs), media=_media))

        breakpoint()
        return {"messages": messages}
