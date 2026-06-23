"""
This module provides utility functions for enabling and customizing tracing in the application.
"""

import contextlib
import json
import logging
import warnings
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

import haystack.tracing
from haystack.tracing import Span
from haystack_integrations.tracing.langfuse.tracer import (
    LangfuseSpan as _LangfuseSpan,
    LangfuseTracer as _LangfuseTracer,
    _SUPPORTED_CHAT_GENERATORS,
)
from langfuse import Langfuse

from allin_llmflow.assets.model_services.chat_model_services.openai import _convert_message_to_openai_format
from allin_llmflow.dataclasses.chat_message import ChatMessage
from allin_llmflow.utils.lazy_imports import LazyImport
from allin_llmflow.utils.serialization import default_json_serializer

# Additional package requirements for enabling opentelemetry tracing backend
with LazyImport("Run 'pip install opentelemetry-sdk opentelemetry-exporter-otlp'") as opentelemetry_import:
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

tracing_context: ContextVar[Dict[Any, Any]] = ContextVar("tracing_context", default={})
"""
A context variable containing information related to tracing. This can be used to store trace, user, and session IDs,
custom tags and pipeline version information.
"""

logger = logging.getLogger(__name__)


class LangfuseSpan(_LangfuseSpan):
    """
    A customized Langfuse span that supports additional metadata tracking.
    """

    def set_content_tag(self, key: str, value: Any) -> None:
        """
        Set a content-specific tag for this span.

        :param key: The content tag key.
        :param value: The content tag value.
        """
        if not haystack.tracing.tracer.is_content_tracing_enabled:
            return
        # Langfuse does fancy rendering of input/output messages, so we need to convert them to the format it expects.
        # However, this will cause all other input/output data to be dropped, and this is not ideal.
        # We will need to find a better way to handle this in the future.
        if key.endswith(".input"):
            if "messages" in value and all(isinstance(m, ChatMessage) for m in value["messages"]):
                messages = [_convert_message_to_openai_format(m) for m in value["messages"]]
                self._span.update(input=messages)
            else:
                self._span.update(input=value)
        elif key.endswith(".output"):
            if "replies" in value and all(isinstance(r, ChatMessage) for r in value["replies"]):
                if all(isinstance(r, ChatMessage) for r in value["replies"]):
                    replies = [_convert_message_to_openai_format(m) for m in value["replies"]]
                else:
                    replies = value["replies"]
                self._span.update(output=replies)
            else:
                self._span.update(output=value)

        self._data[key] = value


class LangfuseTracer(_LangfuseTracer):
    """
    A customized Langfuse tracer that supports additional generators, and tracks user and session IDs and custom tags
    from context.
    """

    ALL_SUPPORTED_GENERATORS = {"ChatGenerator"} | set(_SUPPORTED_CHAT_GENERATORS)

    @classmethod
    def register_generator(cls, generator_name: str) -> None:
        """
        Register a generator to be supported by the tracer. This will allow the tracer to create a generation instead of
        a regular span, which can be useful for tracking additional metadata specific to the generator.

        :param generator_name: The name of the generator to register.
        """
        cls.ALL_SUPPORTED_GENERATORS.add(generator_name)

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: Optional[Dict[str, Any]] = None, parent_span: Optional[Span] = None
    ) -> Iterator[Span]:
        """
        Start and manage a new trace span.

        :param operation_name: The name of the operation.
        :param tags: A dictionary of tags to attach to the span.
        :param parent_span: The parent span to use for the newly created span.
        :return: A context manager yielding the span.
        """
        # pylint: disable=protected-access
        # This is to make it consistent with the original implementation
        tags = tags or {}
        span_name = tags.get("haystack.component.name", operation_name)

        if not parent_span:
            if operation_name != "haystack.pipeline.run":
                logger.warning(
                    "Creating a new trace without a parent span is not recommended for operation '%s'.", operation_name
                )
            # Create a new trace if no parent span is provided
            span = LangfuseSpan(
                self._tracer.trace(
                    name=self._name,
                    public=self._public,
                    id=tracing_context.get().get("trace_id"),
                    user_id=tracing_context.get().get("user_id"),
                    session_id=tracing_context.get().get("session_id"),
                    tags=tracing_context.get().get("tags"),
                    version=tracing_context.get().get("version"),
                )
            )
        else:
            if tags.get("haystack.component.type") in self.ALL_SUPPORTED_GENERATORS:
                span = LangfuseSpan(parent_span.raw_span().generation(name=span_name))
            else:
                span = LangfuseSpan(parent_span.raw_span().span(name=span_name))

        self._context.append(span)
        span.set_tags(tags)

        yield span

        # We do not have non-chat generators in the current version of allin-llmflow
        # if tags.get("haystack.component.type") in _SUPPORTED_GENERATORS:
        #     meta = span._data.get("haystack.component.output", {}).get("meta")
        #     if meta:
        #         # Haystack returns one meta dict for each message, but the 'usage' value
        #         # is always the same, let's just pick the first item
        #         m = meta[0]
        #         span._span.update(usage=m.get("usage") or None, model=m.get("model"))
        if tags.get("haystack.component.type") in self.ALL_SUPPORTED_GENERATORS:
            replies = span._data.get("haystack.component.output", {}).get("replies")
            if replies:
                meta = replies[0].meta
                completion_start_time = meta.get("completion_start_time")
                if completion_start_time:
                    try:
                        completion_start_time = datetime.fromisoformat(completion_start_time)
                    except ValueError:
                        logger.error("Invalid completion start time format: %s", completion_start_time)
                        completion_start_time = None

                usage = meta.get("usage")
                if usage and ("input_tokens" in usage or "output_tokens" in usage):
                    # Convert Anthropic usage format to be compatible with Langfuse
                    usage = {
                        "input": usage.get("input_tokens"),
                        "output": usage.get("output_tokens"),
                        "unit": "TOKENS",
                    }
                # We will need to temporarily add "allin-" prefix to GPT models due to a current issue in Langfuse
                # More info at https://github.com/orgs/langfuse/discussions/4231
                model = meta.get("model")
                if model and "gpt" in model.lower():
                    model = "allin-" + model
                span._span.update(
                    usage=usage,
                    model=model,
                    completion_start_time=completion_start_time,
                )

        pipeline_input = tags.get("haystack.pipeline.input_data", None)
        if pipeline_input:
            span._span.update(input=json.dumps(tags["haystack.pipeline.input_data"], default=default_json_serializer))
        pipeline_output = tags.get("haystack.pipeline.output_data", None)
        if pipeline_output:
            span._span.update(output=json.dumps(tags["haystack.pipeline.output_data"], default=default_json_serializer))

        self._context.pop()

        # Flush the trace if it does not have a parent span
        if not parent_span:
            if self.enforce_flush:
                self.flush()
        else:
            span.raw_span().end()

    def current_span(self) -> Optional[Span]:
        """
        Return the currently active span.

        :return: The currently active span.
        """
        if not self._context:
            return None
        return self._context[-1]


def enable_content_tracing() -> None:
    """
    Enable content tracing in the application.
    """
    haystack.tracing.tracer.is_content_tracing_enabled = True


def enable_langfuse_tracing(name: str, release: Optional[str] = None, public: bool = False, **kwargs) -> None:
    """
    Enable langfuse tracing in the application.

    :param name: the name of the trace to be created.
    :param release: the identifier of the current usage of the trace. Default is None.
    :param public: if public, a public link will be generated for the trace. Default is False.
    :param kwargs: additional arguments to pass to the Langfuse tracer.
    """
    langfuse_tracer = LangfuseTracer(tracer=Langfuse(release=release, **kwargs), name=name, public=public)
    haystack.tracing.enable_tracing(langfuse_tracer)


def use_local_opentelemetry_tracing(endpoint: str = "http://localhost:4318/v1/traces") -> None:
    """
    Enable opentelemetry tracing in the application using a local backend.

    :param endpoint: The endpoint of the opentelemetry tracing backend.
    """
    warnings.warn("This method is deprecated and will be removed in the future releases", DeprecationWarning)

    opentelemetry_import.check()

    # Service name is required for most backends
    resource = Resource(attributes={SERVICE_NAME: "haystack"})

    trace_provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    trace_provider.add_span_processor(processor)
    trace.set_tracer_provider(trace_provider)

    haystack.tracing.auto_enable_tracing()
