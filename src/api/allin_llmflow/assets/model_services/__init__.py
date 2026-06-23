"""
This module provides a collection of classes that represent and manage model services in the application.

A model service is a deployed model (or a selection of models) that can be used for inference. Model services can be
used to generate responses to input data, such as text or images, and have wide-ranging use cases in an application.
"""

from typing import Dict, Type

from allin_llmflow.assets.model_services._base_model_service import _ModelService as ModelService
from allin_llmflow.assets.model_services.chat_model_services import *  # noqa: F403
from allin_llmflow.assets.model_services.embedding_model_services import *  # noqa: F403

# Register all supported inference API formats
SUPPORTED_INFERENCE_API_FORMATS: Dict[str, Type[ModelService]] = {
    v.INFERENCE_API_FORMAT: v
    for v in globals().values()
    if isinstance(v, type) and issubclass(v, ModelService) and not v == ModelService and not bool(v.__abstractmethods__)
}
"""A dictionary mapping inference API formats to their corresponding ModelService classes."""
