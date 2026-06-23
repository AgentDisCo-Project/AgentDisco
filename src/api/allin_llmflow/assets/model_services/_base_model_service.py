"""
This module defines the base class for model services. A model service is a deployed model (or a selection of models)
that can be used for inference. Model services can be used to generate responses to input data, such as text or images,
and have wide-ranging use cases in an application.
"""

import abc
from typing import Optional

from allin_llmflow.assets._base_asset import _Asset


class _ModelService(_Asset, metaclass=abc.ABCMeta):
    """
    A ModelService asset refers to a deployed model (or a selection of models) that can be used for inference.

    :param name: The name of the model service.
    :param api_key: The API key to use for the model service, defaults to None.
    :param uri: The URI of the model service.
    :param organization: The organization of the model service, defaults to None.
    """

    ASSET_TYPE = "model-service"
    INFERENCE_API_FORMAT: str = NotImplemented
    """The expected format of the inference API. This should be overridden by subclasses."""

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        uri: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        if uri is None:
            raise ValueError("A URI must be provided to connect to the model service.")
        self.inference_uri = uri
        self.organization = organization
        configs = {
            "uri": uri,
            "organization": organization,
            "inference_api_format": self.INFERENCE_API_FORMAT,
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
