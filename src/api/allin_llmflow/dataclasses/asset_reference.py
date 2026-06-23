"""
This module contains the dataclass for AssetReference.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from typing_extensions import Self

from allin_llmflow.utils.asset_storage import get_local_asset, get_allin_asset


class Source(str, Enum):
    """
    An enumeration of possible sources of an asset.

    :param LOCAL: The asset is stored locally.
    """

    LOCAL = "local"
    ALLIN = "allin"


@dataclass
class AssetReference:
    """
    A dataclass that represents a hashable reference to an asset. This provides a unified way to identify and
    retrieve an asset from the asset registry, either locally or remotely.

    :param name: The name of the asset.
    :param source: The source of the asset.
    """

    name: str
    source: Source = Source.LOCAL
    version: Optional[str] = None
    # Add versioning information for remote assets

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the asset reference to a dictionary.

        :return: A dictionary representation of the asset reference.
        """
        return {"name": self.name, "source": self.source.value, "version": self.version}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create an AssetReference instance from a dictionary.

        :param data: A dictionary representation of the asset reference.
        :return: An instance of AssetReference.
        """
        return cls(name=data["name"], source=Source(data["source"]), version=data.get("version"))

    def load(self) -> Dict[str, Any]:
        """
        Load the asset from the asset registry.

        :return: The loaded asset.
        """
        # Load the asset from the asset registry based on the source
        if self.source == Source.LOCAL:
            # Load the asset from the local asset storage
            return get_local_asset(self.name)
        if self.source == Source.ALLIN:
            # Load the asset from the Allin asset storage
            return get_allin_asset(self.name, self.version)
        # Raise an error if the source is not supported
        raise ValueError(f"Unsupported asset source: {self.source}")
