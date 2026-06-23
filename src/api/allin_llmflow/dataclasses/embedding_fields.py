"""
This module contains the dataclass for EmbeddingFields.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from typing_extensions import Self

from allin_llmflow.dataclasses.asset_reference import AssetReference


@dataclass
class EmbeddingField:
    """
    A dataclass that represents the embedder and search metadata related to embedding fields in the vector database.

    :param from_asset: The asset (a ModelService or an EmbeddingFunction) from which the embeddings are coming.
    :param embedder_kwargs: Additional arguments for the embedder.
    :param search_kwargs: Additional arguments for vector search.
    """

    from_asset: AssetReference
    embedder_kwargs: Optional[Dict[str, Any]] = None
    search_kwargs: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a serialized dictionary representation of the dataclass.

        :return: A dictionary representation of the dataclass.
        """
        return {
            "from_asset": self.from_asset.to_dict(),
            "embedder_kwargs": self.embedder_kwargs,
            "search_kwargs": self.search_kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Creates an instance of the dataclass from a dictionary.

        :param data: A dictionary containing the dataclass attributes.
        :return: An instance of the dataclass.
        """
        return cls(
            from_asset=AssetReference.from_dict(data["from_asset"]),
            embedder_kwargs=data.get("embedder_kwargs"),
            search_kwargs=data.get("search_kwargs"),
        )
