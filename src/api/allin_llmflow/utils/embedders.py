"""
This module contains the utility functions to load the embedder based on the given configuration.
"""

import copy
import tempfile
from functools import partial
from typing import Dict, Any, Optional, Callable, List

from allin_llmflow.assets import Asset
from allin_llmflow.assets.asset_factory import AssetFactory
from allin_llmflow.assets.embedding_functions import EmbeddingFunction
from allin_llmflow.assets.model_services import EmbeddingModelService
from allin_llmflow.dataclasses.asset_reference import AssetReference


def load_embedder(
    from_asset: AssetReference, embedder_kwargs: Optional[Dict[str, Any]] = None
) -> Callable[[str], List[List[float]]]:
    """
    Load the embedder based on the given configuration.

    This function checks the type of embedder specified in the configuration and
    calls the appropriate function to load the embedder.

    :param from_asset: The reference of the asset (an EmbeddingModelService or an EmbeddingFunction) from which
        the embeddings are generated.
    :param embedder_kwargs: The parameters for the embedder.
    :return: The loaded embedder function.
    """
    embedding_asset = AssetFactory.load_from_reference(from_asset)
    return load_embedder_from_assets(embedding_asset, embedder_kwargs)


def load_embedder_from_assets(
    embedding_asset: Asset,
    embedder_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[[str], List[List[float]]]:
    """
    Load the embedder from the given asset.

    :param embedding_asset: The asset from which the embeddings are coming.
    :param embedder_kwargs: The parameters for the embedder.
    :return: The loaded embedder function.
    :raises ValueError: If the asset type is invalid, i.e. not an EmbeddingFunction or an EmbeddingModelService.
    """
    embedder_kwargs = embedder_kwargs or {}
    embedder_kwargs = copy.deepcopy(embedder_kwargs)
    if isinstance(embedding_asset, EmbeddingFunction):
        with tempfile.NamedTemporaryFile() as tmp_file:
            embedding_asset.load_and_init_embedder(local_path=tmp_file.name, embedder_kwargs=embedder_kwargs)
        return embedding_asset.encode_query
    if isinstance(embedding_asset, EmbeddingModelService):
        return partial(
            embedding_asset.infer,
            timeout=embedder_kwargs.pop("timeout", None),
            model=embedder_kwargs.pop("model", None),
            inference_kwargs=embedder_kwargs,
        )

    raise ValueError(f"Invalid type for embedding asset {embedding_asset.name}: {type(embedding_asset)}")
