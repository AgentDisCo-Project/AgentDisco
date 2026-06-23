"""
This module contains factory classes for creating and loading assets.

The assets can be of different types such as ModelService, VectorDatabase and EmbeddingFunction. Each asset type has
its own factory class (ModelServiceFactory, VectorDatabaseFactory and EmbeddingFunctionFactory respectively) that is
responsible for creating and loading instances of that asset type.

The AssetFactory class is a general factory class that can create and load any type of asset. It uses the specific
factory classes for each asset type to do the actual creation and loading.

This module also provides a way to register new types of ModelService, VectorDatabase and EmbeddingFunction assets,
so that they can be created and loaded by the factory classes.
"""

import abc
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from allin_llmflow.assets import Asset
from allin_llmflow.assets.embedding_functions import EmbeddingFunction, SUPPORTED_EMBEDDING_FUNCTIONS
from allin_llmflow.assets.model_services import ModelService, SUPPORTED_INFERENCE_API_FORMATS
from allin_llmflow.assets.tool_services import ToolService, SUPPORTED_CALL_API_FORMATS
from allin_llmflow.assets.vector_databases import VectorDatabase, SUPPORTED_VECTORDB_TYPES
from allin_llmflow.dataclasses.asset_reference import AssetReference, Source

AssetT = TypeVar("AssetT", bound=Asset)


def camel_to_hyphen(s: str) -> str:
    """
    Convert an upper camel case string to a hyphenated string. This is to ensure backward compatibility with the changes
    made to asset type names.

    :param s: The camel case string.
    :return: The hyphenated string.
    """
    return "".join(["-" + c.lower() if c.isupper() else c for c in s]).lstrip("-")


class _BaseFactory(Generic[AssetT], metaclass=abc.ABCMeta):
    """
    Base factory class for creating assets.
    """

    @classmethod
    @abc.abstractmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> AssetT:
        """
        Load an asset from a dictionary.

        :param data: The dictionary containing the asset.
        :return: An asset.
        :raises ValueError: If the asset type is not supported.
        """

    @classmethod
    def load_from_local(cls, asset_name: str) -> AssetT:
        """
        Load an asset from the local storage.

        :param asset_name: The name of the asset to load.
        :return: An asset.
        """
        return cls.load_from_reference(AssetReference(name=asset_name, source=Source.LOCAL))

    @classmethod
    def load_from_allin(cls, asset_name: str, asset_version: Optional[str] = None) -> AssetT:
        """
        Load an asset from the Allin asset storage.

        :param asset_name: The name of the asset to load.
        :param asset_version: The version of the asset to load.
        :return: An asset.
        """
        return cls.load_from_reference(AssetReference(name=asset_name, version=asset_version, source=Source.ALLIN))

    @classmethod
    def load_from_reference(cls, asset_reference: AssetReference) -> AssetT:
        """
        Load an asset from storage using an asset reference.

        :param asset_reference: The asset reference to the asset.
        :return: An asset.
        """
        asset = cls.load_from_dict(asset_reference.load())
        asset.reference = asset_reference
        return asset


class ModelServiceFactory(_BaseFactory[ModelService]):
    """
    Factory class to create ModelService assets.

    :param SUPPORTED_MODEL_SERVICES: A dictionary containing the supported model services.
    """

    SUPPORTED_MODEL_SERVICES: Dict[str, Type[ModelService]] = SUPPORTED_INFERENCE_API_FORMATS

    @classmethod
    def register_model_service(cls, model_service: Type[ModelService]) -> None:
        """
        Register a model service with the factory.

        :param model_service: The ModelService class to register.
        """
        cls.SUPPORTED_MODEL_SERVICES[model_service.INFERENCE_API_FORMAT] = model_service

    @classmethod
    def create(cls, inference_api_format: str, **kwargs) -> ModelService:
        """
        Create a ModelService asset from the given inference API format.

        :param inference_api_format: The inference API format of the model service.
            It must be one of the supported formats, or a ValueError will be raised.
        :param kwargs: Additional keyword arguments to initialize the model service.
        :return: A ModelService asset.
        :raises ValueError: If the inference API format is not supported.
        """
        model_service_cls = cls.SUPPORTED_MODEL_SERVICES.get(inference_api_format)
        if not model_service_cls:
            raise ValueError(f"Unsupported inference_api_format: {inference_api_format}")
        return model_service_cls(**kwargs)

    @classmethod
    def load_from_dict(cls, data: dict) -> ModelService:
        """
        Load a ModelService asset from a dictionary.

        :param data: The dictionary containing the ModelService asset. It must contain an "inference_api_format" key
            and the key must be one of the supported formats, or a ValueError will be raised.
        :return: A ModelService asset.
        :raises ValueError: If the dictionary does not represent a ModelService asset, or the inference API format is
            not supported.
        """
        if camel_to_hyphen(data.get("type", "")) != "model-service":
            raise ValueError("The dictionary does not represent a ModelService asset.")
        try:
            configs_data = data.get("configs", {}).copy()
            inference_api_format = configs_data.pop("inference_api_format")
        except KeyError as err:
            raise ValueError("Missing 'inference_api_format' key in the dictionary") from err
        model_service_cls = cls.SUPPORTED_MODEL_SERVICES.get(inference_api_format)
        if not model_service_cls:
            raise ValueError(f"Unsupported inference_api_format: {inference_api_format}")
        return model_service_cls.from_dict(
            {
                "name": data.get("name"),
                "configs": configs_data,
                "secrets": data.get("secrets", {}),
            }
        )


class ToolServiceFactory(_BaseFactory[ToolService]):
    """
    Factory class to create ToolService assets.

    :param SUPPORTED_TOOL_SERVICES: A dictionary containing the supported tool services.
    """

    SUPPORTED_TOOL_SERVICES: Dict[str, Type[ToolService]] = SUPPORTED_CALL_API_FORMATS

    @classmethod
    def register_tool_service(cls, tool_service: Type[ToolService]) -> None:
        """
        Register a model service with the factory.

        :param tool_service: The ToolService class to register.
        """
        cls.SUPPORTED_TOOL_SERVICES[tool_service.CALL_API_FORMAT] = tool_service

    @classmethod
    def create(cls, call_api_format: str, **kwargs) -> ToolService:
        """
        Create a ToolService asset from the given call API format.

        :param call_api_format: The call API format of the tool service.
            It must be one of the supported formats, or a ValueError will be raised.
        :param kwargs: Additional keyword arguments to initialize the tool service.
        :return: A ToolService asset.
        :raises ValueError: If the call API format is not supported.
        """
        tool_service_cls = cls.SUPPORTED_TOOL_SERVICES.get(call_api_format)
        if not tool_service_cls:
            raise ValueError(f"Unsupported call_api_format: {call_api_format}")
        return tool_service_cls(**kwargs)

    @classmethod
    def load_from_dict(cls, data: dict) -> ToolService:
        """
        Load a ToolService asset from a dictionary.

        :param data: The dictionary containing the ToolService asset. It must contain a "call_api_format" key
            and the key must be one of the supported formats, or a ValueError will be raised.
        :return: A ToolService asset.
        :raises ValueError: If the dictionary does not represent a ToolService asset, or the inference API format is
            not supported.
        """
        if camel_to_hyphen(data.get("type", "")) != "tool-service":
            raise ValueError("The dictionary does not represent a ToolService asset.")
        try:
            configs_data = data.get("configs", {}).copy()
            call_api_format = configs_data.pop("call_api_format")
        except KeyError as err:
            raise ValueError("Missing 'call_api_format' key in the dictionary") from err
        tool_service_cls = cls.SUPPORTED_TOOL_SERVICES.get(call_api_format)
        if not tool_service_cls:
            raise ValueError(f"Unsupported call_api_format: {call_api_format}")
        return tool_service_cls.from_dict(
            {
                "name": data.get("name"),
                "configs": configs_data,
                "secrets": data.get("secrets", {}),
            }
        )


class VectorDatabaseFactory(_BaseFactory[VectorDatabase]):
    """
    Factory class to create VectorDatabase assets.

    :param SUPPORTED_VECTOR_DATABASES: A dictionary containing the supported vector databases.
    """

    SUPPORTED_VECTOR_DATABASES = SUPPORTED_VECTORDB_TYPES

    @classmethod
    def register_vector_database(cls, vector_database: Type[VectorDatabase]) -> None:
        """
        Register a vector database with the factory.
        :param vector_database: The VectorDatabase class to register.
        """
        cls.SUPPORTED_VECTOR_DATABASES[vector_database.VECTOR_DB_TYPE] = vector_database

    @classmethod
    def create(cls, vector_db_type: str, **kwargs) -> VectorDatabase:
        """
        Create a VectorDatabase asset from the given vector database type.

        :param vector_db_type: The type of the vector database.
            It must be one of the supported types, or a ValueError will be raised.
        :param kwargs: Additional keyword arguments to initialize the vector database.
        :return: A VectorDatabase asset.
        :raises ValueError: If the vector database type is not supported.
        """
        vector_database_cls = cls.SUPPORTED_VECTOR_DATABASES.get(vector_db_type)
        if not vector_database_cls:
            raise ValueError(f"Unsupported vector_db_type: {vector_db_type}")
        return vector_database_cls(**kwargs)

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> VectorDatabase:
        """
        Load a VectorDatabase asset from a dictionary.

        :param data: The dictionary containing the VectorDatabase asset. It must contain a "vector_db_type" key
            and the key must be one of the supported types, or a ValueError will be raised.
        :return: A VectorDatabase asset.
        :raises ValueError: If the vector database type is not supported.
        """
        if camel_to_hyphen(data.get("type", "")) != "vector-database":
            raise ValueError("The dictionary does not represent a VectorDatabase asset.")
        try:
            configs_data = data.get("configs", {}).copy()
            vector_db_type = configs_data.pop("vector_db_type")
        except KeyError as err:
            raise ValueError("Missing 'vector_db_type' key in the dictionary") from err
        vector_database_cls = cls.SUPPORTED_VECTOR_DATABASES.get(vector_db_type)
        if not vector_database_cls:
            raise ValueError(f"Unsupported vector_db_type: {vector_db_type}")
        return vector_database_cls.from_dict(
            {
                "name": data.get("name"),
                "configs": configs_data,
                "secrets": data.get("secrets", {}),
            }
        )


class EmbeddingFunctionFactory(_BaseFactory[EmbeddingFunction]):
    """
    Factory class to create EmbeddingFunction assets.
    """

    SUPPORTED_EMBEDDING_FUNCTIONS = SUPPORTED_EMBEDDING_FUNCTIONS

    @classmethod
    def register_embedding_function(cls, embedding_function: Type[EmbeddingFunction]) -> None:
        """
        Register an embedding function with the factory.

        :param embedding_function: The EmbeddingFunction class to register
        """
        cls.SUPPORTED_EMBEDDING_FUNCTIONS[embedding_function.EMBEDDING_STRATEGY] = embedding_function

    @classmethod
    def create(cls, embedding_strategy: str, **kwargs) -> EmbeddingFunction:
        """
        Create an EmbeddingFunction asset from the given embedding strategy.

        :param embedding_strategy: The strategy of the embedding function.
            It must be one of the supported strategies, or a ValueError will be raised.
        :param kwargs: Additional keyword arguments to initialize the embedding function.
        :return: An EmbeddingFunction asset.
        :raises ValueError: If the embedding strategy is not supported.
        """
        embedding_function_cls = cls.SUPPORTED_EMBEDDING_FUNCTIONS.get(embedding_strategy)
        if not embedding_function_cls:
            raise ValueError(f"Unsupported embedding_strategy: {embedding_strategy}")
        return embedding_function_cls(**kwargs)

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> EmbeddingFunction:
        """
        Load an EmbeddingFunction asset from a dictionary.

        :param data: The dictionary containing the EmbeddingFunction asset. It must contain an "embedding_strategy" key
            and the key must be one of the supported strategies, or a ValueError will be raised.
        :return: An EmbeddingFunction asset.
        :raises ValueError: If the embedding strategy is not supported.
        """
        if camel_to_hyphen(data.get("type", "")) != "embedding-function":
            raise ValueError("The dictionary does not represent an EmbeddingFunction asset.")
        try:
            configs_data = data.get("configs", {}).copy()
            embedding_strategy = configs_data.pop("embedding_strategy")
        except KeyError as err:
            raise ValueError("Missing 'embedding_strategy' key in the dictionary") from err
        embedding_function_cls = cls.SUPPORTED_EMBEDDING_FUNCTIONS.get(embedding_strategy)
        if not embedding_function_cls:
            raise ValueError(f"Unsupported embedding_strategy: {embedding_strategy}")
        return embedding_function_cls.from_dict(
            {
                "name": data.get("name"),
                "configs": configs_data,
                "secrets": data.get("secrets", {}),
            }
        )


class AssetFactory(_BaseFactory[Asset]):
    """
    Factory class to create assets.
    """

    @staticmethod
    def create(asset_type: str, **kwargs) -> Asset:
        """
        Create an asset from the given type.

        :param asset_type: The type of the asset to create.
        :param kwargs: Additional keyword arguments to initialize the asset.
        :return: An asset.
        :raises ValueError: If the asset type is not supported.
        """
        if camel_to_hyphen(asset_type) == "model-service":
            return ModelServiceFactory.create(**kwargs)
        if camel_to_hyphen(asset_type) == "tool-service":
            return ToolServiceFactory.create(**kwargs)
        if camel_to_hyphen(asset_type) == "vector-database":
            return VectorDatabaseFactory.create(**kwargs)
        if camel_to_hyphen(asset_type) == "embedding-function":
            return EmbeddingFunctionFactory.create(**kwargs)
        # Add more conditions here to handle future types of assets
        raise ValueError(f"Unsupported asset_type: {asset_type}")

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> Asset:
        """
        Load an asset from a dictionary.

        :param data: The dictionary containing the asset.
        :return: An asset.
        :raises ValueError: If the asset type is not supported.
        """
        asset_type = data.get("type", "")
        if camel_to_hyphen(asset_type) == "model-service":
            return ModelServiceFactory.load_from_dict(data)
        if camel_to_hyphen(asset_type) == "tool-service":
            return ToolServiceFactory.load_from_dict(data)
        if camel_to_hyphen(asset_type) == "vector-database":
            return VectorDatabaseFactory.load_from_dict(data)
        if camel_to_hyphen(asset_type) == "embedding-function":
            return EmbeddingFunctionFactory.load_from_dict(data)
        # Add more conditions here to handle future types of assets
        raise ValueError(f"Unsupported asset_type: {asset_type}")
