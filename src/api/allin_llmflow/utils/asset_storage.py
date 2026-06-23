"""
This module provides utility functions to manage asset storage.
"""

import json
import os
from pathlib import Path
from typing import Union, Any, Dict, Optional

import httpx

ENV_LOCAL_ASSET_STORAGE_PATH = "LOCAL_ASSET_STORAGE_PATH"
ENV_ALLIN_ASSET_STORAGE_TOKEN = "ALLIN_ASSET_STORAGE_TOKEN"
ENV_ALLIN_ASSET_STORAGE_STAGE = "ALLIN_ASSET_STORAGE_STAGE"

_ALLIN_API_BASE_PATH = (
    "https://ai.devops.xiaohongshu.com/api/ai-asset-platform/v1/"
    if os.environ.get(ENV_ALLIN_ASSET_STORAGE_STAGE, "prod") == "prod"
    else "https://ai.devops.sit.xiaohongshu.com/api/ai-asset-platform/v1/"
)
ALLIN_SERVICE_ASSET_STORAGE_API = (
    _ALLIN_API_BASE_PATH + "org/{org}/service-assets/{service_asset}/{asset_version_suffix}"
)


def set_local_asset_storage_path(path: Union[Path, str]) -> None:
    """
    Set the local asset storage path. This path is used to store assets locally.

    :param path: The local asset storage path.
    """
    os.environ[ENV_LOCAL_ASSET_STORAGE_PATH] = str(path)


def get_local_asset_storage_path() -> Path:
    """
    Get the local asset storage path.

    :return: The local asset storage path.
    """
    try:
        return Path(os.environ[ENV_LOCAL_ASSET_STORAGE_PATH])
    except KeyError as err:
        raise ValueError(
            f"Environment variable {ENV_LOCAL_ASSET_STORAGE_PATH} is not set. "
            "Please set it to the local asset storage path via set_local_asset_storage_path()."
        ) from err


def get_local_asset_path(asset_name: str) -> Path:
    """
    Get the local path of the asset file for the given asset name.

    :param asset_name: The name of the asset.
    :return: The path of the asset file in local storage.
    """
    return get_local_asset_storage_path() / f"{asset_name}.json"


def get_local_asset(asset_name: str) -> Dict[str, Any]:
    """
    Get the metadata of the asset file for the given asset name. This function
    also checks if the asset file exists, and raises FileNotFoundError if the asset file is not found.

    :param asset_name: The name of the asset.
    :return: The metadata of the asset file.
    :raises FileNotFoundError: If the asset file is not found.
    """
    asset_path = get_local_asset_path(asset_name)
    if not asset_path.exists():
        raise FileNotFoundError(f"Asset file {asset_path} not found.")
    with open(asset_path, "r", encoding="utf-8") as fr:
        data = json.load(fr)
    return data


def get_allin_asset(asset_name: str, asset_version: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the metadata of the asset file for the given asset name from Allin asset storage.

    :param asset_name: The name of the asset.
    :param asset_version: The version of the asset.
    :return: The metadata of the asset file.
    """
    try:
        org, service_asset = asset_name.split("/")
    except ValueError as err:
        raise ValueError(
            "Invalid asset name format. The asset name should be in the format 'org/service_asset_name'."
        ) from err
    if asset_version:
        asset_version_suffix = f"versions/{asset_version}"
    else:
        asset_version_suffix = "latest-version"
    # Load the asset from the Allin asset storage
    response = (
        httpx.get(
            ALLIN_SERVICE_ASSET_STORAGE_API.format(
                org=org, service_asset=service_asset, asset_version_suffix=asset_version_suffix
            ),
            headers={"QSToken": os.environ.get(ENV_ALLIN_ASSET_STORAGE_TOKEN, "")},
        )
        .raise_for_status()
        .json()
        .get("result")
    )
    data = {
        "name": asset_name,
        "type": response.get("type"),
        "configs": response.get("configs"),
        "secrets": response.get("secrets"),
    }
    return data
