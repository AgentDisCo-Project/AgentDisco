"""
This module provides helper functions and classes for handling mesh configurations and mesh URIs.
"""

import json

import logging
import re
from typing import Dict

MESH_CONFIG_FILE_PATH = "/etc/mesh/mesh-config.json"
MESH_URI_PATTERN = r"^mesh://(?P<service_name>[a-zA-Z0-9-]+)/?(?P<api_path>.*)$"


class MeshConfigHandler:
    """
    A class to handle mesh configuration.

    :param mesh_config_path: The path to the mesh configuration file.
    """

    def __init__(self, mesh_config_path: str = MESH_CONFIG_FILE_PATH):
        self.mesh_config_path = mesh_config_path
        # Load mesh config lazily to avoid unnecessary file reads if not used.
        self.mesh_service_config: Dict[str, str] = {}
        self._mesh_config_loaded = False

    def load_mesh_config(self):
        """
        Load the mesh configuration file.
        """
        logging.info("loading mesh config file: %s", self.mesh_config_path)
        with open(self.mesh_config_path, "r", encoding="utf-8") as f:
            mesh_config = json.load(f)
        self.mesh_service_config = dict(mesh_config["dependServices"])
        self._mesh_config_loaded = True

    def get_mesh_service_base_uri(self, mesh_service_name: str) -> str:
        """
        Retrieve the base url of the mesh service from mesh configuration. This method also loads the mesh configuration
        if it has not been loaded yet.

        :param mesh_service_name: The name of the mesh service.
        :returns: The base url of the mesh service.
        :raises ValueError: If the service name is not found in the mesh configuration.
        """
        if not self._mesh_config_loaded:
            self.load_mesh_config()
        port = self.mesh_service_config.get(mesh_service_name)
        if port is None:
            raise ValueError(f"Service name {mesh_service_name} not found in mesh configuration.")
        return f"http://127.0.0.1:{port}"


mesh_config_handler = MeshConfigHandler()


def get_inference_uri_from_mesh(mesh_uri: str) -> str:
    """
    Get the real http inference uri from the mesh service configuration.

    :param mesh_uri: The mesh uri of the service.
    :returns: The real inference uri.
    :raises ValueError: If the mesh uri is invalid.
    """
    match = re.search(MESH_URI_PATTERN, mesh_uri)
    if not match:
        raise ValueError(f"Invalid mesh uri: {mesh_uri}")
    service_name, api_path = match.groups()
    return f"{mesh_config_handler.get_mesh_service_base_uri(service_name)}/{api_path}"
