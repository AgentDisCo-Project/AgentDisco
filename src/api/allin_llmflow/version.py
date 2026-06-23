"""
This module contains the version of the SDK.
"""

from importlib import metadata

try:
    __version__ = str(metadata.version("allin-llmflow"))
except metadata.PackageNotFoundError:
    __version__ = "main"
