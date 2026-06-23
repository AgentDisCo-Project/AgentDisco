"""
This module initializes the `allin_llmflow` package and sets up the environment.
"""

import os

from allin_llmflow.version import __version__

# Disable Haystack telemetry by setting the environment variable
# See https://docs.haystack.deepset.ai/docs/telemetry for more information
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"

__all__ = ["__version__"]
