"""
This module contains the StreamEventCallback protocol. The StreamEventCallback protocol is used to define the signature
of a callback function that handles streaming events.
"""

from typing import Any, Protocol


class StreamEventCallback(Protocol):
    """
    A protocol for a callback function that handles streaming events.
    """

    def __call__(self, data: Any, event_type: str): ...
