"""
This file contains the dataclass for the StreamingChunk object. This object is used to represent a chunk of streaming
data from a model service. It is inherited from the StreamingChunk object in the Haystack library, with an additional
field for tool calls.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from haystack.dataclasses import StreamingChunk as _StreamingChunk


@dataclass
class StreamingChunk(_StreamingChunk):
    """
    Data class representing a chunk of streaming data from a model service.
    """

    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
