"""
This module contains ByteStreamJoiner, a component that joins multiple lists of byte streams into a single list.
"""

from typing import List

from haystack import component
from haystack.core.component.types import Variadic

from allin_llmflow.dataclasses import ByteStream


@component
class ByteStreamJoiner:
    """
    ByteStreamJoiner is a component that joins multiple lists of byte streams into a single list of byte streams.
    """

    @component.output_types(byte_streams=List[ByteStream])
    def run(self, byte_streams: Variadic[List[ByteStream]]):
        """
        Joins multiple lists of byte streams into a single list of byte streams.

        :param byte_streams: List of list of byte streams to be merged.
        :returns: A dictionary with the following keys:
            - `byte_streams`: Merged list of byte streams.
        """
        return {"byte_streams": [bs for bs_list in byte_streams for bs in bs_list]}
