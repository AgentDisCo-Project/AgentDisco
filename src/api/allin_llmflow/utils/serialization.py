"""
This module provides utility functions for serializing and deserializing data into json.
"""

import dataclasses
from typing import Any, Dict


def default_json_serializer(obj: object) -> Dict[str, Any]:
    """
    Default JSON serializer for objects that are not natively serializable. This function will attempt to serialize
    objects that have a `to_dict` method or are dataclasses.

    :param obj: The object to serialize.
    :return: The serialized object.
    """
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
