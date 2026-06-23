# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Data classes for representing binary data in the Haystack API. The ByteStream class can be used to represent binary data
in the API, and can be converted to and from base64 encoded strings, dictionaries, and files. This is particularly
useful for representing media files in chat messages.
"""

import logging
import mimetypes
from base64 import b64encode, b64decode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


@dataclass
class ByteStream:
    """
    Base data class representing a binary object in the Haystack API.
    """

    data: bytes
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)
    mime_type: Optional[str] = field(default=None)

    @property
    def type(self) -> Optional[str]:
        """
        Return the type of the ByteStream. This is the first part of the mime type, or None if the mime type is not set.

        :return: The type of the ByteStream.
        """
        if self.mime_type:
            return self.mime_type.split("/", maxsplit=1)[0]
        return None

    @property
    def subtype(self) -> Optional[str]:
        """
        Return the subtype of the ByteStream. This is the second part of the mime type,
        or None if the mime type is not set.

        :return: The subtype of the ByteStream.
        """
        if self.mime_type:
            return self.mime_type.split("/", maxsplit=1)[-1]
        return None

    def to_file(self, destination_path: Path):
        """
        Write the ByteStream to a file. Note: the metadata will be lost.

        :param destination_path: The path to write the ByteStream to.
        """
        with open(destination_path, "wb") as fd:
            fd.write(self.data)

    @classmethod
    def from_file_path(
        cls, filepath: Path, mime_type: Optional[str] = None, meta: Optional[Dict[str, Any]] = None
    ) -> "ByteStream":
        """
        Create a ByteStream from the contents read from a file.

        :param filepath: A valid path to a file.
        :param mime_type: The mime type of the file.
        :param meta: Additional metadata to be stored with the ByteStream.
        """
        if mime_type is None:
            mime_type = mimetypes.guess_type(filepath)[0]
            if mime_type is None:
                logger.warning("Could not determine mime type for file %s", filepath)

        with open(filepath, "rb") as fd:
            return cls(data=fd.read(), mime_type=mime_type, meta=meta or {})

    @classmethod
    def from_string(
        cls, text: str, encoding: str = "utf-8", mime_type: Optional[str] = None, meta: Optional[Dict[str, Any]] = None
    ) -> "ByteStream":
        """
        Create a ByteStream encoding a string.

        :param text: The string to encode
        :param encoding: The encoding used to convert the string into bytes
        :param mime_type: The mime type of the file.
        :param meta: Additional metadata to be stored with the ByteStream.
        """
        return cls(data=text.encode(encoding), mime_type=mime_type, meta=meta or {})

    def to_string(self, encoding: str = "utf-8") -> str:
        """
        Convert the ByteStream to a string, metadata will not be included.

        :param encoding: The encoding used to convert the bytes to a string. Defaults to "utf-8".
        :returns: The string representation of the ByteStream.
        :raises: UnicodeDecodeError: If the ByteStream data cannot be decoded with the specified encoding.
        """
        return self.data.decode(encoding)

    @classmethod
    def from_base64(
        cls,
        base64_string: str,
        encoding: str = "utf-8",
        meta: Optional[Dict[str, Any]] = None,
        mime_type: Optional[str] = None,
    ) -> "ByteStream":
        """
        Create a ByteStream from a base64 encoded string.

        :param base64_string: The base64 encoded string representation of the ByteStream data.
        :param encoding: The encoding used to convert the base64 string into bytes.
        :param meta: Additional metadata to be stored with the ByteStream.
        :param mime_type: The mime type of the file.
        :returns: A new ByteStream instance.
        """
        return cls(data=b64decode(base64_string.encode(encoding)), meta=meta or {}, mime_type=mime_type)

    def to_base64(self, encoding: str = "utf-8") -> str:
        """
        Convert the ByteStream data to a base64 encoded string.

        :returns: The base64 encoded string representation of the ByteStream data.
        """
        return b64encode(self.data).decode(encoding)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], encoding: str = "utf-8") -> "ByteStream":
        """
        Create a ByteStream from a dictionary.

        :param data: The dictionary representation of the ByteStream.
        :param encoding: The encoding used to convert the base64 string into bytes.
        :returns: A new ByteStream instance.
        """
        return cls.from_base64(data["data"], encoding=encoding, meta=data.get("meta"), mime_type=data.get("mime_type"))

    def to_dict(self, encoding: str = "utf-8"):
        """
        Convert the ByteStream to a dictionary.

        :param encoding: The encoding used to convert the bytes to a string. Defaults to "utf-8".
        :returns: The dictionary representation of the ByteStream.
        :raises: UnicodeDecodeError: If the ByteStream data cannot be decoded with the specified encoding.
        """
        return {"data": self.to_base64(encoding=encoding), "meta": self.meta, "mime_type": self.mime_type}
