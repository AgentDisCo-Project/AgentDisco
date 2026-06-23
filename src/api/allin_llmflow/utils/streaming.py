"""
This module provides classes for streaming Server-Sent Events (SSE) from an HTTP response.
"""

# Note: initially copied from https://github.com/openai/openai-python/blob/main/src/openai/_streaming.py

import json
from types import TracebackType
from typing import Any, AsyncIterator, Iterator, Mapping, Optional, Callable

import httpx
from typing_extensions import override, Self


class ServerSentEvent:
    """
    Represents a single Server-Sent Event (SSE). This class is used to parse and represent the data from an SSE stream.
    """

    def __init__(
        self,
        *,
        event: str | None = None,
        data: str | None = None,
        id: str | None = None,  # pylint: disable=redefined-builtin
        retry: int | None = None,
    ) -> None:
        if data is None:
            data = ""

        self._id = id
        self._data = data
        self._event = event or None
        self._retry = retry

    @property
    def event(self) -> str | None:
        """
        The event name of the SSE.

        :return: The event name.
        """
        return self._event

    @property
    def id(self) -> str | None:
        """
        The event ID of the SSE.

        :return: The event ID.
        """
        return self._id

    @property
    def retry(self) -> int | None:
        """
        The retry time of the SSE.

        :return: The retry time.
        """
        return self._retry

    @property
    def data(self) -> str:
        """
        The data of the SSE in string format.

        :return: The data string.
        """
        return self._data

    def json(self) -> Any:
        """
        Parse the data as JSON.

        :return: The parsed JSON data.
        """
        try:
            return json.loads(self.data)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse SSE data as JSON: {self.data}") from err

    @override
    def __repr__(self) -> str:
        return f"ServerSentEvent(event={self.event}, data={self.data}, id={self.id}, retry={self.retry})"


class SSEDecoder:
    """
    A decoder class for parsing raw binary data into Server-Sent Events (SSE). This class is used to decode raw binary
    data from an HTTP response into ServerSentEvent objects.
    """

    _data: list[str]
    _event: str | None
    _retry: int | None
    _last_event_id: str | None

    def __init__(self) -> None:
        self._event = None
        self._data = []
        self._last_event_id = None
        self._retry = None

    def iter_bytes(self, iterator: Iterator[bytes]) -> Iterator[ServerSentEvent]:
        """Given an iterator that yields raw binary data, iterate over it & yield every event encountered"""
        for chunk in self._iter_chunks(iterator):
            # Split before decoding so splitlines() only uses \r and \n
            for raw_line in chunk.splitlines():
                line = raw_line.decode("utf-8")
                sse = self.decode(line)
                if sse:
                    yield sse

    def _iter_chunks(self, iterator: Iterator[bytes]) -> Iterator[bytes]:
        """Given an iterator that yields raw binary data, iterate over it and yield individual SSE chunks"""
        data = b""
        for chunk in iterator:
            for line in chunk.splitlines(keepends=True):
                data += line
                if data.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                    yield data
                    data = b""
        if data:
            yield data

    async def aiter_bytes(self, iterator: AsyncIterator[bytes]) -> AsyncIterator[ServerSentEvent]:
        """Given an iterator that yields raw binary data, iterate over it & yield every event encountered"""
        async for chunk in self._aiter_chunks(iterator):
            # Split before decoding so splitlines() only uses \r and \n
            for raw_line in chunk.splitlines():
                line = raw_line.decode("utf-8")
                sse = self.decode(line)
                if sse:
                    yield sse

    async def _aiter_chunks(self, iterator: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Given an iterator that yields raw binary data, iterate over it and yield individual SSE chunks"""
        data = b""
        async for chunk in iterator:
            for line in chunk.splitlines(keepends=True):
                data += line
                if data.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                    yield data
                    data = b""
        if data:
            yield data

    def decode(self, line: str) -> ServerSentEvent | None:
        """
        Decode a single line of an SSE stream.

        :param line: The line to decode.
        :return: The decoded ServerSentEvent, or None if the line is part of a multi-line event.
        """
        # See: https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation  # noqa: E501

        if not line:
            if not self._event and not self._data and not self._last_event_id and self._retry is None:
                return None

            sse = ServerSentEvent(
                event=self._event,
                data="\n".join(self._data),
                id=self._last_event_id,
                retry=self._retry,
            )

            # NOTE: as per the SSE spec, do not reset last_event_id.
            self._event = None
            self._data = []
            self._retry = None

            return sse

        if line.startswith(":"):
            return None

        fieldname, _, value = line.partition(":")

        if value.startswith(" "):
            value = value[1:]

        if fieldname == "event":
            self._event = value
        elif fieldname == "data":
            self._data.append(value)
        elif fieldname == "id":
            if "\0" in value:
                pass
            else:
                self._last_event_id = value
        elif fieldname == "retry":
            try:
                self._retry = int(value)
            except (TypeError, ValueError):
                pass
        else:
            pass  # Field is ignored.

        return None


class ServerSentEventStream:
    """
    Provides an iterator interface for consuming Server-Sent Events (SSE) from an HTTP response. This class assumes
    that the response is an SSE stream, and by default it follows OpenAI's convention of using "data:[DONE]" to signal
    the end of the stream unless a custom end indicator is provided.

    :param response: The HTTP response object.
    :param decoder: The SSE decoder to use. Defaults to `SSEDecoder`.
    """

    response: httpx.Response

    _decoder: SSEDecoder

    def __init__(
        self,
        *,
        response: httpx.Response,
        decoder: Optional[SSEDecoder] = None,
        end_indicator: Callable[[ServerSentEvent], bool] = lambda sse: sse.data.startswith("[DONE]"),
    ) -> None:
        self.response = response
        self._decoder = decoder or SSEDecoder()
        self._iterator = self.__stream__()
        self._end_indicator = end_indicator

    def __next__(self) -> ServerSentEvent:
        """
        Get the next event from the stream.

        :return: The next ServerSentEvent.
        """
        return self._iterator.__next__()

    def __iter__(self) -> Iterator[ServerSentEvent]:
        """
        Iterate over the stream of ServerSentEvents.

        :return: An iterator of ServerSentEvents.
        """
        yield from self._iterator

    def _iter_events(self) -> Iterator[ServerSentEvent]:
        """
        Iterate over the stream of raw ServerSentEvents, before checking "[DONE]" signals and handling errors. This
        method is used internally by the `__stream__` method to handle the stream.

        :return: An iterator of ServerSentEvents.
        """
        yield from self._decoder.iter_bytes(self.response.iter_bytes())

    def __stream__(self) -> Iterator[ServerSentEvent]:
        """
        Stream the ServerSentEvents from the response.

        :return: An iterator of ServerSentEvents.
        """
        iterator = self._iter_events()

        for sse in iterator:
            if self._end_indicator(sse):
                break

            try:
                data = sse.json()
            except ValueError:
                data = sse.data
            if (
                (sse.event is None or sse.event == "error")
                and isinstance(data, Mapping)
                and (error := data.get("error")) is not None
            ):
                message = None
                if isinstance(error, Mapping):
                    message = error.get("message")
                if not message or not isinstance(message, str):
                    message = "error message not found"

                raise httpx.StreamError(
                    message=f"An error occurred during streaming: {message}, request={self.response.request}",
                )

            yield sse

        # Ensure the entire stream is consumed
        for _sse in iterator:
            ...

    def __enter__(self) -> Self:
        """
        Enter the context manager.

        :return: The ServerSentEventStream instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Exit the context manager.

        :param exc_type: The exception type if an exception occurred in the context.
        :param exc: The exception instance if an exception occurred in the context.
        :param exc_tb: The traceback if an exception occurred in the context.
        """
        self.close()

    def close(self) -> None:
        """
        Close the response and release the connection.
        Automatically called if the response body is read to completion.
        """
        self.response.close()


class AsyncServerSentEventStream:
    """
    Provides an asynchronous iterator interface for consuming Server-Sent Events (SSE) from an HTTP response.
    This class assumes that the response is an SSE stream, and by default it follows OpenAI's convention of using
    "data:[DONE]" to signal the end of the stream unless a custom end indicator is provided.

    :param response: The HTTP response object.
    :param decoder: The SSE decoder to use. Defaults to `SSEDecoder`.
    """

    response: httpx.Response
    _decoder: SSEDecoder

    def __init__(
        self,
        *,
        response: httpx.Response,
        decoder: Optional[SSEDecoder] = None,
        end_indicator: Callable[[ServerSentEvent], bool] = lambda sse: sse.data.startswith("[DONE]"),
    ) -> None:
        self.response = response
        self._decoder = decoder or SSEDecoder()
        self._iterator = self.__stream__()
        self._end_indicator = end_indicator

    async def __anext__(self) -> ServerSentEvent:
        """
        Get the next event from the stream.

        :return: The next ServerSentEvent.
        """
        return await self._iterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[ServerSentEvent]:
        """
        Iterate over the stream of ServerSentEvents.

        :return: An iterator of ServerSentEvents.
        """
        async for sse in self._iterator:
            yield sse

    async def _iter_events(self) -> AsyncIterator[ServerSentEvent]:
        """
        Iterate over the stream of raw ServerSentEvents, before checking "[DONE]" signals and handling errors. This
        method is used internally by the `__stream__` method to handle the stream.

        :return: An iterator of ServerSentEvents.
        """
        async for sse in self._decoder.aiter_bytes(self.response.aiter_bytes()):
            yield sse

    async def __stream__(self) -> AsyncIterator[ServerSentEvent]:
        """
        Stream the ServerSentEvents from the response.

        :return: An iterator of ServerSentEvents.
        """
        iterator = self._iter_events()

        async for sse in iterator:
            if self._end_indicator(sse):
                break

            try:
                data = sse.json()
            except ValueError:
                data = sse.data
            if (
                (sse.event is None or sse.event == "error")
                and isinstance(data, Mapping)
                and (error := data.get("error")) is not None
            ):
                message = None
                if isinstance(error, Mapping):
                    message = error.get("message")
                if not message or not isinstance(message, str):
                    message = "error message not found"

                raise httpx.StreamError(
                    message=f"An error occurred during streaming: {message}, request={self.response.request}",
                )

            yield sse

        # Ensure the entire stream is consumed
        async for _sse in iterator:
            ...

    async def __aenter__(self) -> Self:
        """
        Enter the context manager.

        :return: The ServerSentEventStream instance.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Exit the context manager.

        :param exc_type: The exception type if an exception occurred in the context.
        :param exc: The exception instance if an exception occurred in the context.
        :param exc_tb: The traceback if an exception occurred in the context.
        """
        await self.close()

    async def close(self) -> None:
        """
        Close the response and release the connection.
        Automatically called if the response body is read to completion.
        """
        await self.response.aclose()
