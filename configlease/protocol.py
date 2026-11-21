"""JSON-lines wire protocol over 127.0.0.1 TCP.

Each message is a single JSON object terminated by ``\\n`` and encoded UTF-8.
The module only owns (de)serialisation; node and platform never call sockets
directly.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

MAX_LINE_BYTES = 1 << 20  # 1 MiB safety bound per JSON line.


def encode_message(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def decode_message(raw: bytes) -> dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


async def write_message(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    writer.write(encode_message(payload))
    await writer.drain()


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    raw = await reader.readline()
    if not raw:
        raise ConnectionError("peer closed the connection")
    if len(raw) > MAX_LINE_BYTES:
        raise ValueError("message exceeds the JSON line size bound")
    return decode_message(raw)


async def stream_messages(reader: asyncio.StreamReader) -> AsyncIterator[dict[str, Any]]:
    while True:
        try:
            yield await read_message(reader)
        except (ConnectionError, asyncio.IncompleteReadError):
            break
