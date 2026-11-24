from __future__ import annotations

import hashlib
import json
import struct
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def field_bytes(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack(">I", len(encoded)) + encoded


def uint32(value: int) -> bytes:
    return struct.pack(">I", value)


def rolling_digest(previous_hex: str, payload: bytes) -> str:
    previous = bytes.fromhex(previous_hex)
    return sha256_hex(b"CACHELEASE-LEDGER-v1\0" + previous + struct.pack(">Q", len(payload)) + payload)


ZERO_DIGEST = "0" * 64

