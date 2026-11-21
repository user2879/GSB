"""Canonical serialization and digest helpers.

All machine-readable artifacts use the same JSON canonicalization.  No digest
depends on Python's hash seed or on dictionary insertion order.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


SHA256_PREFIX = "sha256:"
FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SERIAL_RE = re.compile(r"^[0-9A-F]{2,64}$")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return SHA256_PREFIX + hashlib.sha256(data).hexdigest()


def digest_obj(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def digest_without_field(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return digest_obj(payload)


def fingerprint_der(der: bytes) -> str:
    return digest_bytes(der)


def normalize_fingerprint(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("fingerprint must be a string")
    text = value.strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError("fingerprint must be sha256 followed by 64 hex characters")
    return SHA256_PREFIX + text


def normalize_serial(value: str | int) -> str:
    if isinstance(value, bool):
        raise ValueError("serial must not be boolean")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("serial must be positive")
        text = format(value, "X")
    elif isinstance(value, str):
        text = value.strip().upper()
        if text.startswith("0X"):
            text = text[2:]
        if not re.fullmatch(r"[0-9A-F]+", text):
            raise ValueError("serial must contain hexadecimal digits")
    else:
        raise ValueError("serial must be a string or integer")
    if len(text) % 2:
        text = "0" + text
    if not (2 <= len(text) <= 64) or not SERIAL_RE.fullmatch(text):
        raise ValueError("serial must contain 2 to 64 hexadecimal digits")
    if int(text, 16) <= 0:
        raise ValueError("serial must be positive")
    return text


def stable_sorted_unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def write_canonical_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(value) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

