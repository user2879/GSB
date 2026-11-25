from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json(value: Any) -> str:
    """Serialize data in the canonical form used by every fingerprint."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (tuple, set)):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def sha256_hex(payload: Any, *, domain: str = "cross-code") -> str:
    body = stable_json(payload).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + body).hexdigest()


def fingerprint(prefix: str, payload: Any) -> str:
    return f"{prefix}:{sha256_hex(payload, domain=prefix)}"


def deterministic_salt_int(fingerprint_value: str, *, length: int = 16) -> int:
    digest = hashlib.blake2b(
        fingerprint_value.encode("utf-8"), digest_size=32
    ).digest()
    return int.from_bytes(digest[: max(1, min(length, 32))], "big", signed=False)


def hash_value_to_width(value: Any, salt_fingerprint: str, width_bits: int) -> int:
    """Hash one canonical value tuple into an unsigned integer of width_bits."""
    payload = {"salt": salt_fingerprint, "value": value}
    digest = hashlib.blake2b(
        stable_json(payload).encode("utf-8"), digest_size=32
    ).digest()
    raw = int.from_bytes(digest, "big", signed=False)
    if width_bits <= 0:
        raise ValueError("width_bits must be positive")
    return raw & ((1 << width_bits) - 1)
