"""Canonical JSON, snake-case stable serialisation, and cache-key helpers."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable


def normalise(value: Any) -> Any:
    """Return a JSON-safe value with stable floating-point zero handling."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be serialised")
        if value == 0.0:
            return 0.0
        return value
    if isinstance(value, dict):
        return {str(key): normalise(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [normalise(item) for item in value]
    if hasattr(value, "as_record") and callable(value.as_record):
        return normalise(value.as_record())
    return value


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    value = normalise(value)
    if pretty:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_cache_key(
    stage: str,
    *,
    content_fingerprint: str,
    parameter_fingerprint: str,
    bit_width: int,
    fractional_bits: int,
    upstream_key: str | None = None,
    stage_parameters: Any | None = None,
) -> str:
    payload: dict[str, Any] = {
        "stage": stage,
        "content_fingerprint": content_fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
        "bit_width": int(bit_width),
        "fractional_bits": int(fractional_bits),
    }
    if upstream_key is not None:
        payload["upstream_key"] = upstream_key
    if stage_parameters is not None:
        payload["stage_parameters"] = stage_parameters
    return sha256_value(payload)


def float_token(value: float | None) -> str:
    if value is None:
        return "M"
    if value == 0.0:
        return "0x0.0p+0"
    return float(value).hex()


def fingerprint_rows(
    rows: Iterable[tuple[str, str, dict[str, float | None]]],
    columns: Iterable[str],
    *,
    include_group: bool = True,
) -> str:
    """Fingerprint row IDs and ordered per-column values without JSON float drift."""

    columns = sorted(columns)
    digest = hashlib.sha256()
    for row_id, group_id, values in sorted(rows, key=lambda item: item[0]):
        digest.update(row_id.encode("utf-8"))
        digest.update(b"\x1f")
        if include_group:
            digest.update(group_id.encode("utf-8"))
            digest.update(b"\x1f")
        for column in columns:
            digest.update(column.encode("utf-8"))
            digest.update(b"=")
            digest.update(float_token(values.get(column)).encode("ascii"))
            digest.update(b"\x1e")
        digest.update(b"\x1d")
    return digest.hexdigest()

