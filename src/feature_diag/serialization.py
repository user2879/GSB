"""JSON-safe wire encoding for snapshots and cache payloads."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .fingerprint import sha256_text


def to_wire(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_wire(item) for item in value]
    if isinstance(value, set):
        return sorted(to_wire(item) for item in value)
    return value


def from_wire(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            array = np.asarray(value["data"], dtype=value["dtype"])
            return array.reshape(value["shape"])
        return {key: from_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [from_wire(item) for item in value]
    return value


def dump_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(
            to_wire(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    return json.dumps(
        to_wire(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def load_json(path: str | Path, *, decode_wire: bool = True) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return from_wire(value) if decode_wire else value


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def payload_fingerprint(payload: Any) -> str:
    return sha256_text(dump_json(payload))
