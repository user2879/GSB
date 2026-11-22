from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "domain-pipeline/1.0.0"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def fingerprint(namespace: str, value: Any) -> str:
    try:
        payload = canonical_json_bytes(value)
    except ValueError:
        payload = canonical_json_bytes(_fingerprint_safe(value))
    return sha256_bytes(namespace.encode("utf-8") + b"\0" + payload)


def _fingerprint_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"__nonfinite_float__": "nan"}
        return {"__nonfinite_float__": "inf" if value > 0 else "-inf"}
    if isinstance(value, dict):
        return {str(key): _fingerprint_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_safe(item) for item in value]
    return value


def fingerprint_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def stable_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite float is not serializable")
    if value == 0.0:
        return 0.0
    return value


def round_half_up_nonnegative(value: float) -> int:
    if value < 0:
        return -int(math.floor(-value + 0.5))
    return int(math.floor(value + 0.5))


def deep_get(document: dict[str, Any], dotted_path: str) -> Any:
    current: Any = document
    for part in dotted_path.split("."):
        if isinstance(current, list):
            index = int(part)
            current = current[index]
        else:
            current = current[part]
    return current


def deep_set(document: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current: Any = document
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            if part not in current or not isinstance(current[part], (dict, list)):
                current[part] = {}
            current = current[part]
    last = parts[-1]
    if isinstance(current, list):
        index = int(last)
        if index < 0 or index >= len(current):
            raise IndexError(dotted_path)
        current[index] = value
    else:
        current[last] = value


def sorted_failures(failures: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(item.get("scope", "")),
            int(item.get("line", 0)),
            str(item.get("path", "")),
            str(item.get("sample_id", "")),
            str(item.get("batch_id", "")),
            int(item.get("operation_index", -1)),
            tuple(str(code) for code in item.get("codes", [])),
        )

    return [dict(item) for item in sorted(failures, key=key)]


def normalized_artifact(artifact: dict[str, Any] | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    return {
        key: value
        for key, value in artifact.items()
        if not key.startswith("_")
    }
