from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FLOAT_DIGITS = 12


def _stable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float is not allowed in canonical output")
        rounded = round(value, FLOAT_DIGITS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, np.ndarray):
        return [_stable_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _stable_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _stable_value(value.to_dict())
    raise TypeError(f"unsupported canonical value: {type(value)!r}")


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        _stable_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any, newline: bool = True) -> bytes:
    text = canonical_dumps(value)
    return (text + ("\n" if newline else "")).encode("utf-8")


def canonical_line(value: Any) -> str:
    return canonical_dumps(value) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numpy_sha256(array: np.ndarray, dtype: str | np.dtype | None = None) -> str:
    source = np.ascontiguousarray(array)
    if dtype is not None:
        source = source.astype(np.dtype(dtype), copy=False)
    return sha256_bytes(source.tobytes(order="C"))


def write_json(path: str | os.PathLike[str], value: Any) -> str:
    data = canonical_bytes(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return sha256_bytes(data)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any]) -> str:
    chunks = [canonical_line(row).encode("utf-8") for row in rows]
    data = b"".join(chunks)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return sha256_bytes(data)


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    rows: list[Any] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nearest_rank_percentile(values: Iterable[float], percentile: int) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    p = min(100, max(0, int(percentile)))
    index = 0 if p == 0 else math.ceil((p / 100.0) * len(ordered)) - 1
    return ordered[min(len(ordered) - 1, max(0, index))]
