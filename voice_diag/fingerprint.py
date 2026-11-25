"""Stable, process-independent hashing and JSON helpers."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def derive_int(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\x1f{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def stable_float(seed: int, *parts: object) -> float:
    """Return a stable value in [-1, 1] without touching process RNG state."""

    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(f"{seed}\x1f{payload}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big")
    return (raw / float((1 << 64) - 1)) * 2.0 - 1.0


def round_float(value: float, digits: int = 12) -> float:
    result = round(float(value), digits)
    return 0.0 if result == 0 else result


def rounded_vector(values: np.ndarray | list[float], digits: int = 12) -> list[float]:
    return [round_float(float(value), digits) for value in values]


def write_json_atomic(path: Path, value: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(canonical_json(value).encode()).hexdigest()[:10]
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}-{suffix}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=indent)
        handle.write("\n")
        handle.flush()
    temp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
