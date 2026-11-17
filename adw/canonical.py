"""Canonical serialization, stable hashing and a tiny deterministic PRNG."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


SCORE_DIGITS = 12
MASK64 = (1 << 64) - 1


def stable_float(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite value in canonical payload: {value!r}")
    if value == 0.0:
        return "0.0"
    return format(float(value), ".17g")


def jsonable(value: Any) -> Any:
    """Convert NumPy/scalar values into a stable JSON-compatible shape."""

    if isinstance(value, np.ndarray):
        return [jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in canonical payload")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def content_key(namespace: str, value: Any) -> str:
    return sha256_text(namespace + "\n" + canonical_json(value))


def sha256_file(path: str | bytes) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quantize_score(value: float, digits: int = SCORE_DIGITS) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("anomaly algorithm produced a non-finite score")
    rounded = round(value, digits)
    return 0.0 if rounded == 0.0 else rounded


def fsum(values: Iterable[float]) -> float:
    return math.fsum(float(x) for x in values)


def stable_median(values: np.ndarray) -> float:
    flat = np.sort(np.asarray(values, dtype=np.float64).reshape(-1), kind="mergesort")
    n = int(flat.size)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(flat[mid])
    return float((flat[mid - 1] + flat[mid]) / 2.0)


@dataclass
class StableRNG:
    """SplitMix64 with explicit integer arithmetic (no NumPy global RNG)."""

    seed: int

    def __post_init__(self) -> None:
        self.state = int(self.seed) & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def random(self) -> float:
        # 53 random bits yield a platform-independent uniform double.
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def randbelow(self, stop: int) -> int:
        if stop <= 0:
            raise ValueError("stop must be positive")
        limit = (1 << 64) - ((1 << 64) % stop)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % stop
