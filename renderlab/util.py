"""Deterministic hashing, serialization and random streams."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


WORKBENCH_FORMAT_VERSION = "renderlab-format-v1"


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


def sha256_hex(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            payload = part
        elif isinstance(part, bytearray):
            payload = bytes(part)
        elif isinstance(part, str):
            payload = part.encode("utf-8")
        else:
            payload = canonical_bytes(part)
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def value_fingerprint(domain: str, value: Any) -> str:
    return sha256_hex(domain, value)


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = canonical_bytes(
        {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "order": "C",
        }
    )
    return sha256_hex("array-v1", header, contiguous.tobytes(order="C"))


def bytes_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_summary(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "digest": array_digest(contiguous),
        "mean": round(float(np.mean(contiguous)), 12),
        "std": round(float(np.std(contiguous)), 12),
        "min": round(float(np.min(contiguous)), 12),
        "max": round(float(np.max(contiguous)), 12),
    }


@dataclass(frozen=True)
class StableTaskKey:
    tier: str
    tier_rank: int
    prompt_index: int
    step: int
    frame_index: int

    def tuple(self) -> tuple[int, int, int, int]:
        return (self.tier_rank, self.prompt_index, self.step, self.frame_index)

    def text(self) -> str:
        return (
            f"tier={self.tier_rank:03d}:{self.tier}|prompt={self.prompt_index:06d}|"
            f"step={self.step:06d}|frame={self.frame_index:06d}"
        )


class HashRandom:
    """A small SHA-256 counter-stream RNG independent of process state.

    The stream is intentionally implemented here rather than using Python's
    ``random`` or a runtime-global NumPy RNG.  The same parts always produce
    the same sequence on every run and every process.
    """

    def __init__(self, domain: str, *parts: Any) -> None:
        self._seed = sha256_hex(domain, *parts).encode("ascii")
        self._counter = 0
        self._normal_spare: float | None = None

    def _u64(self) -> int:
        block = hashlib.sha256(self._seed + struct.pack(">Q", self._counter)).digest()
        self._counter += 1
        return int.from_bytes(block[:8], "big")

    def uniform(self) -> float:
        return (self._u64() + 0.5) / float(1 << 64)

    def normal(self) -> float:
        if self._normal_spare is not None:
            value = self._normal_spare
            self._normal_spare = None
            return value
        u1 = self.uniform()
        u2 = self.uniform()
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        self._normal_spare = radius * math.sin(angle)
        return radius * math.cos(angle)

    def normals(self, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.float64)
        for index in range(count):
            out[index] = self.normal()
        return out

    def uniforms(self, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.float64)
        for index in range(count):
            out[index] = self.uniform()
        return out


def finite_float(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def sorted_strings(values: Iterable[str]) -> list[str]:
    return sorted(str(value) for value in values)
