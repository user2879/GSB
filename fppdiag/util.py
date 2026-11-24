"""Deterministic helpers: canonical JSON, content digests, logging, RNG."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize with sorted keys and stable separators so digest is portable."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def content_digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def bytes_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class StructuredLogger:
    """Append-only JSON-lines logger; one structured event per line."""

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._seq = 0

    def event(self, level: str, event: str, **fields: Any) -> dict:
        self._seq += 1
        record = {"ts_ms": int(time.time() * 1000), "seq": self._seq,
                  "level": level, "event": event}
        record.update(fields)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(record) + "\n")
        return record


class LcgRng:
    """Deterministic 64-bit LCG (Numerical Recipes constants).

    Pure integer arithmetic, platform independent; identical sequence on
    Windows/Linux and under any block partitioning.
    """

    _A = 6364136223846793005
    _C = 1442695040888963407
    _M = 1 << 64

    def __init__(self, seed: int):
        self.state = int(seed) & (self._M - 1)

    def next_u64(self) -> int:
        self.state = (self._A * self.state + self._C) % self._M
        return self.state

    def next_int(self, low: int, high: int) -> int:
        """Uniform integer in the inclusive range [low, high].

        Uses a 63-bit draw so low + offset never exceeds the signed 64-bit
        envelope used by the on-disk column format.
        """
        if high < low:
            raise ValueError("empty RNG range")
        span = high - low + 1
        draw = self.next_u64() >> 1
        return low + (draw % span)

    def next_fixed(self, low_real: int, high_real: int, scale_exp: int) -> int:
        """Uniform fixed-point sample in [low_real, high_real] at scale_exp.

        low_real/high_real are integer ulps at scale_exp (no float involved).
        """
        return self.next_int(low_real, high_real)

    def next_bps(self) -> int:
        """Draw a value in [0, 9999] basis points, used for the null bitmap."""
        return self.next_u64() % 10000
