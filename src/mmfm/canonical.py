"""Canonical serialisation, hashing, local PRNG, and atomic file helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def content_digest(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def make_key(stage: str, payload: Any) -> str:
    return f"{stage}:{content_digest(payload)}"


def stable_u64(*parts: Any) -> int:
    """Derive a uint64 without Python's process-randomised hash()."""

    material = canonical_json(list(parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def stable_signed(radius: int, *parts: Any) -> int:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return int(stable_u64(*parts) % (radius * 2 + 1)) - radius


@dataclass
class HashPRNG:
    """A counter based SHA-256 PRNG.

    It has no dependence on ``random``, NumPy's global state, thread identity,
    wall-clock time, or execution order.
    """

    key: str
    counter: int = 0

    def u64(self) -> int:
        block = hashlib.sha256(f"{self.key}|{self.counter}".encode("utf-8")).digest()
        self.counter += 1
        return int.from_bytes(block[:8], "big")

    def below(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        return self.u64() % bound

    def integer(self, low: int, high: int) -> int:
        if high < low:
            raise ValueError("high must be >= low")
        return low + self.below(high - low + 1)

    def integers(self, count: int, low: int, high: int) -> list[int]:
        return [self.integer(low, high) for _ in range(count)]


def round_div(numerator: int, denominator: int) -> int:
    """Round an integer ratio to nearest, ties away from zero."""

    if denominator == 0:
        raise ZeroDivisionError("integer division by zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    n = abs(numerator)
    d = abs(denominator)
    quotient, remainder = divmod(n, d)
    if remainder * 2 >= d:
        quotient += 1
    return sign * quotient


def rounding_direction(numerator: int, denominator: int) -> str:
    """Return whether rounding moved a value up, down, or not at all."""

    rounded = round_div(numerator, denominator)
    exact_scaled = rounded * denominator
    if exact_scaled == numerator:
        return "exact"
    if (exact_scaled > numerator and denominator > 0) or (
        exact_scaled < numerator and denominator < 0
    ):
        return "up"
    return "down"


def round_half_up_decimal(value: float | Decimal, places: int = 3) -> Decimal:
    quant = Decimal(1).scaleb(-places)
    return Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)


def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value) + "\n")


def sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted(set(values))
