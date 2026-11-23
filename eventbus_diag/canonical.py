"""Canonical serialization and clock helpers.

All identifiers and digests in this project use one explicit JSON encoding.
That makes batch identities independent of Python dictionary insertion order,
thread scheduling, platform line endings, or dispatch order.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable


def stable_serialize(value: Any) -> str:
    """Return the one canonical JSON representation used by the ledger."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_value(value: Any) -> str:
    return digest_text(stable_serialize(value))


def wall_ms() -> int:
    return int(time.time() * 1000)


class MonotonicClock:
    """Indirection that keeps real-time lease tests deterministic in structure."""

    def __init__(self, function: Callable[[], float] | None = None) -> None:
        self._function = function or time.monotonic

    def seconds(self) -> float:
        return float(self._function())

    def milliseconds(self) -> float:
        return self.seconds() * 1000.0
