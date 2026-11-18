"""Structured JSONL logging with a deterministic, machine-parseable line per event."""

from __future__ import annotations

import json
import os
from typing import Any

from .util import monotonic_ms, utc_now_iso


class StructuredLogger:
    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._handle = open(path, "a", encoding="utf-8")

    def log(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ts_utc": utc_now_iso(),
            "monotonic_ms": round(monotonic_ms(), 3),
            "level": level,
            "event": event,
        }
        entry.update(fields)
        self._handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
        self._handle.flush()
        return entry

    def close(self) -> None:
        self._handle.close()
