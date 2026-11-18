"""Append-only structured JSONL logger.

Every line is a self-contained JSON object with a monotonic millisecond
timestamp, an event name and a payload.  The logger is thread-safe so that the
parallel scheduler can record pool events from worker threads.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: str | Path | None) -> None:
        self._lock = threading.Lock()
        self._path = Path(path) if path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, **payload: Any) -> dict[str, Any]:
        record = {
            "ts_ms": time_monotonic_ms(),
            "event": name,
            "payload": payload,
        }
        if self._path is not None:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            with self._lock:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        return record

    @property
    def path(self) -> Path | None:
        return self._path


def time_monotonic_ms() -> int:
    """Monotonic clock in whole milliseconds, used for queue/hold accounting."""
    return int(time.monotonic() * 1000)
