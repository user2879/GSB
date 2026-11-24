"""Structured JSONL logging used by injection and replay tooling."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any


class StructuredLogger:
    """Append-only JSON lines logger safe for scheduler worker threads."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = None if path is None else os.fspath(path)
        self._lock = threading.Lock()
        if self.path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def log(self, event: str, **fields: Any) -> dict:
        record = {
            "ts_ms": time.time_ns() // 1_000_000,
            "event": event,
            **fields,
        }
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with self._lock:
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return record
