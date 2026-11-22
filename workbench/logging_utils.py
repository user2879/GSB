"""Deterministic, append-only JSONL structured logging."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any


class JsonlLogger:
    """Thread-safe JSON Lines logger writing one compact object per line.

    Log entries never depend on dict insertion order for semantics: callers
    pass plain dicts and we serialise with ``sort_keys=True`` so identical
    events always produce identical bytes.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, level: str, event: str, **fields: Any) -> dict:
        entry: dict[str, Any] = {
            "ts_ns": time.time_ns(),
            "level": level,
            "event": event,
        }
        entry.update(fields)
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        return entry

    def info(self, event: str, **fields: Any) -> dict:
        return self.log("INFO", event, **fields)

    def warn(self, event: str, **fields: Any) -> dict:
        return self.log("WARN", event, **fields)

    def error(self, event: str, **fields: Any) -> dict:
        return self.log("ERROR", event, **fields)
