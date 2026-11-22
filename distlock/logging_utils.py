"""Structured JSON-lines logging used by the demo and exception injections."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "ts_monotonic_ms": time.monotonic_ns() // 1_000_000,
            "level": level,
            "event": event,
            **fields,
        }
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
        return entry

    def error(self, code: str, event: str, **fields: Any) -> dict[str, Any]:
        return self.log("ERROR", event, code=code, **fields)

    def close(self) -> None:
        with self._lock:
            self._fh.close()
