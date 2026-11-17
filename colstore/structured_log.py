"""Deterministic structured JSONL logging for scans and injected faults."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

from .errors import ColStoreError


class StructuredLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._counter = 0

    def event(self, level: str, event: str, **fields: Any) -> Dict[str, Any]:
        record = {
            "seq": self._next_seq(),
            "ts_ms": round(time.monotonic() * 1000.0, 6),
            "level": level,
            "event": event,
            **fields,
        }
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with self._lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return record

    def info(self, event: str, **fields: Any) -> Dict[str, Any]:
        return self.event("INFO", event, **fields)

    def error(self, event: str, exc: Optional[ColStoreError] = None, **fields: Any) -> Dict[str, Any]:
        if exc is not None:
            fields["error"] = exc.to_dict()
        return self.event("ERROR", event, **fields)

    def _next_seq(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def read_all(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
