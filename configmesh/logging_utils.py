"""Structured JSON-lines logging with a stable event schema."""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from .util import wall_ms


class StructuredLogger:
    """Thread-safe logger writing one JSON object per line plus memory capture."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            open(path, "w", encoding="utf-8").close()

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts_ms": wall_ms(),
            "event": event,
            **fields,
        }
        with self._lock:
            self.events.append(record)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True,
                                            ensure_ascii=False, default=str) + "\n")
        return record

    def error(self, code: str, event: str, **fields: Any) -> dict[str, Any]:
        return self.log(event, level="error", business_code=code, **fields)

    def info(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.log(event, level="info", **fields)

    def business_codes(self) -> list[str]:
        return [e["business_code"] for e in self.events if "business_code" in e]
