"""Structured JSONL logging confined to the local workspace."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict

_LOCK = threading.Lock()


class StructuredLogger:
    """Append-only JSON lines logger with stable, sorted key output."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def event(self, level: str, code: str, message: str,
              **fields: Any) -> Dict[str, Any]:
        record = {
            "ts_ms": int(time.time() * 1000),
            "level": level,
            "code": code,
            "message": message,
        }
        record.update(sorted(fields.items()))
        line = json.dumps(record, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        with _LOCK:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return record

    def info(self, code: str, message: str, **fields: Any) -> Dict[str, Any]:
        return self.event("INFO", code, message, **fields)

    def error(self, code: str, message: str, **fields: Any) -> Dict[str, Any]:
        return self.event("ERROR", code, message, **fields)
