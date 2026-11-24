"""Structured JSONL logging used by the CLI and injection chain."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional


class StructuredLogger:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                        exist_ok=True)

    def log(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "ts": time.time(),
            "level": level,
            "event": event,
            **fields,
        }
        if self.path:
            with self._lock, open(self.path, "a", encoding="utf-8",
                                  newline="\n") as handle:
                handle.write(json.dumps(entry, sort_keys=True,
                                        ensure_ascii=False) + "\n")
        return entry

    def info(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.log("INFO", event, **fields)

    def error(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.log("ERROR", event, **fields)
