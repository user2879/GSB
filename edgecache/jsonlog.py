"""Append-only structured JSON-lines logging."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        code: str,
        *,
        level: str = "INFO",
        node_id: str | None = None,
        batch_id: str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        record = {
            "timestamp_ms": int(time.time() * 1000),
            "level": level,
            "code": code,
            "node_id": node_id,
            "batch_id": batch_id,
            "details": details,
        }
        if self.path is not None:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return record

