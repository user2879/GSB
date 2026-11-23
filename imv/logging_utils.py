from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .util import canonical_json_text, wall_ms


class StructuredLogger:
    """Thread-safe NDJSON structured logger."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(
        self,
        level: str,
        event: str,
        message: str,
        *,
        code: str | None = None,
        location: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts_wall_ms": wall_ms(),
            "level": level,
            "event": event,
            "message": message,
        }
        if code:
            record["code"] = code
        if location:
            record["location"] = location
        if details:
            record["details"] = details
        line = canonical_json_text(record) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
        return record

    def read_tail(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        records.append({"level": "ERROR", "message": "invalid log line"})
        return records[-limit:]
