"""Thread-safe JSON Lines logging with stable business codes."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    """Append-only UTF-8 JSONL logger.

    The logger deliberately uses a lock so observation worker threads and the
    Tk event thread cannot interleave partial records.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Verify writability early, while still allowing a caller to use a
            # temporary or user-selected path.
            with self.path.open("a", encoding="utf-8"):
                pass

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )

    def emit(self, level: str, code: str, message: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts_utc": self._timestamp(),
            "level": level.upper(),
            "code": code,
            "message": message,
        }
        for key in sorted(fields):
            value = fields[key]
            if value is not None:
                record[key] = value
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self.path is not None:
            with self._lock:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return record

    def info(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        return self.emit("INFO", code, message, **fields)

    def warning(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        return self.emit("WARN", code, message, **fields)

    def error(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        return self.emit("ERROR", code, message, **fields)


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
