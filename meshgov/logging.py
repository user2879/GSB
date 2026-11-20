from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .util import mono_ms, utc_now_iso


class JsonlLogger:
    """Thread-safe append-only structured logger.

    Each line is a complete JSON object and is flushed immediately, which keeps
    the evidence useful if the process is force-terminated.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event_name: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": 1,
            "ts_utc": utc_now_iso(),
            "mono_ms": round(mono_ms(), 3),
            "event": event_name,
        }
        for key, value in fields.items():
            if key in {"schema_version", "ts_utc", "mono_ms", "event"}:
                continue
            record[key] = value
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()
        return record

    def error(self, error: Exception, **fields: Any) -> dict[str, Any]:
        payload = getattr(error, "to_dict", lambda: {"message": str(error)})()
        return self.emit("business_error", **fields, **payload)

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
