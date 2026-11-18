from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .utils import utc_now


class StructuredLogger:
    """Append-only JSONL logger suitable for local support diagnostics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "timestamp_utc": utc_now(),
            "code": code,
            "message": message,
        }
        for key, value in fields.items():
            if value is not None:
                entry[key] = value
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
        return entry

    def error(self, error: Any, *, operation: str, **fields: Any) -> dict[str, Any]:
        if hasattr(error, "as_dict"):
            payload = error.as_dict()
        else:
            payload = {"code": "E_INTERNAL", "message": str(error)}
        details = dict(payload.pop("details", {}) or {})
        details.update(fields)
        return self.emit(
            str(payload.get("code", "E_INTERNAL")),
            str(payload.get("message", error)),
            operation=operation,
            volume_id=payload.get("volume_id"),
            field_path=payload.get("field_path"),
            details=details or None,
        )


def read_log(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    entries: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries
