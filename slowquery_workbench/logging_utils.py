"""Small dependency-free JSONL logger for auditable exception chains."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JsonlLogger:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        level: str,
        event: str,
        message: str = "",
        *,
        code: str | None = None,
        details: Any = None,
        **fields: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": utc_now(),
            "level": level.upper(),
            "event": event,
            "message": message,
        }
        if code is not None:
            record["code"] = code
        if details is not None:
            record["details"] = details
        record.update(fields)
        if self.path:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            with self._lock:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
        return record

    def info(self, event: str, message: str = "", **fields: Any) -> dict[str, Any]:
        return self.emit("INFO", event, message, **fields)

    def warning(self, event: str, message: str = "", **fields: Any) -> dict[str, Any]:
        return self.emit("WARNING", event, message, **fields)

    def error(
        self,
        event: str,
        message: str = "",
        *,
        code: str | None = None,
        details: Any = None,
        **fields: Any,
    ) -> dict[str, Any]:
        return self.emit("ERROR", event, message, code=code, details=details, **fields)


class NullLogger(JsonlLogger):
    def __init__(self) -> None:
        super().__init__(None)
