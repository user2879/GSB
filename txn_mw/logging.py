"""Thread-safe newline-delimited JSON logging for protocol-visible errors."""

from __future__ import annotations

import datetime as _dt
import os
import threading
from pathlib import Path
from typing import Any

from .canonical import canonical_json


class StructuredLogger:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path).resolve() if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, level: str, code: str, message: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "level": level,
            "code": code,
            "message": message,
            **fields,
        }
        line = canonical_json(record) + "\n"
        with self._lock:
            if self.path:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line)
                    handle.flush()
        return record

    def error(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        return self.emit("error", code, message, **fields)

    def info(self, message: str, **fields: Any) -> dict[str, Any]:
        return self.emit("info", "I_NORMAL", message, **fields)


class NullLogger(StructuredLogger):
    def __init__(self) -> None:
        super().__init__(None)
