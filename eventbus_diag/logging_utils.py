"""Thread-safe structured JSONL logging."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

from .canonical import wall_ms
from .errors import BusDiagnosticError


class StructuredLogger:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        echo: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.echo = echo
        self.stream = stream if stream is not None else sys.stderr
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, code: str, message: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "at_wall_ms": wall_ms(),
            "level": level,
            "code": code,
            "message": message,
            "thread": threading.current_thread().name,
        }
        record.update({key: value for key, value in fields.items() if value is not None})
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self.path:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
            if self.echo:
                self.stream.write(line + "\n")
                self.stream.flush()
        return record

    def info(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        return self.log("INFO", code, message, **fields)

    def warning(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        return self.log("WARNING", code, message, **fields)

    def error(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        return self.log("ERROR", code, message, **fields)

    def diagnostic(self, error: BusDiagnosticError, **fields: Any) -> dict[str, Any]:
        merged = error.as_dict()
        merged.update(fields)
        merged.pop("code", None)
        merged.pop("message", None)
        payload = merged.pop("details", {})
        merged.update(payload)
        return self.error(error.code, error.message, **merged)

    def exception(self, error: Exception, **fields: Any) -> dict[str, Any]:
        if isinstance(error, BusDiagnosticError):
            return self.diagnostic(error, **fields)
        return self.error("E_INTERNAL", str(error), error_type=type(error).__name__, **fields)
