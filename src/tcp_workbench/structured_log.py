"""Structured JSON-lines logging."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .clock import monotonic_ms


class StructuredLogger:
    def __init__(self, path: str | Path | None = None, *, echo: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._lock = threading.Lock()

    def emit(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "wall_time": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "mono_ms": monotonic_ms(),
            "level": level,
            "event": event,
            **fields,
        }
        line = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self._lock:
            if self.path is not None:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
            if self.echo:
                print(line, flush=True)
        return record

    def info(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.emit("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.emit("WARNING", event, **fields)

    def error(
        self, event: str, *, code: str, message: str, **fields: Any
    ) -> dict[str, Any]:
        return self.emit("ERROR", event, code=code, message=message, **fields)

    def business_error(self, event: str, error: Any, **fields: Any) -> dict[str, Any]:
        payload = (
            error.as_dict() if hasattr(error, "as_dict") else {"message": str(error)}
        )
        return self.emit("ERROR", event, **payload, **fields)

    def tail(self, limit: int = 100) -> tuple[dict[str, Any], ...]:
        if self.path is None or not self.path.exists():
            return ()
        with self.path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()[-limit:]
        result = []
        for line in lines:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return tuple(result)
