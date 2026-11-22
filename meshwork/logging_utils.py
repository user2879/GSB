from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    """Append-only JSON-lines logger with deterministic field structure."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": level,
            "event": event,
        }
        record.update(fields)
        if self.path is not None:
            line = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            with self._lock:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
        return record

    def info(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.emit("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.emit("WARNING", event, **fields)

    def error(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.emit("ERROR", event, **fields)
