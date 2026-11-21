from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .canonical import utc_now_iso
from .errors import DiagnosticError


class EventLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event: str, **fields: Any) -> dict[str, Any]:
        payload = {"timestamp": utc_now_iso(), "event": event, **fields}
        if self.path is not None:
            line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return payload

    def error(self, error: DiagnosticError) -> dict[str, Any]:
        return self.event(
            "error",
            code=error.code,
            message=error.message,
            location=error.location,
            details=error.details,
        )
