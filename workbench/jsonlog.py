"""Small append-only structured logger local to the workspace."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .util import utc_now_iso


class StructuredLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def event(self, event: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"ts_utc": utc_now_iso(), "event": event}
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        return record
