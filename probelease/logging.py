"""Small dependency-free structured JSONL logger."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: str, code: str = "OK", **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": event,
            "code": code,
            "monotonic_ns": time.monotonic_ns(),
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self.path is not None:
            with self._lock:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return record

