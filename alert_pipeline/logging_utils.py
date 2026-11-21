from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .util import canonical_json, epoch_ms


class StructuredLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        payload = {
            "ts_ms": epoch_ms(),
            "event": event,
            **fields,
        }
        line = canonical_json(payload) + "\n"
        with self._lock:
            descriptor = os.open(
                str(self.path),
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                os.write(descriptor, line.encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def error(self, error: Any, **fields: Any) -> None:
        payload = error.to_dict() if hasattr(error, "to_dict") else {"message": str(error)}
        self.write("error", **payload, **fields)
