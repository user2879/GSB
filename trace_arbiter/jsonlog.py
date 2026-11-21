from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_text
from .errors import ArbiterError


class JsonlLogger:
    def __init__(self, path: str | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._file = None
        if path:
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", encoding="utf-8", newline="\n")

    def emit(self, event: str, **fields: Any) -> None:
        if self._file is None:
            return
        payload = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **fields,
        }
        line = canonical_text(payload) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def error(self, event: str, exc: ArbiterError) -> None:
        self.emit(event, **exc.as_dict())

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()
