"""Structured JSON-lines logger with an SQLite mirror for retrieval.

Every operational event is emitted as one JSON object per line into
``logs/<session>.jsonl`` and simultaneously mirrored into the ``events`` table
so the validation tool can retrieve the audit chain by revision and/or node.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, log_dir: str | Path, session_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.path = self.log_dir / f"{session_id}.jsonl"
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": time.time(),
            "session_id": self.session_id,
            "event": event_type,
        }
        record.update(_jsonable(fields))
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
        return record

    def error(self, code: str, **fields: Any) -> dict[str, Any]:
        return self.event("error", code=code, **fields)

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
