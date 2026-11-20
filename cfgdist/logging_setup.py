"""Deterministic structured (JSON-lines) logging."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class StructuredLogger:
    """Thread-safe JSON-lines logger shared by platform components."""

    def __init__(self, log_path: str | Path | None):
        self._lock = threading.Lock()
        self._path = Path(log_path) if log_path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path | None:
        return self._path

    def log(self, event: str, level: str = "INFO", **fields) -> dict:
        record = dict(fields)
        record.update({
            "ts": utc_now_iso(),
            "event": event,
            "level": level,
        })
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with self._lock:
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return record

    def biz_error(self, event: str, code: str, message: str, **fields):
        record = dict(fields)
        record["error_code"] = code
        record["error_message"] = message
        return self.log(event, level="ERROR", **record)


def read_log_records(log_path: str | Path) -> list[dict]:
    path = Path(log_path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
