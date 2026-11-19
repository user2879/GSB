"""Thread-safe JSON-lines structured logging."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._seq = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, code: str, message: str, level: str = "INFO", **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            record = {
                "seq": self._seq,
                "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "level": level,
                "code": code,
                "message": message,
                "fields": fields,
            }
            if self.path is not None:
                line = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(line + "\n")
                    fh.flush()
            return record


def read_log_codes(path: str | Path | None) -> list[dict[str, Any]]:
    """Read structured log records and return stable, compact summaries."""

    if path is None:
        return []
    log_path = Path(path)
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(
                {
                    "seq": value.get("seq", 0),
                    "code": value.get("code", "UNKNOWN"),
                    "level": value.get("level", "INFO"),
                    "message": value.get("message", ""),
                }
            )
    return sorted(records, key=lambda item: (item["seq"], item["code"]))
