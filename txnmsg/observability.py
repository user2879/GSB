"""Structured JSONL logging for business-code failures."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import BusinessError


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class StructuredLogger:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def write_error(self, error: BusinessError, **extra: Any) -> None:
        if error.logged:
            return
        record = {
            "timestamp": utc_now_text(),
            "code": error.code,
            "message": error.message,
            "event_seq": error.event_seq,
            "partition_id": error.partition_id,
            "transaction_id": error.transaction_id,
            "field": error.field,
            "record_seq": error.record_seq,
            "details": error.details,
        }
        record.update(extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        error.logged = True


def default_log_path(database_path: str | os.PathLike[str]) -> Path:
    database = Path(database_path)
    return database.parent / "logs" / "business.jsonl"
