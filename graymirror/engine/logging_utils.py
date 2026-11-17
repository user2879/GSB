"""Structured JSONL logger.

Each event is written as one JSON object per line so downstream tooling can
parse it without any network dependency.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path: Path | None = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
        }
        record.update(_json_safe(fields))
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return record

    def error_event(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.event("ERROR", event, **fields)

    def info_event(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.event("INFO", event, **fields)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
