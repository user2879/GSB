"""Structured JSONL logging used by both the CLI and the GUI."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class StructuredLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, event: str, **fields) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        return record
