"""Structured JSONL event logging (local file, no network)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "ts_ns": time.time_ns(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        return entry

    def log_error(self, error: Exception) -> dict[str, Any]:
        payload = error.to_dict() if hasattr(error, "to_dict") else {
            "code": "E_INTERNAL",
            "message": str(error),
            "fields": [],
            "context": {},
        }
        return self.log("error", **payload)

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
