"""Structured JSON-lines logging used by the CLI and demo."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class StructuredLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, event: str, **fields) -> dict:
        entry = {"ts": datetime.now(timezone.utc).isoformat(),
                 "level": level, "event": event, **fields}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def info(self, event: str, **fields) -> dict:
        return self.log("INFO", event, **fields)

    def error(self, event: str, **fields) -> dict:
        return self.log("ERROR", event, **fields)
