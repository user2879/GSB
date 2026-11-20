from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_json


class JsonlLogger:
    def __init__(self, log_dir: str | Path, *, source: str = "trace-diag") -> None:
        self.directory = Path(log_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "trace_diag.jsonl"
        self.source = source

    def event(
        self,
        event: str,
        *,
        level: str = "INFO",
        code: str | None = None,
        path: str | None = None,
        message: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": level,
            "source": self.source,
            "event": event,
        }
        if code is not None:
            row["code"] = code
        if path is not None:
            row["path"] = path
        if message is not None:
            row["message"] = message
        if context:
            row["context"] = context
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(row) + "\n")
        return row


def print_json(value: Any, *, stream: Any = None) -> None:
    target = stream or sys.stdout
    target.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
