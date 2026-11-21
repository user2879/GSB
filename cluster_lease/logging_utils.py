"""Small structured JSONL logger used by the CLI and orchestration."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .ledger import utc_now


class StructuredLogger:
    def __init__(self, path: str | None = None, *, echo: bool = False) -> None:
        self.path = Path(path) if path else None
        self.echo = echo
        self.lock = threading.Lock()
        self.sequence = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)

    def log(self, event_code: str, message: str, **fields: Any) -> dict[str, Any]:
        with self.lock:
            self.sequence += 1
            record: dict[str, Any] = {
                "code": event_code,
                "level": "error" if event_code.startswith("E_") else "info",
                "logged_at_utc": utc_now(),
                "message": message,
                "ok": not event_code.startswith("E_"),
                "sequence": self.sequence,
            }
            record.update(fields)
            encoded = canonical_json(record)
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")
            if self.echo:
                sys.stderr.write(encoded + "\n")
                sys.stderr.flush()
            return record

    def service_line(self, cluster_id: str, value: dict[str, Any]) -> None:
        code = str(value.get("code", "SERVICE_LOG"))
        message = str(value.get("message", code))
        self.log(code, message, cluster_id=cluster_id, service=value)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    result.append(value)
    return result
