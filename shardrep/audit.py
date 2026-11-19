"""Structured JSONL and SQLite event logging."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import atomic_write_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class StructuredLogger:
    """Every event carries the required code, shard, version and timestamp."""

    def __init__(self, jsonl_path: Path | None = None, ledger: Any | None = None) -> None:
        self.jsonl_path = jsonl_path
        self.ledger = ledger
        self._lock = threading.Lock()
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        code: str,
        *,
        shard_id: str | None = None,
        version: int | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        event = {
            "code": code,
            "shard_id": shard_id,
            "version": version,
            "timestamp": utc_now(),
            **payload,
        }
        line = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self.jsonl_path is not None:
                with self.jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
            if self.ledger is not None:
                self.ledger.record_event(event)
        return event

    def read_events(self) -> list[dict[str, Any]]:
        if self.jsonl_path is None or not self.jsonl_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(json.loads(line))
        return events


class JsonlReport:
    @staticmethod
    def write(path: Path, value: Any) -> None:
        atomic_write_text(
            path,
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )

