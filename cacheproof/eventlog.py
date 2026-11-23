"""Append-only structured JSON event log."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from .canonical import canonical_text


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_sequence = self._discover_next_sequence()

    def _discover_next_sequence(self) -> int:
        if not self.path.exists():
            return 1
        sequence = 0
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    value = event.get("event_seq")
                    if isinstance(value, int) and value > sequence:
                        sequence = value
        except OSError:
            return 1
        return sequence + 1

    def emit(
        self,
        event: str,
        *,
        business_code: str = "OK",
        operation_seq: int | None = None,
        segment_id: str | None = None,
        entry_key: str | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
        monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            event_seq = self._next_sequence
            self._next_sequence += 1
            payload: dict[str, Any] = {
                "log_format": 1,
                "event_seq": event_seq,
                "operation_seq": operation_seq if operation_seq is not None else event_seq,
                "monotonic_ns": time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
                "event": event,
                "business_code": business_code,
            }
            if segment_id is not None:
                payload["segment_id"] = segment_id
            if entry_key is not None:
                payload["entry_key"] = entry_key
            if field is not None:
                payload["field"] = field
            if details:
                payload["details"] = details
            line = canonical_text(payload) + "\n"
            with self.path.open("a", encoding="utf-8", newline="\n") as output:
                output.write(line)
                output.flush()
                try:
                    os.fsync(output.fileno())
                except OSError:
                    pass
            return payload

