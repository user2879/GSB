from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .clock import Clock, MonotonicClock
from .hashing import canonical_json


class Ledger:
    def __init__(self, path: str | Path, clock: Clock | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or MonotonicClock()
        self._lock = threading.Lock()

    def append(self, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "ledger_clock_ms": self.clock.now_ms(),
            "event_type": event_type,
            **(payload or {}),
        }
        line = canonical_json(event)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return event

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events
