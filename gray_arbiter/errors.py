"""Stable business errors and structured diagnostics."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable


BUSINESS_CODES = {
    "E_POOL",
    "E_BORROW",
    "E_TARGET",
    "E_MIRROR",
    "E_TIMEOUT",
    "E_ROUTE",
    "E_REPLAY",
    "E_SNAPSHOT",
    "E_QUEUE",
    "E_CONFIG",
}


@dataclass
class BusinessError(Exception):
    """An error that is safe to expose as a stable business result."""

    code: str
    message: str
    field: str | None = None
    request_id: str | None = None
    pool_id: str | None = None
    phase: str = "validation"
    details: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in BUSINESS_CODES:
            raise ValueError(f"unknown business code: {self.code}")
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "request_id": self.request_id,
            "pool_id": self.pool_id,
            "phase": self.phase,
        }
        if self.details:
            result["details"] = self.details
        return result


class StructuredLogger:
    """Append-only NDJSON logger with a stable field order.

    The logger deliberately does not include wall-clock values in the event
    ordering decision.  ``sequence`` is the authoritative stable order.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sequence = 0
        self.events: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    self._sequence = max(self._sequence, int(event.get("sequence", 0)))
            except (OSError, ValueError, json.JSONDecodeError):
                pass

    def emit(
        self,
        code: str,
        request_id: str | None,
        pool_id: str | None,
        phase: str,
        message: str,
        **details: Any,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "business_code": code,
                "request_id": request_id,
                "pool_id": pool_id,
                "phase": phase,
                "message": message,
                "timestamp_ns": time.time_ns(),
                "details": details,
            }
            self.events.append(event)
            if self.path:
                with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            return event

    def emit_error(self, error: BusinessError) -> dict[str, Any]:
        details = dict(error.details)
        details.setdefault("field", error.field)
        return self.emit(
            error.code,
            error.request_id,
            error.pool_id,
            error.phase,
            error.message,
            **details,
        )

    def sorted_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self.events, key=lambda item: item["sequence"])

    def write_summary(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"events": self.sorted_events()}
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def errors_to_dict(errors: Iterable[BusinessError]) -> list[dict[str, Any]]:
    return [error.to_dict() for error in errors]
