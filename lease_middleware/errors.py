"""Business error codes and the structured logging helper.

All domain failures raise :class:`MiddlewareError` carrying a stable business
code.  Every failure is also emitted as one structured JSON log line so the
result can be audited independently of the console output.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Supported business codes (stable contract used by the CLI and verifier).
E_SPEC = "E_SPEC"
E_LEASE = "E_LEASE"
E_VERSION = "E_VERSION"
E_CURSOR = "E_CURSOR"
E_UPGRADE = "E_UPGRADE"
E_REPLAY = "E_REPLAY"


@dataclass
class MiddlewareError(Exception):
    code: str
    message: str
    field_path: str = ""
    client_id: str | None = None
    lease_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - trivial
        loc = f" [{self.field_path}]" if self.field_path else ""
        return f"{self.code}: {self.message}{loc}"


class StructuredLogger:
    """Append-only JSONL logger; one JSON object per line."""

    def __init__(self, path: str | Path | None):
        self._lock = threading.Lock()
        self.path: Path | None = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, code: str, message: str, *, client_id: str | None = None,
            lease_id: str | None = None, field_path: str = "", **extra: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "code": code,
            "message": message,
            "client_id": client_id,
            "lease_id": lease_id,
            "field": field_path,
        }
        record.update(extra)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def error(self, err: MiddlewareError) -> None:
        self.log("ERROR", err.code, err.message, client_id=err.client_id,
                 lease_id=err.lease_id, field_path=err.field_path, **err.context)

    def info(self, code: str, message: str, **kw: Any) -> None:
        self.log("INFO", code, message, **kw)
