from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


E_SPEC = "E_SPEC"
E_REV = "E_REV"
E_PORT = "E_PORT"
E_LEDGER = "E_LEDGER"
E_VIEW = "E_VIEW"
E_REPLAY = "E_REPLAY"
E_LATENCY = "E_LATENCY"
E_SERVICE = "E_SERVICE"
E_ENV = "E_ENV"


@dataclass(frozen=True)
class SpecIssue:
    field_path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"field_path": self.field_path, "message": self.message}


class DiagError(Exception):
    """A diagnosable failure with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
        issues: list[SpecIssue] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field_path = field_path
        self.details = details or {}
        self.issues = issues or []

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.field_path is not None:
            payload["field_path"] = self.field_path
        if self.details:
            payload["details"] = self.details
        if self.issues:
            payload["issues"] = [issue.as_dict() for issue in self.issues]
        return payload


def iso_utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class StructuredLogger:
    """Append-only JSON Lines logger safe for this single-process application."""

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        code: str,
        message: str,
        *,
        severity: str = "error",
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts_utc": iso_utc_now(),
            "code": code,
            "severity": severity,
            "message": message,
        }
        if field_path is not None:
            record["field_path"] = field_path
        if details:
            record["details"] = details
        if self.path is not None:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self._lock:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
        return record

    def info(self, code: str, message: str, **kwargs: Any) -> dict[str, Any]:
        return self.emit(code, message, severity="info", **kwargs)

    def error(self, error: DiagError) -> dict[str, Any]:
        return self.emit(
            error.code,
            error.message,
            field_path=error.field_path,
            details=error.details,
        )
