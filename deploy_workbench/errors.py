from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


E_SPEC = "E_SPEC"
E_ENV = "E_ENV"
E_BIND = "E_BIND"
E_GRID = "E_GRID"
E_REPLAY = "E_REPLAY"
E_LEDGER = "E_LEDGER"

EXIT_CODES = {
    E_SPEC: 5,
    E_ENV: 2,
    E_BIND: 3,
    E_GRID: 4,
    E_REPLAY: 6,
    E_LEDGER: 7,
}


@dataclass
class WorkbenchError(Exception):
    """A domain failure with a business code and a stable field location."""

    code: str
    message: str
    environment: str | None = None
    path: str = "$"
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    @property
    def parent_code(self) -> str:
        # All load-time failures reject the complete declaration.  The specific
        # code below E_SPEC identifies the injected/business failure class.
        if self.code in {E_ENV, E_BIND, E_GRID}:
            return E_SPEC
        return self.code

    def to_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "parent_code": self.parent_code,
            "message": self.message,
            "environment": self.environment,
            "path": self.path,
        }
        if self.details:
            report["details"] = dict(self.details)
        return report


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def append_structured_log(log_path: str | os.PathLike[str], event: Mapping[str, Any]) -> None:
    """Append one JSON object per line; never let logging hide the domain error."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": utc_timestamp(), **dict(event)}
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass


def log_error(log_path: str | os.PathLike[str], error: WorkbenchError, *, command: str | None = None) -> None:
    event: dict[str, Any] = {
        "level": "ERROR",
        "event": "domain_rejection",
        **error.to_report(),
    }
    if command:
        event["command"] = command
    append_structured_log(log_path, event)

