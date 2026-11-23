"""Business errors and structured NDJSON diagnostics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BUSINESS_CODES = {
    "E_PAGE",
    "E_ORDER",
    "E_LOG",
    "E_SNAPSHOT",
    "E_REPLAY",
    "E_BUDGET",
    "E_STATE",
    "E_GUI",
}

EXIT_CODES = {
    "E_PAGE": 10,
    "E_ORDER": 11,
    "E_LOG": 12,
    "E_SNAPSHOT": 13,
    "E_REPLAY": 14,
    "E_BUDGET": 15,
    "E_STATE": 16,
    "E_GUI": 17,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def monotonic_ms() -> int:
    return int(__import__("time").monotonic() * 1000)


@dataclass
class BusinessError(Exception):
    """A user-actionable business failure.

    ``context`` is deliberately a plain mapping so that the same payload is
    written to the NDJSON diagnostic log, returned by the CLI, and displayed
    in the Tkinter status area.
    """

    code: str
    message: str
    context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.code not in BUSINESS_CODES:
            raise ValueError(f"unknown business code: {self.code}")
        self.context = dict(self.context or {})
        super().__init__(self.message)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }


def append_diagnostic(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Append one JSON object to a diagnostic NDJSON file.

    Logging must never hide the original business failure.  Failures while
    creating the log are therefore swallowed by callers, not here.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": utc_now_iso(), **dict(record)}
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def report_error(error: BusinessError, log_path: str | os.PathLike[str], **extra: Any) -> None:
    record = {
        "level": "ERROR",
        **error.as_dict(),
        **dict(extra),
    }
    append_diagnostic(log_path, record)

