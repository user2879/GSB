"""Append-only JSON-lines diagnostic logging."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json

_LOCK = threading.Lock()


def append_structured_log(
    path: str | Path | None,
    *,
    code: str,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    if path is None:
        return
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "code": code,
        "severity": severity,
        "message": message,
        "details": details or {},
        "monotonic_ns": time.monotonic_ns(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _LOCK:
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")
