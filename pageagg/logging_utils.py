"""Structured JSONL business logging."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import DomainError


def default_log_path() -> Path:
    override = os.environ.get("PAGEAGG_LOG_PATH")
    if override:
        return Path(override)
    return Path("artifacts") / "logs" / "business.jsonl"


def log_event(
    event: str,
    *,
    path: str | Path | None = None,
    level: str = "INFO",
    **fields: Any,
) -> dict[str, Any]:
    target = Path(path) if path else default_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    with open(target, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def log_domain_error(error: DomainError, *, path: str | Path | None = None, command: str | None = None) -> None:
    log_event(
        "business_error",
        path=path,
        level="ERROR",
        command=command,
        code=error.code,
        category=error.category,
        message=error.message,
        page_id=error.page_id,
        field=error.field,
        event_seq=error.event_seq,
        details=error.details,
    )

