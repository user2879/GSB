from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_json


class StructuredLogger:
    def __init__(self, workspace: Path | str, run_id: str | None = None):
        self.workspace = Path(workspace).resolve(strict=False)
        self.path = self.workspace / "logs" / "slowdiag.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex

    def log(
        self,
        event: str,
        *,
        code: str = "OK",
        severity: str = "info",
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "monotonic_ns": time.monotonic_ns(),
            "run_id": self.run_id,
            "event": event,
            "severity": severity,
            "code": code,
            "message": message,
        }
        if details:
            record["details"] = details
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(record) + "\n")
        return record


def read_jsonl(path: Path | str, limit: int = 50) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.is_file():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()]
