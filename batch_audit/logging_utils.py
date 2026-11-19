"""Small append-only JSONL logger for operational audit trails."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .timeutil import utc_now


class JsonlLogger:
    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled

    def event(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": utc_now(), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
