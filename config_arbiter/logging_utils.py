from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, *, level: str = "INFO", **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
