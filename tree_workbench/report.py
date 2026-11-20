"""Atomic JSON report writing."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def write_json_report(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target

