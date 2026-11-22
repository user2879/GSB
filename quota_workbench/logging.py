"""Deterministic newline-delimited structured logs without wall-clock fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import canonical_json


def append_structured_log(path: Path | str, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sequence = 1
    if target.exists():
        with target.open("r", encoding="utf-8") as handle:
            sequence = sum(1 for line in handle if line.strip()) + 1
    item = dict(payload)
    item["log_sequence"] = sequence
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(item) + "\n")
    return target
