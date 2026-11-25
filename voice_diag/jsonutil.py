"""Line-aware JSON/JSONL validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ConfigFailure


class LineLocator:
    def __init__(self, text: str):
        self.lines = text.splitlines()

    def line_for(self, *needles: str, start: int = 1) -> int | None:
        for index, line in enumerate(self.lines, 1):
            if index < start:
                continue
            if all(needle in line for needle in needles):
                return index
        if needles:
            for index, line in enumerate(self.lines, 1):
                if index >= start and needles[-1] in line:
                    return index
        return None


def parse_json_text(path: Path, text: str, failure_sink: list) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        failure_sink.append(
            ConfigFailure(str(path), exc.lineno, "json", f"invalid JSON at column {exc.colno}: {exc.msg}").failure
        )
        return None


def parse_jsonl(path: Path, text: str, failure_sink: list) -> list[tuple[int, Any]]:
    rows: list[tuple[int, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            rows.append((line_no, json.loads(stripped)))
        except json.JSONDecodeError as exc:
            failure_sink.append(
                ConfigFailure(str(path), line_no, "json", f"invalid JSONL at column {exc.colno}: {exc.msg}").failure
            )
    return rows


def require_dict(value: Any, source: str, line: int | None, field: str, sink: list) -> bool:
    if isinstance(value, dict):
        return True
    sink.append(ConfigFailure(source, line, field, "expected object").failure)
    return False


def require_string(value: Any, source: str, line: int | None, field: str, sink: list) -> bool:
    if isinstance(value, str) and value.strip():
        return True
    sink.append(ConfigFailure(source, line, field, "expected non-empty string").failure)
    return False


def require_int(value: Any, source: str, line: int | None, field: str, sink: list) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    sink.append(ConfigFailure(source, line, field, "expected integer").failure)
    return False
