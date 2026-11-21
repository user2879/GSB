"""Minimal flat YAML parser / dumper.

The project only needs single-level ``key: value`` mappings with scalar
values (int / float / bool / string).  Implementing this locally keeps the
dependency surface at Python stdlib + NumPy, as required.
"""
from __future__ import annotations

from pathlib import Path


def _coerce(text: str):
    text = text.strip()
    if text == "":
        return ""
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def loads(content: str) -> dict:
    result: dict = {}
    for lineno, raw in enumerate(content.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip() if not _quoted_hash(raw) else raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid yaml line {lineno}: {raw!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key on line {lineno}")
        result[key] = _coerce(value)
    return result


def _quoted_hash(line: str) -> bool:
    quote = None
    for ch in line:
        if ch in ("'", '"'):
            quote = None if quote == ch else (quote or ch)
        if ch == "#" and quote is None:
            return False
    return quote is not None


def load(path: str | Path) -> dict:
    return loads(Path(path).read_text(encoding="utf-8"))


def dumps(data: dict) -> str:
    parts = []
    for key in sorted(data):
        value = data[key]
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        elif isinstance(value, float):
            rendered = repr(value)
        else:
            rendered = str(value)
        parts.append(f"{key}: {rendered}")
    return "\n".join(parts) + "\n"


def dump(data: dict, path: str | Path) -> None:
    Path(path).write_text(dumps(data), encoding="utf-8")
