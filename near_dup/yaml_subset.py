"""Small dependency-free YAML subset parser for the sample configuration.

The accepted grammar intentionally stays narrow: a top-level mapping whose
values are scalars, inline lists, or indented block lists.  This is sufficient
for the component's flat fingerprint configuration and keeps PyYAML out of the
runtime dependency set.
"""

from __future__ import annotations

import re
from typing import Any


class YamlSubsetError(ValueError):
    pass


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "#":
            return line[:index]
    return line


def _split_inline(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in value:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and quote == '"':
            current.append(char)
            escaped = True
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == ",":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"nan", ".nan", "+nan", "-nan"}:
        return float("nan")
    if lowered in {"inf", ".inf", "+inf"}:
        return float("inf")
    if lowered in {"-inf", "-.inf"}:
        return float("-inf")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        if text[0] == "'":
            return text[1:-1].replace("''", "'")
        try:
            import json

            return json.loads(text)
        except Exception as exc:  # pragma: no cover - defensive parser path
            raise YamlSubsetError(f"invalid quoted scalar: {text}") from exc
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_parse_scalar(item) for item in _split_inline(inner)]
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", text):
        return float(text)
    return text


def loads(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    active_list_key: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).rstrip()
        if not line.strip() or line.strip() in {"---", "..."}:
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("- "):
            if active_list_key is None or indent == 0:
                raise YamlSubsetError(f"line {line_number}: list item outside a list")
            current = result.get(active_list_key)
            if not isinstance(current, list):
                raise YamlSubsetError(f"line {line_number}: block list target is not a list")
            current.append(_parse_scalar(stripped[2:]))
            continue
        if indent != 0 or ":" not in stripped:
            raise YamlSubsetError(f"line {line_number}: expected a top-level key")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise YamlSubsetError(f"line {line_number}: empty key")
        if key in result:
            raise YamlSubsetError(f"line {line_number}: duplicate key {key!r}")
        value = raw_value.strip()
        if value == "":
            result[key] = []
            active_list_key = key
        else:
            result[key] = _parse_scalar(value)
            active_list_key = None
    return result


def load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return loads(handle.read())

