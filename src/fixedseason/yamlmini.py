"""A deliberately small, dependency-free YAML subset parser.

Supported syntax is the mapping/list/scalar subset needed by the experiment
configuration: indent-based mappings, block scalar lists, inline lists,
quoted strings, comments, booleans, null and Python-style JSON numbers.
Unsupported or ambiguous syntax is rejected with a line number.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_INT_RE = re.compile(r"^[+-]?[0-9][0-9_]*$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:[0-9][0-9_]*\.[0-9_]*|[0-9][0-9_]*)(?:[eE][+-]?[0-9]+)?$"
)


class YamlError(ValueError):
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"line {line}: {message}")
        self.line = line


@dataclass(frozen=True)
class _Line:
    indent: int
    text: str
    number: int


def _strip_comment(raw: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None:
            return raw[:index]
    return raw


def _tokenize(text: str) -> list[_Line]:
    lines: list[_Line] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise YamlError("tab indentation is not supported", number)
        stripped = _strip_comment(raw).rstrip()
        if not stripped.strip():
            continue
        content = stripped.lstrip(" ")
        if content in ("---", "..."):
            continue
        indent = len(stripped) - len(content)
        lines.append(_Line(indent=indent, text=content, number=number))
    return lines


def _split_mapping(text: str, line: int) -> tuple[str, str]:
    quote: str | None = None
    bracket_depth = 0
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if quote is not None:
            continue
        if char in "[{":
            bracket_depth += 1
        elif char in "]}":
            bracket_depth -= 1
        elif char == ":" and bracket_depth == 0:
            key = text[:index].strip()
            value = text[index + 1 :].strip()
            if not key:
                raise YamlError("mapping key is empty", line)
            return key, value
    raise YamlError("expected 'key: value'", line)


def _split_inline(text: str, separator: str, line: int) -> list[str]:
    parts: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if quote is not None:
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _parse_scalar(text: str, line: int) -> Any:
    text = text.strip()
    if not text:
        return None
    if text.startswith('"'):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise YamlError("invalid double-quoted string", line) from exc
    if text.startswith("'"):
        if len(text) < 2 or not text.endswith("'"):
            raise YamlError("unterminated single-quoted string", line)
        return text[1:-1].replace("''", "'")
    lowered = text.lower()
    if lowered in ("null", "~"):
        return None
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part, line) for part in _split_inline(inner, ",", line)]
    if _INT_RE.match(text):
        try:
            return int(text.replace("_", ""))
        except ValueError as exc:
            raise YamlError("invalid integer", line) from exc
    if _FLOAT_RE.match(text):
        try:
            return float(text.replace("_", ""))
        except ValueError as exc:
            raise YamlError("invalid number", line) from exc
    if text[0] in "[{":
        raise YamlError("unsupported or malformed inline collection", line)
    return text


def _parse_block(lines: list[_Line], position: int, indent: int) -> tuple[Any, int]:
    if position >= len(lines):
        return {}, position
    if lines[position].indent < indent:
        return {}, position
    if lines[position].indent > indent:
        raise YamlError("unexpected indentation", lines[position].number)
    is_sequence = lines[position].text.startswith("-")
    if is_sequence:
        result_list: list[Any] = []
        while position < len(lines):
            line = lines[position]
            if line.indent < indent:
                break
            if line.indent > indent:
                raise YamlError("unexpected indentation in list", line.number)
            if not line.text.startswith("-"):
                break
            rest = line.text[1:].strip()
            if not rest:
                if position + 1 < len(lines) and lines[position + 1].indent > indent:
                    value, position = _parse_block(lines, position + 1, lines[position + 1].indent)
                    result_list.append(value)
                else:
                    result_list.append(None)
                    position += 1
            else:
                result_list.append(_parse_scalar(rest, line.number))
                position += 1
        return result_list, position

    result_map: dict[str, Any] = {}
    while position < len(lines):
        line = lines[position]
        if line.indent < indent:
            break
        if line.indent > indent:
            raise YamlError("unexpected indentation in mapping", line.number)
        if line.text.startswith("-"):
            break
        key_text, value_text = _split_mapping(line.text, line.number)
        key = _parse_scalar(key_text, line.number)
        if not isinstance(key, str):
            raise YamlError("mapping keys must be strings", line.number)
        if key in result_map:
            raise YamlError(f"duplicate key {key!r}", line.number)
        position += 1
        if value_text:
            result_map[key] = _parse_scalar(value_text, line.number)
        else:
            if position < len(lines) and lines[position].indent > indent:
                result_map[key], position = _parse_block(
                    lines, position, lines[position].indent
                )
            else:
                result_map[key] = None
    return result_map, position


def loads(text: str) -> Any:
    lines = _tokenize(text)
    if not lines:
        return {}
    value, position = _parse_block(lines, 0, lines[0].indent)
    if position != len(lines):
        raise YamlError("could not parse entire document", lines[position].number)
    return value
