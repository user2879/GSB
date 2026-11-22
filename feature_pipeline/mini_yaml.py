"""Small dependency-free YAML subset used by the example configuration.

The project intentionally requires only NumPy.  This parser covers the YAML
features used by the repository examples and test fixtures: block mappings,
block sequences, nested values, quoted strings, comments, and inline lists or
maps.  JSON documents are valid input as well.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class YamlError(ValueError):
    """Raised when the supported YAML subset cannot be parsed."""


@dataclass(frozen=True)
class _Line:
    indent: int
    text: str
    number: int


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif quote == "'":
            if char == "'":
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index].rstrip()
    return line.rstrip()


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif quote == "'":
            if char == "'":
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _split_key_value(text: str) -> tuple[str, str]:
    quote: str | None = None
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif quote == "'":
            if char == "'":
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        elif char == ":" and depth == 0:
            key = text[:index].strip()
            if not key:
                raise YamlError("empty mapping key")
            return key, text[index + 1 :].strip()
    raise YamlError(f"expected 'key: value', got {text!r}")


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "":
        return None
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(part) for part in _split_top_level(body)]
    if text.startswith("{") and text.endswith("}"):
        body = text[1:-1].strip()
        result: dict[str, Any] = {}
        if not body:
            return result
        for part in _split_top_level(body):
            key, value = _split_key_value(part)
            result[str(_parse_scalar(key))] = _parse_scalar(value)
        return result
    if len(text) >= 2 and text[0] == text[-1] == '"':
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise YamlError(f"invalid double-quoted string: {text!r}") from exc
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1].replace("''", "'")
    lower = text.lower()
    if lower in {"null", "~"}:
        return None
    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    if re.fullmatch(r"[-+]?0[xX][0-9a-fA-F]+", text):
        return int(text, 0)
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)", text
    ) or re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", text):
        return float(text)
    return text


class _Parser:
    def __init__(self, lines: list[_Line]):
        self.lines = lines
        self.index = 0

    def parse(self) -> Any:
        if not self.lines:
            return None
        value = self._parse_block(self.lines[0].indent)
        if self.index != len(self.lines):
            line = self.lines[self.index]
            raise YamlError(f"unexpected content at line {line.number}: {line.text!r}")
        return value

    def _parse_block(self, indent: int) -> Any:
        if self.index >= len(self.lines):
            return None
        line = self.lines[self.index]
        if line.indent != indent:
            raise YamlError(f"bad indentation at line {line.number}")
        if line.text == "-" or line.text.startswith("- "):
            return self._parse_sequence(indent)
        return self._parse_mapping(indent)

    def _parse_nested(self, parent_indent: int) -> Any:
        if self.index >= len(self.lines) or self.lines[self.index].indent <= parent_indent:
            return None
        return self._parse_block(self.lines[self.index].indent)

    def _parse_mapping(self, indent: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while self.index < len(self.lines):
            line = self.lines[self.index]
            if line.indent < indent:
                break
            if line.indent > indent:
                raise YamlError(f"unexpected indentation at line {line.number}")
            if line.text == "-" or line.text.startswith("- "):
                break
            key_text, value_text = _split_key_value(line.text)
            key = str(_parse_scalar(key_text))
            if key in result:
                raise YamlError(f"duplicate key {key!r} at line {line.number}")
            self.index += 1
            if value_text:
                result[key] = _parse_scalar(value_text)
            else:
                result[key] = self._parse_nested(indent)
        return result

    def _parse_sequence_item_mapping(self, rest: str, item_indent: int) -> dict[str, Any]:
        key_text, value_text = _split_key_value(rest)
        result: dict[str, Any] = {}
        key = str(_parse_scalar(key_text))
        if value_text:
            result[key] = _parse_scalar(value_text)
        else:
            result[key] = self._parse_nested(item_indent)
        while self.index < len(self.lines):
            line = self.lines[self.index]
            if line.indent != item_indent + 2:
                break
            if line.text == "-" or line.text.startswith("- "):
                break
            key_text, value_text = _split_key_value(line.text)
            key = str(_parse_scalar(key_text))
            if key in result:
                raise YamlError(f"duplicate key {key!r} at line {line.number}")
            self.index += 1
            if value_text:
                result[key] = _parse_scalar(value_text)
            else:
                result[key] = self._parse_nested(item_indent + 2)
        return result

    def _parse_sequence(self, indent: int) -> list[Any]:
        result: list[Any] = []
        while self.index < len(self.lines):
            line = self.lines[self.index]
            if line.indent < indent:
                break
            if line.indent != indent:
                raise YamlError(f"unexpected indentation at line {line.number}")
            if not (line.text == "-" or line.text.startswith("- ")):
                break
            rest = line.text[1:].strip()
            self.index += 1
            if not rest:
                result.append(self._parse_nested(indent))
            elif ":" in rest and not rest.startswith(("[", "{", "'", '"')):
                result.append(self._parse_sequence_item_mapping(rest, indent))
            else:
                result.append(_parse_scalar(rest))
        return result


def loads(text: str) -> Any:
    """Parse a YAML subset (or a complete JSON document)."""

    stripped = text.lstrip("\ufeff \t\r\n")
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    lines: list[_Line] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise YamlError(f"tabs are not allowed for indentation at line {number}")
        cleaned = _strip_comment(raw)
        if not cleaned.strip() or cleaned.strip() in {"---", "..."}:
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        lines.append(_Line(indent=indent, text=cleaned.strip(), number=number))
    if not lines:
        return None
    return _Parser(lines).parse()


def load(path: str | Path) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise YamlError(f"cannot read {path}: {exc}") from exc
    try:
        return loads(text)
    except YamlError as exc:
        raise YamlError(f"{path}: {exc}") from exc
