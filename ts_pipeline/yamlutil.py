"""A small dependency-free YAML subset parser.

The project uses only the Python standard library and NumPy. This parser
supports the constructs used by this pipeline: block mappings/lists, inline
JSON collections, quoted strings, comments, numbers, booleans and null.
JSON documents are also accepted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class YamlError(ValueError):
    pass


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


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
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#":
            return line[:index]
    return line


def _split_key_value(text: str) -> tuple[str, str] | None:
    quote: str | None = None
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "[{(":
            depth += 1
            continue
        if char in "]})":
            depth -= 1
            continue
        if char == ":" and depth == 0:
            key = text[:index].strip()
            if not key:
                raise YamlError("empty mapping key")
            return key, text[index + 1 :].strip()
    return None


def _split_inline_items(text: str) -> list[str]:
    items: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(text[start:index].strip())
            start = index + 1
    items.append(text[start:].strip())
    return items


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "":
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(item) for item in _split_inline_items(inner)]
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise YamlError(f"invalid inline mapping: {text}") from exc
    if text.startswith('"') and text.endswith('"'):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise YamlError(f"invalid quoted string: {text}") from exc
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1].replace("''", "'")
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


class _Parser:
    def __init__(self, text: str):
        self.lines: list[tuple[int, str, int]] = []
        for number, raw in enumerate(text.splitlines(), start=1):
            leading = raw[: len(raw) - len(raw.lstrip())]
            if "\t" in leading:
                raise YamlError(f"line {number}: tabs are not allowed for indentation")
            content = _strip_comment(raw).rstrip()
            if not content.strip():
                continue
            stripped = content.lstrip()
            if stripped.startswith("---") or stripped.startswith("..."):
                continue
            indent = len(content) - len(stripped)
            self.lines.append((indent, stripped, number))

    def parse(self) -> Any:
        if not self.lines:
            return {}
        value, index = self._parse_block(0, self.lines[0][0])
        if index != len(self.lines):
            _, _, number = self.lines[index]
            raise YamlError(f"line {number}: unexpected indentation")
        return value

    def _parse_block(self, index: int, indent: int) -> tuple[Any, int]:
        if index >= len(self.lines):
            return {}, index
        current_indent, content, _ = self.lines[index]
        if current_indent != indent:
            raise YamlError("inconsistent indentation")
        if content.startswith("-"):
            return self._parse_list(index, indent)
        return self._parse_mapping(index, indent)

    def _parse_mapping(self, index: int, indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while index < len(self.lines):
            current_indent, content, number = self.lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise YamlError(f"line {number}: unexpected indentation")
            if content.startswith("-"):
                break
            pair = _split_key_value(content)
            if pair is None:
                raise YamlError(f"line {number}: expected key: value")
            key, raw_value = pair
            index += 1
            if raw_value:
                result[key] = _parse_scalar(raw_value)
                continue
            if index < len(self.lines) and self.lines[index][0] > indent:
                result[key], index = self._parse_block(index, self.lines[index][0])
            else:
                result[key] = None
        return result, index

    def _parse_list(self, index: int, indent: int) -> tuple[list[Any], int]:
        result: list[Any] = []
        while index < len(self.lines):
            current_indent, content, _ = self.lines[index]
            if current_indent < indent:
                break
            if current_indent != indent or not content.startswith("-"):
                break
            item_text = content[1:].strip()
            index += 1
            if not item_text:
                if index < len(self.lines) and self.lines[index][0] > indent:
                    item, index = self._parse_block(index, self.lines[index][0])
                else:
                    item = None
                result.append(item)
                continue
            pair = _split_key_value(item_text)
            if pair is None:
                result.append(_parse_scalar(item_text))
                continue
            key, raw_value = pair
            item_map: dict[str, Any] = {}
            if raw_value:
                item_map[key] = _parse_scalar(raw_value)
            elif index < len(self.lines) and self.lines[index][0] > indent:
                item_map[key], index = self._parse_block(index, self.lines[index][0])
            else:
                item_map[key] = None
            if index < len(self.lines) and self.lines[index][0] > indent:
                continuation_indent = self.lines[index][0]
                continuation, index = self._parse_mapping(index, continuation_indent)
                item_map.update(continuation)
            result.append(item_map)
        return result, index


def loads_yaml(text: str) -> Any:
    stripped = text.lstrip("\ufeff \r\n")
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return _Parser(text).parse()


def load_yaml_file(path: Path) -> Any:
    return loads_yaml(path.read_text(encoding="utf-8"))
