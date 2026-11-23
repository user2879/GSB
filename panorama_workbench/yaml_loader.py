"""Small YAML loader with an optional PyYAML fast path.

The fallback intentionally implements the YAML subset used by workbench
configuration files: indentation based mappings/lists, quoted and plain
scalars, inline lists/mappings, comments, booleans and nulls.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def load_yaml_text(text: str) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _FallbackYamlParser(text).parse()
    return yaml.safe_load(text)


def load_yaml_file(path: str | Path) -> Any:
    source = Path(path)
    return load_yaml_text(source.read_text(encoding="utf-8"))


class _FallbackYamlParser:
    def __init__(self, text: str) -> None:
        self.lines: list[tuple[int, str]] = []
        for raw in text.splitlines():
            stripped = _strip_comment(raw).rstrip()
            if not stripped.strip() or stripped.lstrip().startswith("---"):
                continue
            indent = len(stripped) - len(stripped.lstrip(" "))
            self.lines.append((indent, stripped.strip()))
        self.index = 0

    def parse(self) -> Any:
        if not self.lines:
            return {}
        value = self._parse_block(self.lines[0][0])
        if self.index != len(self.lines):
            raise ValueError(f"unexpected YAML content near {self.lines[self.index]}")
        return value

    def _parse_block(self, indent: int) -> Any:
        if self.index >= len(self.lines):
            return {}
        if self.lines[self.index][1].startswith("- "):
            return self._parse_list(indent)
        return self._parse_mapping(indent)

    def _parse_mapping(self, indent: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while self.index < len(self.lines):
            current_indent, text = self.lines[self.index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"unexpected indentation: {text}")
            if text.startswith("- "):
                break
            key, separator, tail = text.partition(":")
            if not separator:
                raise ValueError(f"expected mapping entry: {text}")
            key = _parse_scalar(key.strip())
            if not isinstance(key, str):
                key = str(key)
            self.index += 1
            if tail.strip():
                result[key] = _parse_scalar(tail.strip())
            elif self.index < len(self.lines) and self.lines[self.index][0] > indent:
                result[key] = self._parse_block(self.lines[self.index][0])
            else:
                result[key] = None
        return result

    def _parse_list(self, indent: int) -> list[Any]:
        result: list[Any] = []
        while self.index < len(self.lines):
            current_indent, text = self.lines[self.index]
            if current_indent < indent:
                break
            if current_indent != indent or not text.startswith("- "):
                break
            payload = text[2:].strip()
            self.index += 1
            if not payload:
                if self.index < len(self.lines) and self.lines[self.index][0] > indent:
                    result.append(self._parse_block(self.lines[self.index][0]))
                else:
                    result.append(None)
                continue
            if payload.startswith(("{", "[")):
                result.append(_parse_scalar(payload))
                continue
            if _looks_like_mapping(payload):
                key, _, tail = payload.partition(":")
                item: dict[str, Any] = {}
                if tail.strip():
                    item[str(_parse_scalar(key.strip()))] = _parse_scalar(tail.strip())
                elif self.index < len(self.lines) and self.lines[self.index][0] > indent:
                    child_indent = self.lines[self.index][0]
                    if self.lines[self.index][1].startswith("- "):
                        item[str(_parse_scalar(key.strip()))] = self._parse_list(child_indent)
                    else:
                        item[str(_parse_scalar(key.strip()))] = self._parse_block(child_indent)
                else:
                    item[str(_parse_scalar(key.strip()))] = None
                if self.index < len(self.lines) and self.lines[self.index][0] > indent:
                    continuation_indent = self.lines[self.index][0]
                    if not self.lines[self.index][1].startswith("- "):
                        continuation = self._parse_mapping(continuation_indent)
                        item.update(continuation)
                result.append(item)
            else:
                result.append(_parse_scalar(payload))
        return result


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
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def _looks_like_mapping(text: str) -> bool:
    quote: str | None = None
    for index, char in enumerate(text):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == ":" and quote is None:
            return True
    return False


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if value.startswith(("[", "{")):
        try:
            return _FlowParser(value).parse()
        except ValueError:
            pass
    if value.startswith("("):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value[1:-1]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    return value


class _FlowParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def parse(self) -> Any:
        value = self._value()
        self._skip_space()
        if self.index != len(self.text):
            raise ValueError("trailing YAML flow content")
        return value

    def _value(self) -> Any:
        self._skip_space()
        if self.index >= len(self.text):
            raise ValueError("unexpected end of flow value")
        char = self.text[self.index]
        if char == "[":
            return self._list()
        if char == "{":
            return self._mapping()
        if char in {"'", '"'}:
            return self._quoted()
        return self._bare()

    def _list(self) -> list[Any]:
        self.index += 1
        result: list[Any] = []
        self._skip_space()
        if self._peek() == "]":
            self.index += 1
            return result
        while True:
            result.append(self._value())
            self._skip_space()
            char = self._peek()
            if char == ",":
                self.index += 1
                continue
            if char == "]":
                self.index += 1
                return result
            raise ValueError("expected comma or closing bracket")

    def _mapping(self) -> dict[str, Any]:
        self.index += 1
        result: dict[str, Any] = {}
        self._skip_space()
        if self._peek() == "}":
            self.index += 1
            return result
        while True:
            key = self._value()
            self._skip_space()
            if self._peek() != ":":
                raise ValueError("expected colon in flow mapping")
            self.index += 1
            result[str(key)] = self._value()
            self._skip_space()
            char = self._peek()
            if char == ",":
                self.index += 1
                continue
            if char == "}":
                self.index += 1
                return result
            raise ValueError("expected comma or closing brace")

    def _quoted(self) -> str:
        quote = self.text[self.index]
        start = self.index
        self.index += 1
        escaped = False
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if escaped:
                escaped = False
                continue
            if char == "\\" and quote == '"':
                escaped = True
                continue
            if char == quote:
                raw = self.text[start : self.index]
                try:
                    return str(ast.literal_eval(raw))
                except (ValueError, SyntaxError):
                    return raw[1:-1]
        raise ValueError("unterminated quoted string")

    def _bare(self) -> Any:
        start = self.index
        while self.index < len(self.text):
            char = self.text[self.index]
            if char in {",", "]", "}", ":"}:
                break
            self.index += 1
        token = self.text[start : self.index].strip()
        if not token:
            raise ValueError("empty flow value")
        return _parse_scalar_token(token)

    def _peek(self) -> str:
        if self.index >= len(self.text):
            return ""
        return self.text[self.index]

    def _skip_space(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1


def _parse_scalar_token(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    return value
