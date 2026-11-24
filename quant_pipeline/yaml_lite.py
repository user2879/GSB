"""YAML loading with PyYAML when available and a deterministic fallback.

The generated ``.yaml`` files are JSON-compatible YAML, so they load in both
paths.  The fallback also accepts ordinary block mappings/sequences used by
small hand-written configurations, keeping the runtime dependency list at the
project requirement of ``numpy`` and ``pytest`` only.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


class YamlLiteError(ValueError):
    pass


def load_yaml(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return loads_yaml(text)
    try:
        result = yaml.safe_load(text)
    except Exception as exc:
        raise YamlLiteError(str(exc)) from exc
    return {} if result is None else result


def loads_yaml(text: str) -> Any:
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    parser = _BlockParser(text)
    return parser.parse()


class _BlockParser:
    def __init__(self, text: str) -> None:
        self.lines: list[tuple[int, str]] = []
        for raw in text.splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            if raw.strip() in ("---", "..."):
                continue
            expanded = raw.expandtabs(2)
            indent = len(expanded) - len(expanded.lstrip(" "))
            content = _strip_comment(expanded[indent:]).rstrip()
            if content:
                self.lines.append((indent, content))
        self.index = 0

    def parse(self) -> Any:
        if not self.lines:
            return {}
        value = self._parse_block(self.lines[0][0])
        if self.index != len(self.lines):
            raise YamlLiteError(f"unexpected YAML content at line item {self.index + 1}")
        return value

    def _parse_block(self, indent: int) -> Any:
        if self.index >= len(self.lines):
            return None
        line_indent, content = self.lines[self.index]
        if line_indent < indent:
            return None
        if content.startswith("- ") or content == "-":
            return self._parse_sequence(indent)
        return self._parse_mapping(indent)

    def _parse_sequence(self, indent: int) -> list[Any]:
        result: list[Any] = []
        while self.index < len(self.lines):
            line_indent, content = self.lines[self.index]
            if line_indent < indent or not (content.startswith("- ") or content == "-"):
                break
            if line_indent > indent:
                raise YamlLiteError(f"bad sequence indentation near item {self.index + 1}")
            rest = content[1:].strip()
            self.index += 1
            if not rest:
                if self.index < len(self.lines) and self.lines[self.index][0] > indent:
                    result.append(self._parse_block(self.lines[self.index][0]))
                else:
                    result.append(None)
                continue
            key_value = _split_key_value(rest)
            if key_value is None:
                result.append(_parse_scalar(rest))
                continue
            key, value_text = key_value
            item: dict[str, Any] = {}
            if value_text == "":
                if self.index < len(self.lines) and self.lines[self.index][0] > indent:
                    item[key] = self._parse_block(self.lines[self.index][0])
                else:
                    item[key] = None
            else:
                item[key] = _parse_scalar(value_text)
            while self.index < len(self.lines):
                sub_indent, sub_content = self.lines[self.index]
                if sub_indent <= indent:
                    break
                kv = _split_key_value(sub_content)
                if kv is None:
                    raise YamlLiteError(f"expected mapping entry near item {self.index + 1}")
                sub_key, sub_value = kv
                self.index += 1
                if sub_value == "":
                    if self.index < len(self.lines) and self.lines[self.index][0] > sub_indent:
                        item[sub_key] = self._parse_block(self.lines[self.index][0])
                    else:
                        item[sub_key] = None
                else:
                    item[sub_key] = _parse_scalar(sub_value)
            result.append(item)
        return result

    def _parse_mapping(self, indent: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while self.index < len(self.lines):
            line_indent, content = self.lines[self.index]
            if line_indent < indent:
                break
            if line_indent > indent:
                raise YamlLiteError(f"bad mapping indentation near item {self.index + 1}")
            kv = _split_key_value(content)
            if kv is None:
                raise YamlLiteError(f"expected mapping entry near item {self.index + 1}")
            key, value_text = kv
            self.index += 1
            if value_text == "":
                if self.index < len(self.lines) and self.lines[self.index][0] > indent:
                    result[key] = self._parse_block(self.lines[self.index][0])
                else:
                    result[key] = None
            else:
                result[key] = _parse_scalar(value_text)
        return result


def _strip_comment(text: str) -> str:
    quote: str | None = None
    escaped = False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None and (i == 0 or text[i - 1].isspace()):
            return text[:i]
    return text


def _split_key_value(text: str) -> tuple[str, str] | None:
    quote: str | None = None
    depth = 0
    for i, char in enumerate(text):
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif quote is None:
            if char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
            elif char == ":" and depth == 0:
                key = text[:i].strip()
                if not key:
                    return None
                return _unquote_key(key), text[i + 1 :].strip()
    return None


def _unquote_key(key: str) -> str:
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ("'", '"'):
        return key[1:-1]
    return key


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in ("null", "~"):
        return None
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if text.startswith(("[", "{", "'", '"')):
        try:
            return ast.literal_eval(_translate_flow_yaml(text))
        except (ValueError, SyntaxError):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            pass
    return text


def _translate_flow_yaml(text: str) -> str:
    # Support JSON-style flow values plus common true/false/null spellings.
    return re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", text)))

