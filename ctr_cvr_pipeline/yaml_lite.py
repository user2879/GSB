"""Small YAML subset loader used to avoid adding a second dependency.

The accepted subset deliberately covers the training configuration shape:
mappings, indented lists, inline lists, quoted scalars, numbers and booleans.
JSON is valid input as well.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import ConfigError


_INT_RE = re.compile(r"^[+-]?[0-9]+$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(?:[eE][+-]?[0-9]+)?$|^[+-]?[0-9]+[eE][+-]?[0-9]+$"
)


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for idx, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote is not None:
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None:
            return line[:idx]
    return line


def _split_inline(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for idx, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote is not None:
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
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _split_key_value(text: str) -> tuple[str, str] | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for idx, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote is not None:
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
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        elif char == ":" and depth == 0:
            key = text[:idx].strip()
            if not key:
                return None
            return key, text[idx + 1 :].strip()
    return None


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "":
        return None
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        if text.startswith('"'):
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise ConfigError("E_CONFIG_INVALID", "yaml", f"invalid quoted string: {exc}") from exc
        return text[1:-1].replace("''", "'")
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_parse_scalar(part) for part in _split_inline(inner)]
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        if not inner:
            return {}
        obj: dict[str, Any] = {}
        for part in _split_inline(inner):
            pair = _split_key_value(part)
            if pair is None:
                raise ConfigError("E_CONFIG_INVALID", "yaml", f"invalid inline mapping item: {part}")
            key, value = pair
            obj[_parse_scalar(key) if key[:1] in ("'", '"') else key] = _parse_scalar(value)
        return obj
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def loads_yaml(text: str) -> Any:
    try:
        if text.lstrip().startswith(("{", "[")):
            return json.loads(text)
    except json.JSONDecodeError:
        pass

    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw).rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        if "\t" in stripped[:indent]:
            raise ConfigError("E_CONFIG_INVALID", "yaml", "tabs are not supported for indentation")
        lines.append((indent, stripped.strip()))
    if not lines:
        return {}

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return None, index
        is_list = lines[index][0] == indent and lines[index][1].startswith("-")
        if is_list:
            result_list: list[Any] = []
            while index < len(lines):
                cur_indent, content = lines[index]
                if cur_indent < indent:
                    break
                if cur_indent != indent or not content.startswith("-"):
                    break
                item_text = content[1:].strip()
                index += 1
                if not item_text:
                    child_indent = lines[index][0] if index < len(lines) else indent + 2
                    item, index = parse_block(index, child_indent)
                    result_list.append(item)
                    continue
                pair = _split_key_value(item_text)
                if pair is not None and not item_text.startswith(("'", '"')):
                    key_text, value_text = pair
                    item_obj: dict[str, Any] = {}
                    key = _parse_scalar(key_text) if key_text[:1] in ("'", '"') else key_text
                    if value_text == "":
                        if index < len(lines) and lines[index][0] > indent:
                            child, index = parse_block(index, lines[index][0])
                            item_obj[key] = child
                        else:
                            item_obj[key] = None
                    else:
                        item_obj[key] = _parse_scalar(value_text)
                    while index < len(lines) and lines[index][0] > indent:
                        child_indent, child_content = lines[index]
                        if child_content.startswith("-"):
                            break
                        child_pair = _split_key_value(child_content)
                        if child_pair is None:
                            raise ConfigError(
                                "E_CONFIG_INVALID", "yaml", f"invalid mapping line: {child_content}"
                            )
                        child_key, child_value = child_pair
                        child_key = (
                            _parse_scalar(child_key) if child_key[:1] in ("'", '"') else child_key
                        )
                        index += 1
                        if child_value == "":
                            if index < len(lines) and lines[index][0] > child_indent:
                                child, index = parse_block(index, lines[index][0])
                                item_obj[child_key] = child
                            else:
                                item_obj[child_key] = None
                        else:
                            item_obj[child_key] = _parse_scalar(child_value)
                    result_list.append(item_obj)
                else:
                    result_list.append(_parse_scalar(item_text))
            return result_list, index

        result_map: dict[str, Any] = {}
        while index < len(lines):
            cur_indent, content = lines[index]
            if cur_indent < indent:
                break
            if cur_indent != indent or content.startswith("-"):
                break
            pair = _split_key_value(content)
            if pair is None:
                raise ConfigError("E_CONFIG_INVALID", "yaml", f"invalid mapping line: {content}")
            key_text, value_text = pair
            key = _parse_scalar(key_text) if key_text[:1] in ("'", '"') else key_text
            index += 1
            if value_text == "":
                if index < len(lines) and lines[index][0] > indent:
                    child, index = parse_block(index, lines[index][0])
                    result_map[key] = child
                else:
                    result_map[key] = None
            else:
                result_map[key] = _parse_scalar(value_text)
        return result_map, index

    result, consumed = parse_block(0, lines[0][0])
    if consumed != len(lines):
        raise ConfigError("E_CONFIG_INVALID", "yaml", f"unconsumed YAML near line group {consumed}")
    return result


def load_yaml_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return loads_yaml(handle.read())
