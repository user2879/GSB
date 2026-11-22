from __future__ import annotations

import ast
import json
import re
from typing import Any


class YamlCompatError(ValueError):
    pass


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    result: list[str] = []
    for character in line:
        if escaped:
            result.append(character)
            escaped = False
            continue
        if character == "\\" and quote is not None:
            result.append(character)
            escaped = True
            continue
        if quote is not None:
            result.append(character)
            if character == quote:
                quote = None
            continue
        if character in ("'", '"'):
            quote = character
            result.append(character)
            continue
        if character == "#":
            break
        result.append(character)
    return "".join(result).rstrip()


def _split_flow(value: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    quote: str | None = None
    depth = 0
    for character in value:
        if quote is not None:
            current.append(character)
            if character == quote:
                quote = None
            continue
        if character in ("'", '"'):
            quote = character
            current.append(character)
        elif character in "[{":
            depth += 1
            current.append(character)
        elif character in "]}":
            depth -= 1
            current.append(character)
        elif character == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if "".join(current).strip():
        items.append("".join(current).strip())
    return items


def _split_mapping(text: str) -> tuple[str, str] | None:
    quote: str | None = None
    depth = 0
    for index, character in enumerate(text):
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in ("'", '"'):
            quote = character
        elif character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
        elif character == ":" and depth == 0:
            key = text[:index].strip()
            if not key:
                return None
            return key, text[index + 1 :].strip()
    return None


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in ("null", "~"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in (".nan", "nan"):
        return float("nan")
    if lowered in (".inf", "+.inf", "inf", "+inf"):
        return float("inf")
    if lowered in ("-.inf", "-inf"):
        return float("-inf")
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(item) for item in _split_flow(body)]
    if text.startswith("{") and text.endswith("}"):
        body = text[1:-1].strip()
        if not body:
            return {}
        result: dict[str, Any] = {}
        for item in _split_flow(body):
            pair = _split_mapping(item)
            if pair is None:
                raise YamlCompatError(f"invalid inline mapping item: {item!r}")
            key, value = pair
            result[str(_parse_scalar(key))] = _parse_scalar(value)
        return result
    if text[0:1] in ("'", '"'):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise YamlCompatError(f"invalid quoted scalar: {text!r}") from exc
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text
    ):
        return float(text)
    if parsed := _try_json(text):
        return parsed
    return text


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def safe_load(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    parsed_json = _try_json(stripped)
    if parsed_json is not None:
        return parsed_json
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if raw_line.strip() in ("---", "..."):
            continue
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line[:indent]:
            raise YamlCompatError("tabs are not supported for YAML indentation")
        lines.append((indent, line.strip()))
    if not lines:
        return None

    def parse_block(position: int, indent: int) -> tuple[Any, int]:
        if position >= len(lines):
            return None, position
        is_list = lines[position][0] == indent and lines[position][1].startswith("-")
        if is_list:
            result_list: list[Any] = []
            while position < len(lines):
                current_indent, content = lines[position]
                if current_indent < indent:
                    break
                if current_indent != indent or not content.startswith("-"):
                    break
                rest = content[1:].strip()
                position += 1
                if not rest:
                    if position < len(lines) and lines[position][0] > indent:
                        item, position = parse_block(position, lines[position][0])
                    else:
                        item = None
                    result_list.append(item)
                    continue
                pair = _split_mapping(rest)
                if pair is None:
                    result_list.append(_parse_scalar(rest))
                    continue
                key, value_text = pair
                item_map: dict[str, Any] = {}
                if value_text == "":
                    if position < len(lines) and (
                        lines[position][0] > indent
                        or (
                            lines[position][0] == indent
                            and lines[position][1].startswith("-")
                        )
                    ):
                        item_map[key], position = parse_block(
                            position, lines[position][0]
                        )
                    else:
                        item_map[key] = None
                else:
                    item_map[key] = _parse_scalar(value_text)
                while position < len(lines) and lines[position][0] > indent:
                    child_indent, child_content = lines[position]
                    child_pair = _split_mapping(child_content)
                    if child_pair is None or child_content.startswith("-"):
                        break
                    child_key, child_value = child_pair
                    position += 1
                    if child_value == "":
                        if position < len(lines) and (
                            lines[position][0] > child_indent
                            or (
                                lines[position][0] == child_indent
                                and lines[position][1].startswith("-")
                            )
                        ):
                            item_map[child_key], position = parse_block(
                                position, lines[position][0]
                            )
                        else:
                            item_map[child_key] = None
                    else:
                        item_map[child_key] = _parse_scalar(child_value)
                result_list.append(item_map)
            return result_list, position

        result_map: dict[str, Any] = {}
        while position < len(lines):
            current_indent, content = lines[position]
            if current_indent < indent:
                break
            if current_indent != indent or content.startswith("-"):
                break
            pair = _split_mapping(content)
            if pair is None:
                raise YamlCompatError(f"invalid mapping line: {content!r}")
            key, value_text = pair
            position += 1
            if value_text == "":
                if position < len(lines) and (
                    lines[position][0] > indent
                    or (
                        lines[position][0] == indent
                        and lines[position][1].startswith("-")
                    )
                ):
                    value, position = parse_block(position, lines[position][0])
                else:
                    value = None
            else:
                value = _parse_scalar(value_text)
            result_map[key] = value
        return result_map, position

    result, _ = parse_block(0, lines[0][0])
    return result


def safe_dump(value: Any, **_: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
