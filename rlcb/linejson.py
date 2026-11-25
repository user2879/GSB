"""Line-number aware JSON parsing.

Two passes over the same text:

1. ``json.loads`` produces the fully parsed value (correct number, escape,
   Unicode and nesting semantics).
2. A small structural tokenizer records the 1-based start line of every
   object and of each member key/value. It only tracks structural characters
   and string spans (with escape handling); scalar grammar is never
   reimplemented.

The two trees have identical shape and are produced in the same pre-order, so
locations are zipped back onto parsed values as :class:`Located` dicts.
"""

from __future__ import annotations

import bisect
import json
from typing import Any


class Located(dict):
    """Parsed object value plus source-line metadata."""


def _line_starts(text: str) -> list[int]:
    offsets = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def _line_for(offsets: list[int], target: int) -> int:
    return bisect.bisect_right(offsets, target)


def _skip_ws(text: str, idx: int) -> int:
    n = len(text)
    while idx < n and text[idx] in " \t\r\n":
        idx += 1
    return idx


def _skip_string(text: str, idx: int) -> int:
    idx += 1  # past opening quote
    n = len(text)
    while idx < n:
        ch = text[idx]
        if ch == "\\":
            idx += 2
            continue
        if ch == '"':
            return idx + 1
        idx += 1
    raise json.JSONDecodeError("unterminated string", text, idx)


def _skip_scalar(text: str, idx: int) -> int:
    n = len(text)
    while idx < n and text[idx] not in " \t\r\n,}]":
        idx += 1
    return idx


def parse_with_lines(text: str) -> Any:
    offsets = _line_starts(text)
    parsed = json.loads(text)
    records: list[tuple[int, dict[str, tuple[int, int]]]] = []

    def walk(idx: int) -> int:
        idx = _skip_ws(text, idx)
        ch = text[idx]
        if ch == "[":
            idx += 1
            while True:
                idx = _skip_ws(text, idx)
                if text[idx] == "]":
                    return idx + 1
                idx = walk(idx)
                idx = _skip_ws(text, idx)
                if text[idx] == ",":
                    idx += 1
                else:
                    return idx + 1
        if ch == "{":
            start = idx
            members: dict[str, tuple[int, int]] = {}
            records.append((start, members))  # pre-order registration
            idx += 1
            while True:
                idx = _skip_ws(text, idx)
                if text[idx] == "}":
                    return idx + 1
                key_line = _line_for(offsets, idx)
                key_end = _skip_string(text, idx)
                key = json.loads(text[idx:key_end])
                idx = _skip_ws(text, key_end)
                idx += 1  # ':'
                idx = _skip_ws(text, idx)
                value_line = _line_for(offsets, idx)
                members[key] = (key_line, value_line)
                idx = walk(idx)
                idx = _skip_ws(text, idx)
                if text[idx] == ",":
                    idx += 1
                else:
                    return idx + 1
        return _skip_scalar(text, idx)

    walk(0)
    queue = list(records)

    def merge(node: Any) -> Any:
        if isinstance(node, dict):
            if not queue:
                raise json.JSONDecodeError("location tree mismatch", text, 0)
            start, members = queue.pop(0)
            located = Located()
            located["__line__"] = _line_for(offsets, start)
            key_lines: dict[str, int] = {}
            value_lines: dict[str, int] = {}
            for key, child in node.items():
                kl, vl = members[key]
                key_lines[key] = kl
                value_lines[key] = vl
                located[key] = merge(child)
            located["__key_lines__"] = key_lines
            located["__value_lines__"] = value_lines
            return located
        if isinstance(node, list):
            return [merge(child) for child in node]
        return node

    return merge(parsed)


def object_line(obj: Any) -> int | None:
    if isinstance(obj, Located):
        return obj.get("__line__")
    return None


def key_line(obj: Any, key: str) -> int | None:
    if isinstance(obj, Located):
        return obj.get("__key_lines__", {}).get(key)
    return None


def value_line(obj: Any, key: str) -> int | None:
    if isinstance(obj, Located):
        return obj.get("__value_lines__", {}).get(key)
    return None
