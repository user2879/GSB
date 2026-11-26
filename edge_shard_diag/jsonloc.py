from __future__ import annotations

import json
from json import JSONDecoder, JSONDecodeError
from pathlib import Path
from typing import Any

from .errors import Codes, DiagnosticError


class LocatedDict(dict):
    def __init__(self, *args: Any, line: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.line = line
        self.key_lines: dict[str, int] = {}
        self.value_lines: dict[str, int] = {}

    def field_line(self, key: str, *, value: bool = False) -> int:
        lines = self.value_lines if value else self.key_lines
        return int(lines.get(key, self.line))


class LocatedList(list):
    def __init__(self, *args: Any, item_lines: dict[int, int] | None = None) -> None:
        super().__init__(*args)
        self.item_lines = item_lines or {}

    def item_line(self, index: int) -> int:
        return int(self.item_lines.get(index, 1))


def line_at(text: str, pos: int) -> int:
    return text.count("\n", 0, max(0, pos)) + 1


def parse_located_json(text: str) -> Any:
    """Parse JSON while retaining opening lines for objects, keys and array items."""

    try:
        json.loads(text)
    except JSONDecodeError as exc:
        raise DiagnosticError(
            Codes.SPEC,
            f"invalid JSON: {exc.msg}",
            field="$",
            line=exc.lineno,
        ) from exc

    decoder = JSONDecoder()
    length = len(text)

    def skip_ws(pos: int) -> int:
        while pos < length and text[pos] in " \t\r\n":
            pos += 1
        return pos

    def parse(pos: int = 0) -> tuple[Any, int]:
        pos = skip_ws(pos)
        if pos >= length:
            raise DiagnosticError(Codes.SPEC, "unexpected end of JSON", line=line_at(text, pos))
        char = text[pos]

        if char == "{":
            obj = LocatedDict(line=line_at(text, pos))
            pos = skip_ws(pos + 1)
            if pos < length and text[pos] == "}":
                return obj, pos + 1
            while True:
                pos = skip_ws(pos)
                key_start = pos
                key, end = decoder.raw_decode(text, pos)
                if not isinstance(key, str):
                    raise DiagnosticError(Codes.SPEC, "object key must be a string", line=line_at(text, key_start))
                obj.key_lines[key] = line_at(text, key_start)
                pos = skip_ws(end)
                if pos >= length or text[pos] != ":":
                    raise DiagnosticError(Codes.SPEC, "expected ':'", field=key, line=line_at(text, pos))
                pos = skip_ws(pos + 1)
                obj.value_lines[key] = line_at(text, pos)
                value, pos = parse(pos)
                obj[key] = value
                pos = skip_ws(pos)
                if pos < length and text[pos] == ",":
                    pos += 1
                    continue
                if pos < length and text[pos] == "}":
                    return obj, pos + 1
                raise DiagnosticError(Codes.SPEC, "expected ',' or '}'", field=key, line=line_at(text, pos))

        if char == "[":
            arr = LocatedList()
            lines: dict[int, int] = {}
            pos = skip_ws(pos + 1)
            if pos < length and text[pos] == "]":
                arr.item_lines = lines
                return arr, pos + 1
            index = 0
            while True:
                pos = skip_ws(pos)
                lines[index] = line_at(text, pos)
                value, pos = parse(pos)
                arr.append(value)
                index += 1
                pos = skip_ws(pos)
                if pos < length and text[pos] == ",":
                    pos += 1
                    continue
                if pos < length and text[pos] == "]":
                    arr.item_lines = lines
                    return arr, pos + 1
                raise DiagnosticError(Codes.SPEC, "expected ',' or ']'", line=line_at(text, pos))

        value, end = decoder.raw_decode(text, pos)
        return value, end

    value, _ = parse(0)
    return value


def load_located_json(path: str | Path) -> tuple[Any, str]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DiagnosticError(Codes.SPEC, f"cannot load {path.name}: {exc}", source=str(path)) from exc
    try:
        data = parse_located_json(text)
    except DiagnosticError as exc:
        if exc.source is None:
            exc.source = str(path)
        raise
    return data, text
