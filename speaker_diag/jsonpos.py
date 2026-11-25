"""Small strict JSON parser that records source line numbers for every path.

The standard library validates JSON but discards locations.  E_SPEC needs the
first offending field and its source line, including repeated object keys such
as a worker's ``port``.  This parser only supplies locations; callers use the
standard library value model (dict/list/str/int/float/bool/None).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


@dataclass
class PositionedJSON:
    value: Any
    locations: dict[tuple[Any, ...], int]

    def line(self, path: tuple[Any, ...] | list[Any] | str, default: int = 1) -> int:
        if isinstance(path, str):
            path = tuple(part for part in path.split(".") if part)
        return self.locations.get(tuple(path), self.locations.get((), default))


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.length = len(text)
        self.index = 0
        self.locations: dict[tuple[Any, ...], int] = {}

    def line_at(self, index: int) -> int:
        return self.text.count("\n", 0, index) + 1

    def error(self, message: str) -> json.JSONDecodeError:
        return json.JSONDecodeError(message, self.text, self.index)

    def skip_ws(self) -> None:
        while self.index < self.length and self.text[self.index] in " \t\r\n":
            self.index += 1

    def parse(self) -> Any:
        self.skip_ws()
        if self.index >= self.length:
            raise self.error("empty JSON document")
        value = self.parse_value(())
        self.skip_ws()
        if self.index != self.length:
            raise self.error("extra data after JSON document")
        return value

    def parse_value(self, path: tuple[Any, ...]) -> Any:
        self.skip_ws()
        if self.index >= self.length:
            raise self.error("expected a JSON value")
        char = self.text[self.index]
        if char == "{":
            return self.parse_object(path)
        if char == "[":
            return self.parse_array(path)
        if char == '"':
            return self.parse_string()
        if char == "t" and self.text.startswith("true", self.index):
            self.index += 4
            return True
        if char == "f" and self.text.startswith("false", self.index):
            self.index += 5
            return False
        if char == "n" and self.text.startswith("null", self.index):
            self.index += 4
            return None
        match = _NUMBER.match(self.text, self.index)
        if match:
            self.index = match.end()
            token = match.group(0)
            if "." in token or "e" in token or "E" in token:
                return float(token)
            return int(token)
        raise self.error(f"unexpected character {char!r}")

    def parse_string(self) -> str:
        start = self.index
        self.index += 1
        escaped = False
        while self.index < self.length:
            char = self.text[self.index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                self.index += 1
                try:
                    return json.loads(self.text[start : self.index])
                except json.JSONDecodeError as exc:
                    raise self.error(str(exc)) from exc
            elif ord(char) < 0x20:
                raise self.error("control character in string")
            self.index += 1
        raise self.error("unterminated string")

    def parse_object(self, path: tuple[Any, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self.index += 1
        self.skip_ws()
        if self.index < self.length and self.text[self.index] == "}":
            self.index += 1
            return result
        while True:
            self.skip_ws()
            if self.index >= self.length or self.text[self.index] != '"':
                raise self.error("expected object key")
            key_start = self.index
            key = self.parse_string()
            child_path = path + (key,)
            self.locations[child_path] = self.line_at(key_start)
            self.skip_ws()
            if self.index >= self.length or self.text[self.index] != ":":
                raise self.error("expected ':' after object key")
            self.index += 1
            result[key] = self.parse_value(child_path)
            self.skip_ws()
            if self.index >= self.length:
                raise self.error("unterminated object")
            char = self.text[self.index]
            self.index += 1
            if char == "}":
                return result
            if char != ",":
                raise self.error("expected ',' or '}'")

    def parse_array(self, path: tuple[Any, ...]) -> list[Any]:
        result: list[Any] = []
        self.index += 1
        self.skip_ws()
        if self.index < self.length and self.text[self.index] == "]":
            self.index += 1
            return result
        item_index = 0
        while True:
            child_path = path + (item_index,)
            item_start = self.index
            self.locations[child_path] = self.line_at(item_start)
            result.append(self.parse_value(child_path))
            item_index += 1
            self.skip_ws()
            if self.index >= self.length:
                raise self.error("unterminated array")
            char = self.text[self.index]
            self.index += 1
            if char == "]":
                return result
            if char != ",":
                raise self.error("expected ',' or ']'")


def parse_positioned_json(text: str) -> PositionedJSON:
    parser = _Parser(text)
    value = parser.parse()
    parser.locations[()] = 1
    return PositionedJSON(value=value, locations=parser.locations)

