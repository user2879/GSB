"""带行号/字段路径定位能力的 JSON 解析器（标准库实现）。

标准库 ``json`` 不暴露字段位置，无法满足 E_FORMAT 必须给出行号与
字段路径的要求。这里用一个小型递归下降解析器，复用标准库的字符串
扫描函数保证转义语义一致，并为每个对象/数组/标量记录起始行号。
"""

from __future__ import annotations

import re
from json import JSONDecodeError
from json.decoder import scanstring
from pathlib import Path


class JsonPosError(ValueError):
    def __init__(self, message: str, line: int, column: int,
                 incomplete: bool = False) -> None:
        super().__init__(f"{message} (line {line}, column {column})")
        self.line = line
        self.column = column
        self.incomplete = incomplete


class IncompleteJsonError(JsonPosError):
    """文档在结构完成前结束（典型的文件尾部截断场景）。"""


class LocDict(dict):
    def __init__(self, line: int) -> None:
        super().__init__()
        self.line = line
        self.key_lines: dict[str, int] = {}


class LocList(list):
    def __init__(self, line: int) -> None:
        super().__init__()
        self.line = line
        self.item_lines: dict[int, int] = {}


_NUMBER_RE = re.compile(
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
)


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0
        self.length = len(text)

    def _line(self, index: int | None = None) -> int:
        if index is None:
            index = self.index
        return self.text.count("\n", 0, index) + 1

    def _column(self, index: int | None = None) -> int:
        if index is None:
            index = self.index
        last_newline = self.text.rfind("\n", 0, index)
        return index - last_newline

    def _skip_ws(self) -> None:
        while self.index < self.length and self.text[self.index] in " \t\r\n":
            self.index += 1

    def _fail(self, message: str, incomplete: bool = False) -> None:
        error_type = IncompleteJsonError if incomplete else JsonPosError
        raise error_type(message, self._line(), self._column(), incomplete)

    def parse(self) -> object:
        self._skip_ws()
        if self.index >= self.length:
            self._fail("空文档，无法解析 JSON", incomplete=True)
        value = self._parse_value()
        self._skip_ws()
        if self.index != self.length:
            self._fail(f"文档结束后仍有多余字符 {self.text[self.index]!r}")
        return value

    def _parse_value(self) -> object:
        self._skip_ws()
        if self.index >= self.length:
            self._fail("期望 JSON 值但文档结束", incomplete=True)
        char = self.text[self.index]
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array()
        if char == '"':
            value, _ = scanstring(self.text, self.index + 1, True)
            self.index = _
            return value
        if char == "t":
            return self._parse_literal("true", True)
        if char == "f":
            return self._parse_literal("false", False)
        if char == "n":
            return self._parse_literal("null", None)
        if char == "-" or char.isdigit():
            return self._parse_number()
        self._fail(f"非法 JSON 起始字符 {char!r}")

    def _parse_literal(self, literal: str, value: object) -> object:
        if not self.text.startswith(literal, self.index):
            self._fail(f"期望字面量 {literal!r}")
        self.index += len(literal)
        return value

    def _parse_number(self) -> int | float:
        match = _NUMBER_RE.match(self.text, self.index)
        if match is None:
            self._fail("非法数字")
        token = match.group(0)
        self.index = match.end()
        if self.index >= self.length:
            self._fail("数字在文档结束处被截断", incomplete=True)
        if "." in token or "e" in token or "E" in token:
            return float(token)
        return int(token)

    def _parse_object(self) -> LocDict:
        result = LocDict(self._line())
        self.index += 1
        self._skip_ws()
        if self._consume("}"):
            return result
        while True:
            self._skip_ws()
            if self.index >= self.length:
                self._fail("对象缺少闭合 '}'", incomplete=True)
            if self.text[self.index] != '"':
                self._fail("对象键必须是字符串")
            key_line = self._line()
            key, self.index = scanstring(self.text, self.index + 1, True)
            self._skip_ws()
            if not self._consume(":"):
                self._fail("对象键后缺少 ':'")
            value = self._parse_value()
            result[key] = value
            result.key_lines[key] = key_line
            self._skip_ws()
            if self._consume("}"):
                return result
            if not self._consume(","):
                self._fail("对象成员后缺少 ',' 或 '}'")

    def _parse_array(self) -> LocList:
        result = LocList(self._line())
        self.index += 1
        self._skip_ws()
        if self._consume("]"):
            return result
        while True:
            item_line = self._line()
            value = self._parse_value()
            result.item_lines[len(result)] = item_line
            result.append(value)
            self._skip_ws()
            if self._consume("]"):
                return result
            if not self._consume(","):
                self._fail("数组元素后缺少 ',' 或 ']'")

    def _consume(self, char: str) -> bool:
        if self.index < self.length and self.text[self.index] == char:
            self.index += 1
            return True
        if self.index >= self.length:
            self._fail(f"期望 {char!r} 但文档结束", incomplete=True)
        return False


def parse_text(text: str) -> object:
    try:
        return _Parser(text).parse()
    except JSONDecodeError as exc:
        raise IncompleteJsonError(
            f"JSON 字符串或结构被截断: {exc.msg}",
            exc.lineno, exc.colno, incomplete=True) from exc


def parse_file(path: str | Path) -> object:
    return parse_text(Path(path).read_text(encoding="utf-8"))


def locate(root: object, path_tokens: list[str | int]) -> int | None:
    """按字段路径（字典键或数组下标）返回该节点起始行号。"""
    current = root
    for token in path_tokens:
        if isinstance(token, str):
            if not isinstance(current, LocDict) or token not in current:
                return None
            line = current.key_lines.get(token)
            current = current[token]
        else:
            if not isinstance(current, LocList) or token >= len(current):
                return None
            line = current.item_lines.get(token)
            current = current[token]
    if isinstance(current, (LocDict, LocList)):
        return current.line
    return line
