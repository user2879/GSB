"""逻辑谓词的规范化、解析与分片键覆盖识别。

支持文法（无 OR，AND 可任意嵌套）::

    expr    := cond (AND cond)*
    cond    := column OP value
             | column IN "(" value ("," value)* ")"
    OP      := = | == | != | <> | < | <= | > | >=

规范化保证：空白差异、关键字大小写、IN 列表顺序、AND 子句顺序都不影响
规范化文本与路由标识。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .determinism import content_digest, stable_hash
from .errors import ConfigError

_TOKEN_RE = re.compile(
    r"""
      \s*(?:
        (?P<op><=|>=|<>|!=|==|=|<|>)
      | (?P<lparen>\()
      | (?P<rparen>\))
      | (?P<comma>,)
      | (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
      | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<number>-?\d+(?:\.\d+)?)
      )
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Condition:
    column: str
    operator: str
    values: tuple[Any, ...] = field(default_factory=tuple)

    def normalized_text(self) -> str:
        if self.operator == "IN":
            ordered = sorted(self.values, key=_literal_sort_key)
            rendered = ", ".join(_render_literal(value) for value in ordered)
            return f"{self.column} IN ({rendered})"
        return f"{self.column} {self.operator} {_render_literal(self.values[0])}"

    def columns(self) -> tuple[str]:
        return (self.column,)


@dataclass(frozen=True)
class Predicate:
    conditions: tuple[Condition, ...]

    def normalized_text(self) -> str:
        parts = sorted(condition.normalized_text() for condition in self.conditions)
        return " AND ".join(parts)

    def route_key(self) -> str:
        """稳定路由标识：规范化文本的短摘要，语义相同则标识相同。"""
        digest = content_digest(self.normalized_text())[:16]
        return f"rk_{digest}_{stable_hash(self.normalized_text()) % 1_000_000:06d}"

    def columns_present(self) -> frozenset[str]:
        columns: set[str] = set()
        for condition in self.conditions:
            columns.add(condition.column)
        return frozenset(columns)

    def covers_key(self, shard_key_columns: tuple[str, ...]) -> bool:
        present = self.columns_present()
        return all(column in present for column in shard_key_columns)

    def broadcast_reason(self, shard_key_columns: tuple[str, ...]) -> str:
        missing = [column for column in shard_key_columns if column not in self.columns_present()]
        return "谓词未覆盖完整分片键，缺失列: " + ", ".join(missing)

    def equality_map(self) -> dict[str, Any]:
        """返回 column -> 等值/IN 取值（多值 IN 返回排序后的列表）。"""
        result: dict[str, Any] = {}
        for condition in self.conditions:
            if condition.operator in ("=", "=="):
                result[condition.column] = condition.values[0]
            elif condition.operator == "IN":
                result[condition.column] = sorted(condition.values, key=_literal_sort_key)
        return result

    def range_conditions(self) -> list[Condition]:
        return [
            condition
            for condition in self.conditions
            if condition.operator in ("<", "<=", ">", ">=")
        ]


def _literal_sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _render_literal(value: Any) -> str:
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if value is None:
        return "NULL"
    return str(value)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match_whitespace = re.match(r"\s+", text[position:])
        if match_whitespace:
            position += match_whitespace.end()
            continue
        match = _TOKEN_RE.match(text, position)
        if not match:
            raise ConfigError(
                f"无法识别的谓词片段: {text[position:]!r}",
                paths=["predicate"],
            )
        position = match.end()
        kind = match.lastgroup
        value = match.group()
        if kind is None:
            continue
        tokens.append((kind, value.strip()))
    return tokens


def _coerce_literal(raw: str) -> Any:
    if raw[0] in ("'", '"'):
        inner = raw[1:-1].replace("''", "'")
        return inner
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    upper = raw.upper()
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    if upper == "NULL":
        return None
    return raw


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.index = 0

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _take(self) -> tuple[str, str]:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def parse(self) -> Predicate:
        conditions: list[Condition] = [self._parse_condition()]
        while True:
            token = self._peek()
            if token is None:
                break
            kind, value = token
            if kind == "ident" and value.upper() == "AND":
                self._take()
                conditions.append(self._parse_condition())
            else:
                raise ConfigError(
                    f"谓词条件之间缺少 AND，遇到 {value!r}",
                    paths=["predicate"],
                )
        return Predicate(tuple(conditions))

    def _parse_condition(self) -> Condition:
        token = self._peek()
        if token is None or token[0] != "ident":
            raise ConfigError("谓词条件必须以列名开始", paths=["predicate"])
        column = self._take()[1].lower()
        operator_token = self._peek()
        if operator_token is None:
            raise ConfigError(f"列 {column} 后缺少操作符", paths=["predicate"])
        kind, value = operator_token
        if kind == "ident" and value.upper() == "IN":
            return self._parse_in(column)
        if kind != "op":
            raise ConfigError(f"列 {column} 后应为比较操作符", paths=["predicate"])
        self._take()
        operator = "=" if value == "==" else ("!=" if value == "<>" else value)
        literal = self._parse_literal()
        return Condition(column=column, operator=operator, values=(literal,))

    def _parse_in(self, column: str) -> Condition:
        self._take()  # IN
        if self._peek() is None or self._peek()[0] != "lparen":
            raise ConfigError(f"{column} IN 后缺少左括号", paths=["predicate"])
        self._take()
        values: list[Any] = [self._parse_literal()]
        while self._peek() is not None and self._peek()[0] == "comma":
            self._take()
            values.append(self._parse_literal())
        if self._peek() is None or self._peek()[0] != "rparen":
            raise ConfigError(f"{column} IN 列表缺少右括号", paths=["predicate"])
        self._take()
        if not values:
            raise ConfigError(f"{column} IN 列表不能为空", paths=["predicate"])
        return Condition(column=column, operator="IN", values=tuple(values))

    def _parse_literal(self) -> Any:
        token = self._peek()
        if token is None:
            raise ConfigError("谓词缺少字面量", paths=["predicate"])
        kind, value = self._take()
        if kind not in ("string", "number", "ident"):
            raise ConfigError(f"非法字面量 {value!r}", paths=["predicate"])
        return _coerce_literal(value)


def parse_predicate(text: str) -> Predicate:
    if not isinstance(text, str) or not text.strip():
        raise ConfigError("谓词不能为空", paths=["predicate"])
    return _Parser(_tokenize(text)).parse()
