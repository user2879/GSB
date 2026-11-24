from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .errors import AppError
from .util import canonical_json


_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<OP>>=|<=|==|!=|=|>|<)
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<COMMA>,)
  | (?P<NUMBER>-?\d+)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int


def _query_error(message: str, *, query_id: str, field: str = "where") -> AppError:
    return AppError(
        "E_SPEC",
        message,
        block_id=f"query:{query_id}",
        field=field,
    )


def _lex(text: str, *, query_id: str) -> list[Token]:
    tokens: list[Token] = []
    position = 0
    while position < len(text):
        match = _TOKEN_RE.match(text, position)
        if not match:
            raise _query_error(
                f"无法识别的谓词字符 {text[position]!r}，位置 {position}",
                query_id=query_id,
            )
        kind = match.lastgroup
        assert kind is not None
        value = match.group(0)
        if kind != "WS":
            if kind == "IDENT" and value.upper() in {"AND", "IN"}:
                kind = value.upper()
            else:
                kind = "IDENT" if kind == "IDENT" else kind
            tokens.append(Token(kind, value, position))
        position = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token], query_id: str) -> None:
        self.tokens = tokens
        self.query_id = query_id
        self.index = 0

    def peek(self, offset: int = 0) -> Token | None:
        index = self.index + offset
        if index >= len(self.tokens):
            return None
        return self.tokens[index]

    def take(self, expected: str | None = None) -> Token:
        token = self.peek()
        if token is None:
            raise self.error("谓词意外结束")
        if expected is not None and token.kind != expected:
            raise self.error(f"期望 {expected}，实际为 {token.value!r}")
        self.index += 1
        return token

    def error(self, message: str) -> AppError:
        token = self.peek()
        where = f"，位置 {token.position}" if token else ""
        return _query_error(message + where, query_id=self.query_id)

    def parse(self) -> dict[str, Any]:
        if not self.tokens:
            return {"op": "and", "children": []}
        node = self.parse_and()
        if self.peek() is not None:
            raise self.error(f"多余记号 {self.peek().value!r}")
        return node

    def parse_and(self) -> dict[str, Any]:
        children = [self.parse_atom()]
        while self.peek() is not None and self.peek().kind == "AND":
            self.take("AND")
            children.append(self.parse_atom())
        return normalize_and(children)

    def parse_atom(self) -> dict[str, Any]:
        token = self.peek()
        if token is None:
            raise self.error("缺少谓词")
        if token.kind == "LPAREN":
            self.take("LPAREN")
            node = self.parse_and()
            self.take("RPAREN")
            return node
        if token.kind != "IDENT":
            raise self.error("谓词必须以列名开始")
        column = self.take("IDENT").value
        next_token = self.peek()
        if next_token is None:
            raise self.error("列名后缺少操作符")
        if next_token.kind == "IN":
            self.take("IN")
            self.take("LPAREN")
            values = [self.parse_integer()]
            while self.peek() is not None and self.peek().kind == "COMMA":
                self.take("COMMA")
                values.append(self.parse_integer())
            self.take("RPAREN")
            return {
                "op": "in",
                "column": column,
                "values": sorted(set(values)),
            }
        if next_token.kind != "OP":
            raise self.error("列名后缺少比较操作符")
        operator = self.take("OP").value
        value = self.parse_integer()
        operator_map = {
            "=": "eq",
            "==": "eq",
            "!=": "ne",
            ">": "gt",
            ">=": "gte",
            "<": "lt",
            "<=": "lte",
        }
        return {"op": operator_map[operator], "column": column, "value": value}

    def parse_integer(self) -> int:
        token = self.take()
        if token.kind != "NUMBER":
            raise self.error("样本与谓词仅支持 64 位整数")
        return int(token.value)


def normalize_and(children: Iterable[dict[str, Any]]) -> dict[str, Any]:
    flattened: list[dict[str, Any]] = []
    for child in children:
        if child.get("op") == "and":
            flattened.extend(child.get("children", []))
        else:
            flattened.append(child)
    unique: dict[str, dict[str, Any]] = {}
    for child in flattened:
        unique[canonical_json(child)] = child
    ordered = [unique[key] for key in sorted(unique)]
    return {"op": "and", "children": ordered}


def _int_value(value: Any, *, query_id: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _query_error("谓词值必须为整数", query_id=query_id, field=field)
    if value < -(2**63) or value > 2**63 - 1:
        raise _query_error("谓词值超出有符号 64 位整数范围", query_id=query_id, field=field)
    return value


def from_structured(payload: dict[str, Any], *, query_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise _query_error("结构化谓词必须为对象", query_id=query_id)
    op = str(payload.get("op", "")).lower()
    if op == "and":
        children = payload.get("children", payload.get("predicates", []))
        if not isinstance(children, list):
            raise _query_error("AND.children 必须为数组", query_id=query_id)
        return normalize_and(
            from_structured(child, query_id=query_id) for child in children
        )
    column = payload.get("column")
    if not isinstance(column, str) or not column:
        raise _query_error("谓词缺少 column", query_id=query_id)
    if op == "in":
        values = payload.get("values")
        if not isinstance(values, list) or not values:
            raise _query_error("IN 谓词 values 必须为非空数组", query_id=query_id)
        return {
            "op": "in",
            "column": column,
            "values": sorted(
                {
                    _int_value(item, query_id=query_id, field="where.values")
                    for item in values
                }
            ),
        }
    if op == "range":
        low = payload.get("low", payload.get("min"))
        high = payload.get("high", payload.get("max"))
        if low is None and high is None:
            raise _query_error("range 谓词至少需要 low 或 high", query_id=query_id)
        result: dict[str, Any] = {"op": "range", "column": column}
        if low is not None:
            result["low"] = _int_value(low, query_id=query_id, field="where.low")
        if high is not None:
            result["high"] = _int_value(high, query_id=query_id, field="where.high")
        return result
    if op not in {"eq", "ne", "gt", "gte", "lt", "lte"}:
        raise _query_error(f"不支持的谓词操作符 {op!r}", query_id=query_id)
    if "value" not in payload:
        raise _query_error(f"{op} 谓词缺少 value", query_id=query_id)
    return {
        "op": op,
        "column": column,
        "value": _int_value(payload["value"], query_id=query_id, field="where.value"),
    }


def parse_where(
    where: str | dict[str, Any] | None, *, query_id: str
) -> dict[str, Any]:
    if where is None:
        return {"op": "and", "children": []}
    if isinstance(where, str):
        tokens = _lex(where, query_id=query_id)
        return _Parser(tokens, query_id).parse()
    if isinstance(where, dict):
        return from_structured(where, query_id=query_id)
    raise _query_error("where 必须为字符串、对象或 null", query_id=query_id)


def leaf_columns(node: dict[str, Any]) -> list[str]:
    if node.get("op") == "and":
        columns: set[str] = set()
        for child in node.get("children", []):
            columns.update(leaf_columns(child))
        return sorted(columns)
    return [str(node["column"])]


def evaluate_leaf(value: int, leaf: dict[str, Any]) -> bool:
    op = leaf["op"]
    if op == "eq":
        return value == leaf["value"]
    if op == "ne":
        return value != leaf["value"]
    if op == "gt":
        return value > leaf["value"]
    if op == "gte":
        return value >= leaf["value"]
    if op == "lt":
        return value < leaf["value"]
    if op == "lte":
        return value <= leaf["value"]
    if op == "in":
        return value in leaf["values"]
    if op == "range":
        low = leaf.get("low")
        high = leaf.get("high")
        return (low is None or value >= low) and (high is None or value <= high)
    raise ValueError(f"unknown predicate op: {op}")


def evaluate_node(row: dict[str, int], node: dict[str, Any]) -> bool:
    if node.get("op") == "and":
        return all(evaluate_node(row, child) for child in node.get("children", []))
    return evaluate_leaf(row[str(node["column"])], node)


def evaluate_column_mask(values: Iterable[int], node: dict[str, Any], column: str) -> list[bool]:
    leaves = _leaves_for_column(node, column)
    return [all(evaluate_leaf(value, leaf) for leaf in leaves) for value in values]


def _leaves_for_column(node: dict[str, Any], column: str) -> list[dict[str, Any]]:
    if node.get("op") == "and":
        result: list[dict[str, Any]] = []
        for child in node.get("children", []):
            result.extend(_leaves_for_column(child, column))
        return result
    if node.get("column") == column:
        return [node]
    return []
