"""A deliberately small, deterministic SQL parser.

Supported grammar (case-insensitive, arbitrary whitespace and redundant
parentheses):

    SELECT select_item [, ...]
    FROM table
    [WHERE predicate]
    [GROUP BY column [, ...]]
    [ORDER BY output_name [ASC|DESC] [, ...]]

select_item is a source column or COUNT/SUM/MIN/MAX over a source column
(COUNT(*) is also accepted), with an optional AS alias. Predicates support
comparisons, IN, AND, parentheses and NULL-aware three-valued semantics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_bytes
from .errors import syntax_error


AGGREGATES = frozenset({"count", "sum", "min", "max"})
COMPARISONS = frozenset({"=", "!=", "<>", "<", "<=", ">", ">="})


@dataclass(frozen=True)
class Token:
    kind: str
    value: Any
    start: int
    end: int


@dataclass(frozen=True)
class Condition:
    column: str
    op: str
    values: tuple[Any, ...]
    pos: int

    def canonical(self) -> dict[str, Any]:
        if self.op == "in":
            normalized = sorted({normalize_literal(value) for value in self.values}, key=_literal_key)
            return {"column": self.column, "op": "in", "values": normalized}
        return {"column": self.column, "op": self.op, "value": normalize_literal(self.values[0])}


@dataclass(frozen=True)
class SelectItem:
    kind: str  # column | aggregate | star
    alias: str
    column: str | None
    function: str | None
    pos: int

    def canonical(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "alias": self.alias}
        if self.column is not None:
            result["column"] = self.column
        if self.function is not None:
            result["function"] = self.function
        return result


@dataclass(frozen=True)
class OrderItem:
    name: str
    direction: str
    pos: int

    def canonical(self) -> dict[str, str]:
        return {"name": self.name, "direction": self.direction}


@dataclass(frozen=True)
class ParsedQuery:
    table: str
    table_pos: int
    select: tuple[SelectItem, ...]
    where: tuple[Condition, ...]
    group_by: tuple[str, ...]
    group_positions: tuple[int, ...]
    order_by: tuple[OrderItem, ...]
    sql: str

    def canonical(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "select": [item.canonical() for item in self.select],
            "where": [item.canonical() for item in self.where],
            "group_by": sorted(self.group_by),
            "order_by": [item.canonical() for item in self.order_by],
        }

    @property
    def has_aggregate(self) -> bool:
        return any(item.kind == "aggregate" for item in self.select)

    @property
    def is_aggregate_query(self) -> bool:
        return self.has_aggregate or bool(self.group_by)


_NUMBER_RE = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize_literal(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite SQL literal")
        if value == 0:
            return 0
        if value.is_integer() and abs(value) <= 9_007_199_254_740_992:
            return int(value)
        return value
    if isinstance(value, str):
        return value
    if value is None:
        return None
    raise ValueError(f"unsupported literal type: {type(value)!r}")


def _literal_key(value: Any) -> bytes:
    return canonical_bytes(normalize_literal(value))


def tokenize(sql: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        if char.isspace():
            index += 1
            continue
        if sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if char in "(),;*":
            tokens.append(Token("PUNCT", char, index, index + 1))
            index += 1
            continue
        if char in "<>!=":
            two = sql[index : index + 2]
            if two in {"<=", ">=", "<>", "!="}:
                tokens.append(Token("OP", "<>" if two == "!=" else two, index, index + 2))
                index += 2
                continue
            if char in "<>=":
                tokens.append(Token("OP", char, index, index + 1))
                index += 1
                continue
            raise syntax_error("unexpected '!' (use != for inequality)", index)
        if char == "-":
            tokens.append(Token("OP", "-", index, index + 1))
            index += 1
            continue
        if char.isdigit() or (char == "." and index + 1 < length and sql[index + 1].isdigit()):
            match = _NUMBER_RE.match(sql, index)
            if match is None:
                raise syntax_error("invalid numeric literal", index)
            text = match.group(0)
            value: int | float
            if "." in text or "e" in text.lower():
                value = float(text)
            else:
                value = int(text)
            tokens.append(Token("NUMBER", value, index, match.end()))
            index = match.end()
            continue
        if char == "'":
            start = index
            index += 1
            pieces: list[str] = []
            while index < length:
                current = sql[index]
                if current == "'":
                    if index + 1 < length and sql[index + 1] == "'":
                        pieces.append("'")
                        index += 2
                        continue
                    index += 1
                    break
                pieces.append(current)
                index += 1
            else:
                raise syntax_error("unterminated string literal", start)
            tokens.append(Token("STRING", "".join(pieces), start, index))
            continue
        match = _WORD_RE.match(sql, index)
        if match is not None:
            tokens.append(Token("WORD", match.group(0), index, match.end()))
            index = match.end()
            continue
        raise syntax_error(f"unexpected character {char!r}", index)
    tokens.append(Token("EOF", None, length, length))
    return tokens


class Parser:
    def __init__(self, sql: str) -> None:
        self.sql = sql
        self.tokens = tokenize(sql)
        self.index = 0

    def peek(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def expect_punct(self, value: str) -> Token:
        token = self.peek()
        if token.kind == "PUNCT" and token.value == value:
            return self.advance()
        raise syntax_error(f"expected {value!r}", token.start)

    def accept_punct(self, value: str) -> Token | None:
        token = self.peek()
        if token.kind == "PUNCT" and token.value == value:
            return self.advance()
        return None

    def peek_word(self) -> str | None:
        token = self.peek()
        if token.kind == "WORD":
            return str(token.value).upper()
        return None

    def accept_word(self, value: str) -> Token | None:
        token = self.peek()
        if token.kind == "WORD" and str(token.value).upper() == value.upper():
            return self.advance()
        return None

    def expect_word(self, value: str) -> Token:
        token = self.peek()
        if token.kind == "WORD" and str(token.value).upper() == value.upper():
            return self.advance()
        raise syntax_error(f"expected keyword {value}", token.start)

    def expect_identifier(self, label: str = "identifier") -> Token:
        token = self.peek()
        if token.kind != "WORD":
            raise syntax_error(f"expected {label}", token.start)
        return self.advance()

    def parse(self) -> ParsedQuery:
        if self.peek().kind == "EOF":
            raise syntax_error("empty SQL text", 0)
        self.expect_word("SELECT")
        select_items = self.parse_select_list()
        self.expect_word("FROM")
        table_token = self.expect_identifier("table name")
        where: tuple[Condition, ...] = ()
        group_by: tuple[str, ...] = ()
        group_positions: tuple[int, ...] = ()
        order_by: tuple[OrderItem, ...] = ()

        if self.accept_word("WHERE") is not None:
            conditions = self.parse_and_conditions()
            where = tuple(sorted(conditions, key=lambda item: canonical_bytes(item.canonical())))
            if self.peek_word() == "OR":
                raise syntax_error("OR is not supported; combine predicates with AND", self.peek().start)

        if self.accept_word("GROUP") is not None:
            self.expect_word("BY")
            groups: list[str] = []
            positions: list[int] = []
            while True:
                token = self.expect_identifier("GROUP BY column")
                groups.append(str(token.value).lower())
                positions.append(token.start)
                if self.accept_punct(",") is None:
                    break
            group_by = tuple(groups)
            group_positions = tuple(positions)

        if self.accept_word("ORDER") is not None:
            self.expect_word("BY")
            order_items: list[OrderItem] = []
            while True:
                token = self.expect_identifier("ORDER BY column or alias")
                direction = "ASC"
                if self.accept_word("ASC") is not None:
                    direction = "ASC"
                elif self.accept_word("DESC") is not None:
                    direction = "DESC"
                order_items.append(OrderItem(str(token.value).lower(), direction, token.start))
                if self.accept_punct(",") is None:
                    break
            order_by = tuple(order_items)

        self.accept_punct(";")
        tail = self.peek()
        if tail.kind != "EOF":
            raise syntax_error(f"unexpected token {tail.value!r}", tail.start)

        return ParsedQuery(
            table=str(table_token.value).lower(),
            table_pos=table_token.start,
            select=tuple(select_items),
            where=where,
            group_by=group_by,
            group_positions=group_positions,
            order_by=order_by,
            sql=self.sql,
        )

    def parse_select_list(self) -> list[SelectItem]:
        items: list[SelectItem] = []
        while True:
            token = self.peek()
            if token.kind == "PUNCT" and token.value == "*":
                self.advance()
                items.append(SelectItem("star", "*", None, None, token.start))
            else:
                identifier = self.expect_identifier("selected column or aggregate")
                name = str(identifier.value).lower()
                following = self.peek()
                if (
                    name in AGGREGATES
                    and following.kind == "PUNCT"
                    and following.value == "("
                ):
                    self.advance()
                    argument = self.peek()
                    if argument.kind == "PUNCT" and argument.value == "*":
                        if name != "count":
                            raise syntax_error("only COUNT accepts *", argument.start)
                        self.advance()
                        column: str | None = "*"
                    else:
                        col_token = self.expect_identifier("aggregate column")
                        column = str(col_token.value).lower()
                    self.expect_punct(")")
                    alias = self.parse_optional_alias()
                    if alias is None:
                        alias = f"{name}_{'star' if column == '*' else column}"
                    items.append(SelectItem("aggregate", alias, column, name, identifier.start))
                else:
                    alias = self.parse_optional_alias()
                    items.append(SelectItem("column", alias or name, name, None, identifier.start))
            if self.accept_punct(",") is None:
                break
        return items

    def parse_optional_alias(self) -> str | None:
        if self.accept_word("AS") is None:
            return None
        token = self.expect_identifier("alias")
        return str(token.value).lower()

    def parse_and_conditions(self) -> list[Condition]:
        conditions = self.parse_predicate_primary()
        while self.accept_word("AND") is not None:
            conditions.extend(self.parse_predicate_primary())
        if not conditions:
            raise syntax_error("expected predicate", self.peek().start)
        return conditions

    def parse_predicate_primary(self) -> list[Condition]:
        if self.accept_punct("(") is not None:
            conditions = self.parse_and_conditions()
            self.expect_punct(")")
            return conditions

        column_token = self.expect_identifier("predicate column")
        column = str(column_token.value).lower()
        pos = column_token.start
        if self.accept_word("IN") is not None:
            self.expect_punct("(")
            values: list[Any] = []
            while True:
                values.append(self.parse_literal())
                if self.accept_punct(",") is None:
                    break
            self.expect_punct(")")
            return [Condition(column, "in", tuple(values), pos)]

        token = self.peek()
        if token.kind != "OP" or token.value not in COMPARISONS:
            raise syntax_error("expected comparison operator or IN", token.start)
        op = str(self.advance().value)
        value = self.parse_literal()
        return [Condition(column, "=" if op == "==" else op, (value,), pos)]

    def parse_literal(self) -> Any:
        token = self.peek()
        if token.kind == "OP" and token.value == "-":
            self.advance()
            number = self.peek()
            if number.kind != "NUMBER":
                raise syntax_error("expected a number after '-'", number.start)
            self.advance()
            return normalize_literal(-number.value)
        if token.kind == "NUMBER":
            self.advance()
            return normalize_literal(token.value)
        if token.kind == "STRING":
            self.advance()
            return token.value
        raise syntax_error("expected a numeric or string literal", token.start)


def parse_sql(sql: str) -> ParsedQuery:
    if not isinstance(sql, str):
        raise syntax_error("SQL text must be a string", 0)
    return Parser(sql).parse()


def condition_to_text(condition: dict[str, Any]) -> str:
    if condition["op"] == "in":
        values = ", ".join(repr(value) for value in condition["values"])
        return f"{condition['column']} IN ({values})"
    return f"{condition['column']} {condition['op']} {condition['value']!r}"


def all_conditions(where: Iterable[Condition]) -> list[Condition]:
    return sorted(where, key=lambda item: canonical_bytes(item.canonical()))
