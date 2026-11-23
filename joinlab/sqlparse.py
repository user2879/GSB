from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .errors import schema_error
from .model import (
    ColumnRef,
    FilterPredicate,
    JoinPredicate,
    Query,
    SelectItem,
    Table,
)


_TOKEN_RE = re.compile(
    r"(?P<WS>\s+)"
    r"|(?P<NUMBER>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<OP><=|>=|<>|!=|=|<|>)"
    r"|(?P<PUNCT>[(),.*;])"
    r"|(?P<MINUS>-)"
)

_AGGREGATES = {"SUM", "COUNT", "AVG", "MIN", "MAX"}
_CLAUSE_WORDS = {"FROM", "WHERE", "GROUP", "ORDER", "INNER", "JOIN", "ON"}


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int


class QueryParser:
    def __init__(self, sql: str) -> None:
        self.sql = sql
        self.tokens = self._tokenize(sql)
        self.index = 0

    def _tokenize(self, sql: str) -> list[Token]:
        tokens: list[Token] = []
        cursor = 0
        while cursor < len(sql):
            match = _TOKEN_RE.match(sql, cursor)
            if not match:
                raise schema_error(
                    f"查询在字符 {cursor} 处含非法字符 {sql[cursor]!r}",
                    field="query.sql",
                    position=cursor,
                )
            kind = match.lastgroup or ""
            if kind != "WS":
                value = match.group()
                tokens.append(Token(kind, value, cursor))
            cursor = match.end()
        tokens.append(Token("EOF", "<EOF>", len(sql)))
        return tokens

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.index + offset, len(self.tokens) - 1)]

    def consume(self) -> Token:
        token = self.peek()
        self.index += 1
        return token

    def accept_keyword(self, word: str) -> bool:
        token = self.peek()
        if token.kind == "IDENT" and token.value.upper() == word:
            self.index += 1
            return True
        return False

    def expect_keyword(self, word: str) -> None:
        if not self.accept_keyword(word):
            token = self.peek()
            raise schema_error(
                f"查询期望关键字 {word}，实际为 {token.value!r}",
                field="query.sql",
                position=token.position,
            )

    def accept(self, value: str) -> bool:
        token = self.peek()
        if token.kind in {"PUNCT", "OP", "MINUS"} and token.value == value:
            self.index += 1
            return True
        return False

    def expect(self, value: str) -> Token:
        token = self.peek()
        if token.kind in {"PUNCT", "OP", "MINUS"} and token.value == value:
            self.index += 1
            return token
        raise schema_error(
            f"查询期望 {value!r}，实际为 {token.value!r}",
            field="query.sql",
            position=token.position,
        )

    def parse_identifier(self, purpose: str) -> str:
        token = self.peek()
        if token.kind != "IDENT":
            raise schema_error(
                f"查询期望{purpose}标识符，实际为 {token.value!r}",
                field="query.sql",
                position=token.position,
            )
        self.index += 1
        return token.value

    def parse_column_ref(self) -> ColumnRef:
        first = self.parse_identifier("列")
        self.expect(".")
        second = self.parse_identifier("列")
        return ColumnRef(first, second)

    def parse_literal(self) -> int | float | None:
        sign = 1
        if self.accept("-"):
            sign = -1
        if self.accept_keyword("NULL"):
            if sign < 0:
                raise schema_error("NULL 前不能有负号", field="query.sql")
            return None
        token = self.peek()
        if token.kind != "NUMBER":
            raise schema_error(
                f"查询期望数值字面量，实际为 {token.value!r}",
                field="query.sql",
                position=token.position,
            )
        self.index += 1
        raw = token.value
        if "." in raw or "e" in raw.lower():
            value: int | float = sign * float(raw)
        else:
            value = sign * int(raw)
        return value

    def parse_select_items(self) -> list[SelectItem]:
        items: list[SelectItem] = []
        while True:
            token = self.peek()
            aggregate = None
            column = None
            count_star = False
            if token.kind == "IDENT" and token.value.upper() in _AGGREGATES:
                aggregate = token.value.upper()
                self.consume()
                self.expect("(")
                if self.accept("*"):
                    if aggregate != "COUNT":
                        raise schema_error(
                            f"只有 COUNT 支持 *，实际为 {aggregate}", field="query.select"
                        )
                    count_star = True
                else:
                    column = self.parse_column_ref()
                self.expect(")")
                default_alias = (
                    f"{aggregate.lower()}_star"
                    if count_star
                    else f"{aggregate.lower()}_{column.column}"  # type: ignore[union-attr]
                )
            else:
                column = self.parse_column_ref()
                default_alias = column.column

            alias = default_alias
            if self.accept_keyword("AS"):
                alias = self.parse_identifier("别名")
            items.append(
                SelectItem(
                    column=column,
                    aggregate=aggregate,
                    alias=alias,
                    count_star=count_star,
                )
            )
            if not self.accept(","):
                break
        return items

    def _starts_predicate_atom(self) -> bool:
        token = self.peek()
        return token.kind in {"IDENT", "NUMBER", "MINUS"} or token.value == "("

    def parse_boolean_expression(self) -> list[FilterPredicate]:
        predicates: list[FilterPredicate] = []

        def parse_factor() -> None:
            if self.accept("("):
                parse_inner()
                self.expect(")")
                return
            predicates.append(self.parse_comparison())

        def parse_inner() -> None:
            parse_factor()
            while self.accept_keyword("AND"):
                parse_factor()

        parse_inner()
        return predicates

    def parse_comparison(self) -> FilterPredicate:
        token = self.peek()
        if token.kind == "IDENT":
            column = self.parse_column_ref()
            if self.accept_keyword("IS"):
                negate = self.accept_keyword("NOT")
                self.expect_keyword("NULL")
                return FilterPredicate(column, "IS NOT NULL" if negate else "IS NULL", None, True)
            op_token = self.peek()
            if op_token.kind != "OP":
                raise schema_error(
                    f"列 {column.key} 后缺少比较运算符",
                    table=column.table,
                    column=column.column,
                    field="query.predicate",
                )
            self.consume()
            value = self.parse_literal()
            return FilterPredicate(column, op_token.value, value)

        if token.kind in {"NUMBER", "MINUS"}:
            value = self.parse_literal()
            op_token = self.peek()
            if op_token.kind != "OP":
                raise schema_error("数值字面量后缺少比较运算符", field="query.predicate")
            self.consume()
            column = self.parse_column_ref()
            flipped = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(op_token.value, op_token.value)
            return FilterPredicate(column, flipped, value)

        raise schema_error(
            f"无法解析谓词，遇到 {token.value!r}", field="query.predicate", position=token.position
        )

    def parse_join_refs(self) -> JoinPredicate:
        left = self.parse_column_ref()
        op_token = self.peek()
        if op_token.kind != "OP":
            raise schema_error("连接谓词缺少比较运算符", field="query.joins")
        self.consume()
        right = self.parse_column_ref()
        if op_token.value != "=":
            raise schema_error(
                "连接图枚举仅接受等值连接谓词",
                table=left.table,
                column=left.column,
                field="query.joins.op",
            )
        return JoinPredicate(left, right, "=")

    def parse_join_expression(self) -> list[JoinPredicate]:
        predicates: list[JoinPredicate] = []

        def parse_factor() -> None:
            if self.accept("("):
                parse_inner()
                self.expect(")")
                return
            predicates.append(self.parse_join_refs())

        def parse_inner() -> None:
            parse_factor()
            while self.accept_keyword("AND"):
                parse_factor()

        parse_inner()
        return predicates

    def parse(self) -> Query:
        self.expect_keyword("SELECT")
        select = self.parse_select_items()
        self.expect_keyword("FROM")

        source_order: list[str] = []
        joins: list[JoinPredicate] = []
        first_table = self.parse_identifier("表")
        source_order.append(first_table)
        while True:
            if self.accept_keyword("INNER"):
                self.expect_keyword("JOIN")
            elif self.accept_keyword("JOIN"):
                pass
            else:
                break
            table = self.parse_identifier("表")
            source_order.append(table)
            self.expect_keyword("ON")
            joins.extend(self.parse_join_expression())

        filters: list[FilterPredicate] = []
        if self.accept_keyword("WHERE"):
            filters = self.parse_boolean_expression()

        group_by: list[ColumnRef] = []
        if self.accept_keyword("GROUP"):
            self.expect_keyword("BY")
            while True:
                group_by.append(self.parse_column_ref())
                if not self.accept(","):
                    break

        order_by: list[ColumnRef] = []
        if self.accept_keyword("ORDER"):
            self.expect_keyword("BY")
            while True:
                order_by.append(self.parse_column_ref())
                if self.accept_keyword("ASC") or self.accept_keyword("DESC"):
                    pass
                if not self.accept(","):
                    break

        self.accept(";")
        if self.peek().kind != "EOF":
            token = self.peek()
            raise schema_error(
                f"查询末尾存在未解析内容 {token.value!r}",
                field="query.sql",
                position=token.position,
            )

        unique_sources = tuple(dict.fromkeys(source_order))
        if len(unique_sources) != len(source_order):
            raise schema_error("FROM/JOIN 中存在重复表名", field="query.tables")
        joins = _dedupe_sorted(joins, key=lambda item: item.key)
        filters = _dedupe_sorted(filters, key=lambda item: item.key)
        return Query(
            sql=self.sql,
            source_order=unique_sources,
            tables=tuple(sorted(unique_sources)),
            joins=tuple(joins),
            filters=tuple(filters),
            select=tuple(select),
            group_by=tuple(group_by),
            order_by=tuple(order_by),
        )


def _dedupe_sorted(items: Iterable, *, key) -> list:
    result = []
    seen = set()
    for item in sorted(items, key=key):
        marker = key(item)
        if marker not in seen:
            seen.add(marker)
            result.append(item)
    return result


def parse_query(sql: str) -> Query:
    return QueryParser(sql).parse()
