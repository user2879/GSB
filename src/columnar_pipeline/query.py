from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, sha256_obj
from .errors import data_error
from .model import (
    NUMERIC_COLUMN_TYPES,
    SUPPORTED_AGGREGATES,
    AggregateSpec,
    Query,
    Table,
)


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
COMPARISON_OPS = {
    "=": "eq",
    ">": "gt",
    ">=": "ge",
    "<": "lt",
    "<=": "le",
}
OPERATOR_SYMBOLS = {value: key for key, value in COMPARISON_OPS.items()}


@dataclass(frozen=True)
class Token:
    kind: str
    value: Any
    position: int


def _tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char == "(":
            tokens.append(Token("LPAREN", "(", index))
            index += 1
            continue
        if char == ")":
            tokens.append(Token("RPAREN", ")", index))
            index += 1
            continue
        if char in "><=":
            if char in "><" and index + 1 < length and text[index + 1] == "=":
                tokens.append(Token("OP", char + "=", index))
                index += 2
            else:
                tokens.append(Token("OP", char, index))
                index += 1
            continue
        if char == "'":
            start = index
            index += 1
            pieces: list[str] = []
            closed = False
            while index < length:
                if text[index] == "'":
                    if index + 1 < length and text[index + 1] == "'":
                        pieces.append("'")
                        index += 2
                        continue
                    index += 1
                    closed = True
                    break
                pieces.append(text[index])
                index += 1
            if not closed:
                raise data_error(
                    "unterminated string literal",
                    f"query.where@{start}",
                )
            tokens.append(Token("STRING", "".join(pieces), start))
            continue
        if char.isdigit() or (char == "-" and index + 1 < length and text[index + 1].isdigit()):
            start = index
            index += 1
            while index < length and text[index].isdigit():
                index += 1
            raw = text[start:index]
            try:
                value = int(raw)
            except ValueError as error:
                raise data_error("invalid integer literal", f"query.where@{start}") from error
            tokens.append(Token("NUMBER", value, start))
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (text[index].isalnum() or text[index] == "_"):
                index += 1
            tokens.append(Token("IDENT", text[start:index], start))
            continue
        raise data_error(
            f"unexpected character {char!r} in WHERE expression",
            f"query.where@{index}",
        )
    tokens.append(Token("EOF", None, length))
    return tokens


class _FilterParser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> Token:
        return self.tokens[self.index]

    def take(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def parse(self) -> dict[str, Any]:
        expression = self.parse_expression()
        if self.peek().kind != "EOF":
            token = self.peek()
            raise data_error(
                f"unexpected token {token.value!r} after filter expression",
                f"query.where@{token.position}",
            )
        return expression

    def parse_expression(self) -> dict[str, Any]:
        terms = [self.parse_term()]
        while self.peek().kind == "IDENT" and str(self.peek().value).upper() == "AND":
            self.take()
            terms.append(self.parse_term())
        if len(terms) == 1:
            return terms[0]
        flattened: list[dict[str, Any]] = []
        for term in terms:
            if term.get("op") == "and":
                flattened.extend(term["terms"])
            else:
                flattened.append(term)
        flattened.sort(key=canonical_json)
        return {"op": "and", "terms": flattened}

    def parse_term(self) -> dict[str, Any]:
        token = self.peek()
        if token.kind == "LPAREN":
            self.take()
            expression = self.parse_expression()
            closing = self.peek()
            if closing.kind != "RPAREN":
                raise data_error(
                    "missing closing parenthesis in WHERE expression",
                    f"query.where@{token.position}",
                )
            self.take()
            return expression
        return self.parse_comparison()

    def parse_comparison(self) -> dict[str, Any]:
        left = self.take()
        if left.kind != "IDENT":
            raise data_error(
                "expected a column identifier",
                f"query.where@{left.position}",
            )
        operator = self.take()
        if operator.kind != "OP":
            raise data_error(
                "expected comparison operator (=, >, >=, <, <=)",
                f"query.where@{operator.position}",
            )
        right = self.take()
        if right.kind not in {"STRING", "NUMBER"}:
            raise data_error(
                "expected a string or integer literal",
                f"query.where@{right.position}",
            )
        return {
            "op": COMPARISON_OPS[str(operator.value)],
            "column": str(left.value),
            "value": right.value,
        }


def parse_where(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise data_error("WHERE expression must be a string", "query.where")
    if not text.strip():
        raise data_error("WHERE expression must not be empty", "query.where")
    return _FilterParser(_tokenize(text)).parse()


def _walk_filter(node: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    if node.get("op") == "and":
        for term in node["terms"]:
            yield from _walk_filter(term)
        return
    yield node


def _validate_filter(
    node: Mapping[str, Any],
    tables: tuple[Table, ...],
    table_maps: dict[str, dict[str, Any]],
) -> None:
    for comparison in _walk_filter(node):
        column_name = str(comparison["column"])
        column_types: set[str] = set()
        scales: set[int] = set()
        for table in tables:
            column = table_maps[table.name].get(column_name)
            if column is None:
                raise data_error(
                    f"filter column {column_name!r} does not exist in table {table.name!r}",
                    f"query.where.column.{column_name}",
                    details={"table": table.name},
                )
            column_types.add(column.type_name)
            scales.add(column.scale)
        if len(column_types) != 1 or len(scales) != 1:
            raise data_error(
                f"filter column {column_name!r} has inconsistent types across source tables",
                f"query.where.column.{column_name}",
            )
        column_type = next(iter(column_types))
        literal = comparison["value"]
        if comparison["op"] == "eq":
            compatible = (column_type == "string" and isinstance(literal, str)) or (
                column_type in NUMERIC_COLUMN_TYPES
                and isinstance(literal, int)
                and not isinstance(literal, bool)
            )
            if not compatible:
                raise data_error(
                    f"literal type for filter column {column_name!r} is incompatible with {column_type}",
                    f"query.where.column.{column_name}",
                )
        else:
            if column_type not in NUMERIC_COLUMN_TYPES:
                raise data_error(
                    f"range comparison requires an int or fixed column, got {column_type}",
                    f"query.where.column.{column_name}",
                )
            if not isinstance(literal, int) or isinstance(literal, bool):
                raise data_error(
                    "range comparison requires an integer literal",
                    f"query.where.column.{column_name}",
                )


def _as_string_list(value: Any, location: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value:
        raise data_error("expected a non-empty string or string list", location)
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise data_error("expected a non-empty string", f"{location}[{index}]")
        result.append(item)
    return result


def parse_query(query_doc: Mapping[str, Any], tables: tuple[Table, ...]) -> Query:
    if not isinstance(query_doc, Mapping):
        raise data_error("query must be a JSON object", "query")
    table_maps = {table.name: table.column_map() for table in tables}
    if "from" not in query_doc:
        raise data_error("query.from is required", "query.from")
    raw_from = _as_string_list(query_doc["from"], "query.from")
    if len(set(raw_from)) != len(raw_from):
        raise data_error("query.from contains duplicate table names", "query.from")
    known_tables = {table.name for table in tables}
    for index, name in enumerate(raw_from):
        if name not in known_tables:
            raise data_error(
                f"query source table {name!r} does not exist",
                f"query.from[{index}]",
            )
    from_tables = tuple(sorted(raw_from))
    source_tables = tuple(table for table in tables if table.name in from_tables)

    raw_group_by = _as_string_list(query_doc.get("group_by"), "query.group_by")
    if len(set(raw_group_by)) != len(raw_group_by):
        raise data_error("query.group_by contains duplicate columns", "query.group_by")
    for index, column_name in enumerate(raw_group_by):
        for table in source_tables:
            if column_name not in table_maps[table.name]:
                raise data_error(
                    f"group column {column_name!r} does not exist in table {table.name!r}",
                    f"query.group_by[{index}]",
                    details={"table": table.name},
                )
        types = {table_maps[table.name][column_name].type_name for table in source_tables}
        if len(types) > 1:
            raise data_error(
                f"group column {column_name!r} has inconsistent types across source tables",
                f"query.group_by[{index}]",
            )

    raw_aggregates = query_doc.get("aggregates")
    if not isinstance(raw_aggregates, list) or not raw_aggregates:
        raise data_error("query.aggregates must be a non-empty list", "query.aggregates")
    aggregates: list[AggregateSpec] = []
    used_names: set[str] = set()
    for index, raw in enumerate(raw_aggregates):
        location = f"query.aggregates[{index}]"
        if not isinstance(raw, Mapping):
            raise data_error("aggregate must be a JSON object", location)
        name = raw.get("name")
        op = raw.get("op")
        column_name = raw.get("column")
        if not isinstance(name, str) or not IDENTIFIER_RE.match(name):
            raise data_error("aggregate name must be a valid identifier", f"{location}.name")
        if name in used_names:
            raise data_error(f"duplicate aggregate name {name!r}", f"{location}.name")
        used_names.add(name)
        if not isinstance(op, str) or op.lower() not in SUPPORTED_AGGREGATES:
            raise data_error(
                "aggregate op must be one of count, sum, avg",
                f"{location}.op",
            )
        op = op.lower()
        if not isinstance(column_name, str) or not column_name:
            raise data_error("aggregate column must be a non-empty string", f"{location}.column")
        if column_name == "*" and op != "count":
            raise data_error("only count supports column '*'", f"{location}.column")
        scale = 0
        if column_name != "*":
            scales: set[int] = set()
            types: set[str] = set()
            for table in source_tables:
                column = table_maps[table.name].get(column_name)
                if column is None:
                    raise data_error(
                        f"aggregate column {column_name!r} does not exist in table {table.name!r}",
                        f"{location}.column",
                        details={"table": table.name},
                    )
                scales.add(column.scale)
                types.add(column.type_name)
            if len(scales) != 1 or len(types) != 1:
                raise data_error(
                    f"aggregate column {column_name!r} has inconsistent types across source tables",
                    f"{location}.column",
                )
            type_name = next(iter(types))
            if op in {"sum", "avg"}:
                scale = next(iter(scales))
            if op in {"sum", "avg"} and type_name not in NUMERIC_COLUMN_TYPES:
                raise data_error(
                    f"aggregate {op} requires an int or fixed column, got {type_name}",
                    f"{location}.column",
                )
        aggregates.append(AggregateSpec(name=name, op=op, column=column_name, scale=scale))
    aggregates.sort(key=lambda aggregate: aggregate.name)

    where_text = query_doc.get("where")
    where_ast = None if where_text is None else parse_where(where_text)
    if where_ast is not None:
        _validate_filter(where_ast, source_tables, table_maps)

    canonical = {
        "from": list(from_tables),
        "where": where_ast,
        "group_by": list(raw_group_by),
        "aggregates": [
            {"name": aggregate.name, "op": aggregate.op, "column": aggregate.column}
            for aggregate in aggregates
        ],
    }
    query_id = sha256_obj(canonical)
    return Query(
        from_tables=from_tables,
        group_by=tuple(raw_group_by),
        aggregates=tuple(aggregates),
        where=where_ast,
        where_text=where_text,
        query_id=query_id,
    )


def comparison_to_text(node: Mapping[str, Any]) -> str:
    if node.get("op") == "and":
        return "(" + " AND ".join(comparison_to_text(term) for term in node["terms"]) + ")"
    value = node["value"]
    literal = "'" + str(value).replace("'", "''") + "'" if isinstance(value, str) else str(value)
    return f"{node['column']} {OPERATOR_SYMBOLS[str(node['op'])]} {literal}"
