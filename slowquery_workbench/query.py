"""Parser and canonicalizer for the deliberately small SELECT dialect.

Supported syntax is intentionally bounded: one or two inner-joined tables,
simple projections, conjunctive comparisons/IN/range predicates, optional
ORDER BY columns and LIMIT. Anything outside that surface fails with E_PARSE.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence, Union

from .errors import WorkbenchError, fail


WORD = "WORD"
STRING = "STRING"
NUMBER = "NUMBER"
OP = "OP"
LPAREN = "LPAREN"
RPAREN = "RPAREN"
COMMA = "COMMA"
DOT = "DOT"
STAR = "STAR"
SEMI = "SEMI"


RESERVED_WORDS = {
    "select",
    "from",
    "where",
    "join",
    "inner",
    "on",
    "and",
    "or",
    "as",
    "in",
    "between",
    "order",
    "by",
    "limit",
    "asc",
    "desc",
    "group",
    "having",
    "distinct",
    "union",
    "intersect",
    "except",
    "offset",
}

FORBIDDEN_WORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "merge",
    "pragma",
    "attach",
    "detach",
    "vacuum",
    "reindex",
    "with",
    "recursive",
    "group",
    "having",
    "distinct",
    "union",
    "intersect",
    "except",
    "exists",
    "or",
}

AGGREGATE_WORDS = {
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "total",
    "group_concat",
    "json_group_array",
    "json_group_object",
}


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int

    @property
    def lower(self) -> str:
        return self.text.lower()


def _parse_error(message: str, position: int | None = None) -> WorkbenchError:
    details = {"position": position} if position is not None else None
    return fail("E_PARSE", message, details)


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
        if sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            if end < 0:
                raise _parse_error("unterminated block comment", index)
            index = end + 2
            continue
        if char == "'":
            start = index
            index += 1
            value: list[str] = []
            closed = False
            while index < length:
                if sql[index] == "'":
                    if index + 1 < length and sql[index + 1] == "'":
                        value.append("'")
                        index += 2
                        continue
                    index += 1
                    closed = True
                    break
                value.append(sql[index])
                index += 1
            if not closed:
                raise _parse_error("unterminated string literal", start)
            tokens.append(Token(STRING, "".join(value), start))
            continue
        if char.isdigit():
            start = index
            while index < length and sql[index].isdigit():
                index += 1
            if index < length and sql[index] == ".":
                index += 1
                while index < length and sql[index].isdigit():
                    index += 1
            tokens.append(Token(NUMBER, sql[start:index], start))
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (sql[index].isalnum() or sql[index] in "_$"):
                index += 1
            tokens.append(Token(WORD, sql[start:index], start))
            continue
        if char in "<>!=":
            start = index
            index += 1
            if index < length and sql[start:index + 1] in {"<=", ">=", "<>", "!="}:
                index += 1
            op = sql[start:index]
            if op == "!":
                raise _parse_error("invalid operator", start)
            tokens.append(Token(OP, "<>" if op == "!=" else op, start))
            continue
        if char in "=+-":
            tokens.append(Token(OP, char, index))
            index += 1
            continue
        if char == "(":
            tokens.append(Token(LPAREN, char, index))
            index += 1
            continue
        if char == ")":
            tokens.append(Token(RPAREN, char, index))
            index += 1
            continue
        if char == ",":
            tokens.append(Token(COMMA, char, index))
            index += 1
            continue
        if char == ".":
            tokens.append(Token(DOT, char, index))
            index += 1
            continue
        if char == "*":
            tokens.append(Token(STAR, char, index))
            index += 1
            continue
        if char == ";":
            tokens.append(Token(SEMI, char, index))
            index += 1
            continue
        raise _parse_error(f"unsupported character {char!r}", index)
    return tokens


@dataclass(frozen=True)
class ColumnRef:
    table: str | None
    name: str

    def sql(self) -> str:
        return f"{self.table}.{self.name}" if self.table else self.name


@dataclass(frozen=True)
class Literal:
    value: str | Decimal
    is_string: bool

    def sql(self) -> str:
        if self.is_string:
            escaped = str(self.value).replace("'", "''")
            return f"'{escaped}'"
        return format(self.value, "f")

    def python_value(self) -> str | Decimal:
        return self.value


Operand = Union[ColumnRef, Literal]


@dataclass(frozen=True)
class Projection:
    kind: str  # column | star | table_star
    column: ColumnRef | None = None
    table: str | None = None

    def sql(self) -> str:
        if self.kind == "star":
            return "*"
        if self.kind == "table_star":
            return f"{self.table}.*"
        assert self.column is not None
        return self.column.sql()


@dataclass(frozen=True)
class TableRef:
    name: str
    alias: str | None = None

    @property
    def scope(self) -> str:
        return self.alias or self.name

    def sql(self, index_hint: str | None = None) -> str:
        text = self.name
        if self.alias:
            text += f" AS {self.alias}"
        if index_hint:
            quoted = '"' + index_hint.replace('"', '""') + '"'
            text += f" INDEXED BY {quoted}"
        return text


@dataclass(frozen=True)
class Predicate:
    kind: str  # compare | in | between
    column: ColumnRef
    op: str | None = None
    right: Operand | None = None
    values: tuple[Literal, ...] = ()
    low: Literal | None = None
    high: Literal | None = None

    def sql(self) -> str:
        if self.kind == "compare":
            assert self.op is not None and self.right is not None
            return f"{self.column.sql()} {self.op} {operand_sql(self.right)}"
        if self.kind == "in":
            rendered = ", ".join(item.sql() for item in self.values)
            return f"{self.column.sql()} IN ({rendered})"
        assert self.low is not None and self.high is not None
        return f"{self.column.sql()} BETWEEN {self.low.sql()} AND {self.high.sql()}"


@dataclass(frozen=True)
class OrderTerm:
    column: ColumnRef
    descending: bool = False

    def sql(self) -> str:
        return f"{self.column.sql()} {'DESC' if self.descending else 'ASC'}"


@dataclass(frozen=True)
class Query:
    projection: tuple[Projection, ...]
    tables: tuple[TableRef, ...]
    join_conditions: tuple[Predicate, ...] = ()
    predicates: tuple[Predicate, ...] = ()
    order_by: tuple[OrderTerm, ...] = ()
    limit: int | None = None

    def to_sql(self, index_hints: dict[str, str] | None = None) -> str:
        hints = index_hints or {}
        projection = ", ".join(item.sql() for item in self.projection)
        from_sql = self.tables[0].sql(hints.get(self.tables[0].scope))
        if len(self.tables) > 1:
            from_sql += " JOIN " + self.tables[1].sql(hints.get(self.tables[1].scope))
        text = f"SELECT {projection} FROM {from_sql}"
        if self.join_conditions:
            text += " ON " + " AND ".join(item.sql() for item in self.join_conditions)
        if self.predicates:
            text += " WHERE " + " AND ".join(item.sql() for item in self.predicates)
        if self.order_by:
            text += " ORDER BY " + ", ".join(item.sql() for item in self.order_by)
        if self.limit is not None:
            text += f" LIMIT {self.limit}"
        return text

    @property
    def canonical_sql(self) -> str:
        return self.to_sql()

    @property
    def query_id(self) -> str:
        digest = hashlib.sha256(self.canonical_sql.encode("utf-8")).hexdigest()
        return "q_" + digest

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(sorted({table.name for table in self.tables}))

    def table_by_scope(self, scope: str) -> TableRef | None:
        lowered = scope.lower()
        for table in self.tables:
            if table.scope == lowered or table.name == lowered:
                return table
        return None

    def column_ref(self, table: str | None, name: str) -> ColumnRef:
        if table is not None:
            matched = self.table_by_scope(table)
            if matched is not None:
                return ColumnRef(matched.scope, name)
        return ColumnRef(table, name)


def operand_sql(operand: Operand) -> str:
    return operand.sql()


def _is_word(token: Token, word: str) -> bool:
    return token.kind == WORD and token.lower == word


def _find_top_level(
    tokens: Sequence[Token], word: str, start: int = 0, end: int | None = None
) -> int | None:
    depth = 0
    stop = len(tokens) if end is None else end
    for index in range(start, stop):
        token = tokens[index]
        if token.kind == LPAREN:
            depth += 1
        elif token.kind == RPAREN:
            depth -= 1
            if depth < 0:
                raise _parse_error("unmatched closing parenthesis", token.pos)
        elif depth == 0 and _is_word(token, word):
            return index
    return None


def _split_top_level(tokens: Sequence[Token], separator_kind: str) -> list[list[Token]]:
    parts: list[list[Token]] = []
    current: list[Token] = []
    depth = 0
    for token in tokens:
        if token.kind == LPAREN:
            depth += 1
        elif token.kind == RPAREN:
            depth -= 1
            if depth < 0:
                raise _parse_error("unmatched closing parenthesis", token.pos)
        if token.kind == separator_kind and depth == 0:
            parts.append(current)
            current = []
        else:
            current.append(token)
    if depth != 0:
        raise _parse_error("unmatched parenthesis")
    parts.append(current)
    return parts


def _split_and(tokens: Sequence[Token]) -> list[list[Token]]:
    parts: list[list[Token]] = []
    current: list[Token] = []
    depth = 0
    for token in tokens:
        if token.kind == LPAREN:
            depth += 1
        elif token.kind == RPAREN:
            depth -= 1
            if depth < 0:
                raise _parse_error("unmatched closing parenthesis", token.pos)
        if depth == 0 and _is_word(token, "and"):
            parts.append(current)
            current = []
        else:
            current.append(token)
    if depth != 0:
        raise _parse_error("unmatched parenthesis")
    parts.append(current)

    merged: list[list[Token]] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        has_between = any(_is_word(token, "between") for token in part)
        if has_between and index + 1 < len(parts):
            part = part + [Token(WORD, "AND", -1)] + parts[index + 1]
            index += 2
        else:
            index += 1
        merged.append(part)
    return merged


def _parse_column(tokens: Sequence[Token]) -> ColumnRef:
    if len(tokens) == 1 and tokens[0].kind == WORD:
        word = tokens[0].lower
        if word in RESERVED_WORDS:
            raise _parse_error(f"expected column name, got keyword {word}", tokens[0].pos)
        return ColumnRef(None, word)
    if len(tokens) == 3 and tokens[0].kind == WORD and tokens[1].kind == DOT and tokens[2].kind == WORD:
        table, name = tokens[0].lower, tokens[2].lower
        if table in RESERVED_WORDS or name in RESERVED_WORDS:
            raise _parse_error("invalid qualified column", tokens[0].pos)
        return ColumnRef(table, name)
    position = tokens[0].pos if tokens else None
    raise _parse_error("expected a simple column reference", position)


def _parse_literal(tokens: Sequence[Token]) -> Literal:
    sign = ""
    body = list(tokens)
    if len(body) == 2 and body[0].kind == OP and body[0].text in {"+", "-"} and body[1].kind == NUMBER:
        sign = body[0].text
        body = body[1:]
    if len(body) != 1:
        position = tokens[0].pos if tokens else None
        raise _parse_error("expected a scalar literal", position)
    token = body[0]
    if token.kind == STRING:
        return Literal(token.text, True)
    if token.kind == NUMBER:
        try:
            value = Decimal(sign + token.text)
        except InvalidOperation as exc:
            raise _parse_error("invalid numeric literal", token.pos) from exc
        normalized = format(value.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        if normalized in {"", "-0"}:
            normalized = "0"
        return Literal(Decimal(normalized), False)
    raise _parse_error("expected a string or numeric literal", token.pos)


def _parse_operand(tokens: Sequence[Token]) -> Operand:
    if not tokens:
        raise _parse_error("missing operand")
    try:
        return _parse_literal(tokens)
    except WorkbenchError:
        return _parse_column(tokens)


def _flip_operator(op: str) -> str:
    return {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(op, op)


def _parse_predicate(tokens: Sequence[Token]) -> Predicate:
    if not tokens:
        raise _parse_error("empty predicate")

    between = _find_top_level(tokens, "between")
    if between is not None:
        if _find_top_level(tokens, "between", between + 1) is not None:
            raise _parse_error("multiple BETWEEN clauses in one predicate")
        column = _parse_column(tokens[:between])
        tail = tokens[between + 1 :]
        split = _find_top_level(tail, "and")
        if split is None:
            raise _parse_error("BETWEEN requires low AND high bounds")
        low = _parse_literal(tail[:split])
        high = _parse_literal(tail[split + 1 :])
        return Predicate(kind="between", column=column, low=low, high=high)

    in_pos = _find_top_level(tokens, "in")
    if in_pos is not None:
        column = _parse_column(tokens[:in_pos])
        tail = tokens[in_pos + 1 :]
        if len(tail) < 3 or tail[0].kind != LPAREN or tail[-1].kind != RPAREN:
            raise _parse_error("IN requires a parenthesized literal list", tokens[in_pos].pos)
        inner = tail[1:-1]
        values: list[Literal] = []
        for part in _split_top_level(inner, COMMA):
            values.append(_parse_literal(part))
        if not values:
            raise _parse_error("IN list must not be empty")
        return Predicate(kind="in", column=column, values=tuple(values))

    operator_index: int | None = None
    depth = 0
    for index, token in enumerate(tokens):
        if token.kind == LPAREN:
            depth += 1
        elif token.kind == RPAREN:
            depth -= 1
        elif depth == 0 and token.kind == OP and token.text in {"=", "<>", "<", "<=", ">", ">="}:
            if operator_index is not None:
                raise _parse_error("multiple comparison operators in one predicate", token.pos)
            operator_index = index
    if operator_index is None:
        raise _parse_error("unsupported predicate; use comparison, IN or BETWEEN", tokens[0].pos)
    left_tokens = tokens[:operator_index]
    right_tokens = tokens[operator_index + 1 :]
    left = _parse_operand(left_tokens)
    right = _parse_operand(right_tokens)
    operator = tokens[operator_index].text
    if isinstance(left, Literal) and isinstance(right, ColumnRef):
        left, right = right, left
        operator = _flip_operator(operator)
    if not isinstance(left, ColumnRef):
        raise _parse_error("predicate must contain a column", tokens[operator_index].pos)
    return Predicate(kind="compare", column=left, op=operator, right=right)


def _parse_projection(tokens: Sequence[Token]) -> Projection:
    if len(tokens) == 1 and tokens[0].kind == STAR:
        return Projection(kind="star")
    if (
        len(tokens) == 3
        and tokens[0].kind == WORD
        and tokens[1].kind == DOT
        and tokens[2].kind == STAR
    ):
        return Projection(kind="table_star", table=tokens[0].lower)
    return Projection(kind="column", column=_parse_column(tokens))


def _parse_table_ref(tokens: Sequence[Token], start: int) -> tuple[TableRef, int]:
    if start >= len(tokens) or tokens[start].kind != WORD:
        position = tokens[start].pos if start < len(tokens) else None
        raise _parse_error("expected table name", position)
    table_token = tokens[start]
    name = table_token.lower
    if name in RESERVED_WORDS:
        raise _parse_error(f"expected table name, got keyword {name}", table_token.pos)
    index = start + 1
    alias: str | None = None
    if index < len(tokens) and _is_word(tokens[index], "as"):
        if index + 1 >= len(tokens) or tokens[index + 1].kind != WORD:
            raise _parse_error("AS requires an alias", tokens[index].pos)
        alias = tokens[index + 1].lower
        index += 2
    elif index < len(tokens) and tokens[index].kind == WORD and tokens[index].lower not in RESERVED_WORDS:
        alias = tokens[index].lower
        index += 1
    if alias in RESERVED_WORDS:
        raise _parse_error("reserved word cannot be used as alias", table_token.pos)
    return TableRef(name=name, alias=alias), index


def _parse_from(tokens: Sequence[Token]) -> tuple[tuple[TableRef, ...], tuple[Predicate, ...]]:
    if not tokens:
        raise _parse_error("FROM clause is empty")
    first, index = _parse_table_ref(tokens, 0)
    tables = [first]
    join_conditions: list[Predicate] = []
    if index == len(tokens):
        return tuple(tables), ()
    if _is_word(tokens[index], "inner"):
        index += 1
    if index >= len(tokens) or not _is_word(tokens[index], "join"):
        raise _parse_error("only one INNER JOIN between two tables is supported", tokens[index].pos)
    second, index = _parse_table_ref(tokens, index + 1)
    tables.append(second)
    if index >= len(tokens) or not _is_word(tokens[index], "on"):
        raise _parse_error("JOIN requires an ON condition", tokens[index].pos if index < len(tokens) else None)
    on_tokens = tokens[index + 1 :]
    if not on_tokens:
        raise _parse_error("JOIN ON condition is empty")
    for part in _split_and(on_tokens):
        predicate = _parse_predicate(part)
        if predicate.kind != "compare" or not isinstance(predicate.right, ColumnRef) or predicate.op != "=":
            raise _parse_error("JOIN ON supports equality between columns only")
        join_conditions.append(predicate)
    return tuple(tables), tuple(join_conditions)


def parse_select(sql: str) -> Query:
    if not isinstance(sql, str) or not sql.strip():
        raise _parse_error("query is empty")
    tokens = tokenize(sql)
    while tokens and tokens[-1].kind == SEMI:
        tokens.pop()
    if any(token.kind == SEMI for token in tokens):
        raise _parse_error("only one SELECT statement is allowed")
    if not tokens or not _is_word(tokens[0], "select"):
        raise _parse_error("only SELECT queries are supported")

    for index, token in enumerate(tokens):
        if token.kind != WORD:
            continue
        word = token.lower
        if word == "select" and index != 0:
            raise _parse_error("subqueries are not supported", token.pos)
        if word in FORBIDDEN_WORDS:
            raise _parse_error(f"unsupported clause or statement keyword: {word}", token.pos)
        if word in AGGREGATE_WORDS and index + 1 < len(tokens) and tokens[index + 1].kind == LPAREN:
            raise _parse_error("aggregate functions are not supported", token.pos)
        if word == "offset":
            raise _parse_error("OFFSET is not supported", token.pos)

    from_index = _find_top_level(tokens, "from", 1)
    if from_index is None or from_index == 1:
        raise _parse_error("SELECT requires a projection and FROM clause")
    projection_tokens = tokens[1:from_index]
    projection_parts = _split_top_level(projection_tokens, COMMA)
    projection = tuple(_parse_projection(part) for part in projection_parts)

    where_index = _find_top_level(tokens, "where", from_index + 1)
    order_index = _find_top_level(tokens, "order", from_index + 1)
    limit_index = _find_top_level(tokens, "limit", from_index + 1)
    clauses = [position for position in (where_index, order_index, limit_index) if position is not None]
    from_end = min(clauses) if clauses else len(tokens)
    tables, join_conditions = _parse_from(tokens[from_index + 1 : from_end])

    predicates: tuple[Predicate, ...] = ()
    if where_index is not None:
        next_positions = [
            position for position in (order_index, limit_index) if position is not None and position > where_index
        ]
        where_end = min(next_positions) if next_positions else len(tokens)
        where_tokens = tokens[where_index + 1 : where_end]
        if not where_tokens:
            raise _parse_error("WHERE clause is empty", tokens[where_index].pos)
        predicates = tuple(_parse_predicate(part) for part in _split_and(where_tokens))

    order_by: tuple[OrderTerm, ...] = ()
    if order_index is not None:
        if order_index + 1 >= len(tokens) or not _is_word(tokens[order_index + 1], "by"):
            raise _parse_error("ORDER must be followed by BY", tokens[order_index].pos)
        order_start = order_index + 2
        order_end = limit_index if limit_index is not None and limit_index > order_index else len(tokens)
        order_tokens = tokens[order_start:order_end]
        if not order_tokens:
            raise _parse_error("ORDER BY list is empty", tokens[order_index].pos)
        terms: list[OrderTerm] = []
        for part in _split_top_level(order_tokens, COMMA):
            descending = False
            body = list(part)
            if body and body[-1].kind == WORD and body[-1].lower in {"asc", "desc"}:
                descending = body[-1].lower == "desc"
                body = body[:-1]
            terms.append(OrderTerm(column=_parse_column(body), descending=descending))
        order_by = tuple(terms)

    limit: int | None = None
    if limit_index is not None:
        limit_tokens = tokens[limit_index + 1 :]
        if len(limit_tokens) != 1 or limit_tokens[0].kind != NUMBER or "." in limit_tokens[0].text:
            raise _parse_error("LIMIT requires one non-negative integer", tokens[limit_index].pos)
        limit = int(limit_tokens[0].text)

    if len(tables) > 2:
        raise _parse_error("at most two tables are supported")
    return Query(
        projection=projection,
        tables=tables,
        join_conditions=join_conditions,
        predicates=predicates,
        order_by=order_by,
        limit=limit,
    )


def parse(sql: str) -> Query:
    return parse_select(sql)


def is_slow(elapsed_ms: float, scanned_rows: int, threshold_ms: int, threshold_rows: int) -> bool:
    if not isinstance(threshold_ms, int) or isinstance(threshold_ms, bool) or threshold_ms < 0:
        raise fail("E_SESSION", "threshold_ms must be a non-negative integer")
    if not isinstance(threshold_rows, int) or isinstance(threshold_rows, bool) or threshold_rows < 0:
        raise fail("E_SESSION", "threshold_rows must be a non-negative integer")
    return elapsed_ms >= threshold_ms or scanned_rows >= threshold_rows


def merge_redundant_predicates(predicates: Iterable[Predicate]) -> tuple[tuple[Predicate, ...], list[str]] | None:
    """Return predicates with redundant bounds removed, or None when unchanged."""

    items = list(predicates)
    keep = [True] * len(items)
    rationale: list[str] = []

    seen_exact: dict[str, int] = {}
    for index, predicate in enumerate(items):
        key = predicate.sql()
        if key in seen_exact:
            keep[index] = False
            rationale.append(f"移除重复谓词 {key}")
        else:
            seen_exact[key] = index

    groups: dict[tuple[str | None, str, str], list[int]] = {}
    for index, predicate in enumerate(items):
        if not keep[index] or predicate.kind != "compare" or predicate.op not in {">", ">=", "<", "<="}:
            continue
        if not isinstance(predicate.right, Literal):
            continue
        direction = "lower" if predicate.op in {">", ">="} else "upper"
        key = (predicate.column.table, predicate.column.name, direction)
        groups.setdefault(key, []).append(index)

    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        values = []
        comparable = True
        for index in indices:
            right = items[index].right
            assert isinstance(right, Literal)
            if right.is_string != items[indices[0]].right.is_string:  # type: ignore[union-attr]
                comparable = False
                break
            values.append(right.value)
        if not comparable:
            continue
        direction = key[2]
        winner = indices[0]
        for index in indices[1:]:
            candidate_value = values[indices.index(index)]
            winner_value = values[indices.index(winner)]
            if direction == "lower":
                better = candidate_value > winner_value or (
                    candidate_value == winner_value and items[index].op == ">" and items[winner].op == ">="
                )
            else:
                better = candidate_value < winner_value or (
                    candidate_value == winner_value and items[index].op == "<" and items[winner].op == "<="
                )
            if better:
                winner = index
        for index in indices:
            if index != winner:
                keep[index] = False
        rationale.append(
            f"合并列 {items[winner].column.sql()} 的 {len(indices)} 个{'下界' if direction == 'lower' else '上界'}谓词"
        )

    if not rationale:
        return None
    return tuple(item for item, keep_item in zip(items, keep) if keep_item), rationale


def replace_query(query: Query, **changes: object) -> Query:
    return replace(query, **changes)
