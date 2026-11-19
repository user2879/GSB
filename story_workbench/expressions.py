"""Small, intentionally restricted arithmetic language for edge effects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .canonical import add_double, double, multiply_double, negate_double
from .errors import SpecInvalid


ExprNode = tuple[Any, ...]

_NUMBER_RE = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    offset: int


def _tokenize(expression: str, *, location: dict[str, Any]) -> tuple[Token, ...]:
    tokens: list[Token] = []
    position = 0
    while position < len(expression):
        char = expression[position]
        if char.isspace():
            position += 1
            continue
        if char in "+*()":
            tokens.append(Token(char, char, position))
            position += 1
            continue
        if char == "-":
            # Minus is accepted only as a unary sign. Binary subtraction is not.
            tokens.append(Token("UNARY_MINUS", char, position))
            position += 1
            continue
        number = _NUMBER_RE.match(expression, position)
        if number:
            text = number.group(0)
            try:
                value = double(text)
            except (TypeError, ValueError) as exc:
                raise SpecInvalid(
                    f"effect expression contains a non-finite number: {text}",
                    location=location,
                ) from exc
            tokens.append(Token("NUMBER", repr(value), position))
            position = number.end()
            continue
        ident = _IDENT_RE.match(expression, position)
        if ident:
            tokens.append(Token("IDENT", ident.group(0), position))
            position = ident.end()
            continue
        if expression.startswith("**", position) or char == "/":
            raise SpecInvalid(
                f"effect expression operator {char!r} is not allowed; only + and * are permitted",
                location={**location, "offset": position},
            )
        raise SpecInvalid(
            f"unexpected character {char!r} in effect expression",
            location={**location, "offset": position},
        )
    tokens.append(Token("EOF", "", len(expression)))
    return tuple(tokens)


class _Parser:
    def __init__(self, tokens: Sequence[Token], location: dict[str, Any]) -> None:
        self.tokens = tokens
        self.location = location
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def consume(self, kind: str) -> Token:
        token = self.current
        if token.kind != kind:
            raise SpecInvalid(
                f"expected {kind}, found {token.kind}",
                location={**self.location, "offset": token.offset},
            )
        self.index += 1
        return token

    def parse(self) -> ExprNode:
        node = self.parse_sum()
        if self.current.kind != "EOF":
            raise SpecInvalid(
                f"unexpected token {self.current.value!r}",
                location={**self.location, "offset": self.current.offset},
            )
        return node

    def parse_sum(self) -> ExprNode:
        node = self.parse_product()
        while self.current.kind == "+":
            self.consume("+")
            right = self.parse_product()
            node = ("add", node, right)
        return node

    def parse_product(self) -> ExprNode:
        node = self.parse_factor()
        while self.current.kind == "*":
            self.consume("*")
            right = self.parse_factor()
            node = ("mul", node, right)
        return node

    def parse_factor(self) -> ExprNode:
        token = self.current
        if token.kind == "NUMBER":
            self.consume("NUMBER")
            return ("num", float(token.value))
        if token.kind == "IDENT":
            self.consume("IDENT")
            return ("var", token.value)
        if token.kind == "(":
            self.consume("(")
            node = self.parse_sum()
            self.consume(")")
            return node
        if token.kind in {"+", "UNARY_MINUS"}:
            if token.kind == "+":
                self.consume("+")
                return self.parse_factor()
            self.consume("UNARY_MINUS")
            child = self.parse_factor()
            return ("neg", child)
        raise SpecInvalid(
            f"expected a number, component, or parenthesized expression; found {token.value!r}",
            location={**self.location, "offset": token.offset},
        )


def parse_expression(expression: str, *, location: dict[str, Any]) -> ExprNode:
    if not isinstance(expression, str) or not expression.strip():
        raise SpecInvalid("effect expression must be a non-empty string", location=location)
    return _Parser(_tokenize(expression, location=location), location).parse()


def validate_expression(
    node: ExprNode,
    *,
    components: Sequence[str],
    location: dict[str, Any],
) -> None:
    allowed = set(components)
    kind = node[0]
    if kind == "num":
        return
    if kind == "var":
        if node[1] not in allowed:
            raise SpecInvalid(
                f"unknown state variable {node[1]!r} in effect expression",
                location=location,
                details={"allowed_variables": list(components)},
            )
        return
    if kind == "neg":
        validate_expression(node[1], components=components, location=location)
        return
    if kind in {"add", "mul"}:
        validate_expression(node[1], components=components, location=location)
        validate_expression(node[2], components=components, location=location)
        return
    raise SpecInvalid(f"unknown expression node {kind!r}", location=location)


def render_expression(node: ExprNode) -> str:
    kind = node[0]
    if kind == "num":
        return repr(double(node[1]))
    if kind == "var":
        return node[1]
    if kind == "neg":
        return f"(-{render_expression(node[1])})"
    if kind == "add":
        return f"({render_expression(node[1])} + {render_expression(node[2])})"
    if kind == "mul":
        return f"({render_expression(node[1])} * {render_expression(node[2])})"
    raise ValueError(f"unknown expression node {kind!r}")


def coefficient_count(node: ExprNode) -> int:
    kind = node[0]
    if kind == "num":
        return 1
    if kind == "var":
        return 0
    if kind == "neg":
        return coefficient_count(node[1])
    if kind in {"add", "mul"}:
        return coefficient_count(node[1]) + coefficient_count(node[2])
    raise ValueError(f"unknown expression node {kind!r}")


def replace_coefficient(node: ExprNode, ordinal: int, value: float) -> ExprNode:
    if ordinal < 0:
        raise IndexError("negative coefficient ordinal")
    counter = [0]

    def visit(current: ExprNode) -> ExprNode:
        kind = current[0]
        if kind == "num":
            index = counter[0]
            counter[0] += 1
            if index == ordinal:
                return ("num", double(value))
            return current
        if kind == "var":
            return current
        if kind == "neg":
            return ("neg", visit(current[1]))
        if kind in {"add", "mul"}:
            return (kind, visit(current[1]), visit(current[2]))
        raise ValueError(f"unknown expression node {kind!r}")

    result = visit(node)
    if ordinal >= counter[0]:
        raise IndexError(f"coefficient ordinal {ordinal} out of range (count={counter[0]})")
    return result


def evaluate_expression(
    node: ExprNode,
    state: Mapping[str, float],
    *,
    nonfinite_callback: Callable[[], None] | None = None,
) -> float:
    kind = node[0]
    try:
        if kind == "num":
            return double(node[1])
        if kind == "var":
            return double(state[node[1]])
        if kind == "neg":
            return negate_double(evaluate_expression(node[1], state, nonfinite_callback=nonfinite_callback))
        if kind == "add":
            left = evaluate_expression(node[1], state, nonfinite_callback=nonfinite_callback)
            right = evaluate_expression(node[2], state, nonfinite_callback=nonfinite_callback)
            return add_double(left, right)
        if kind == "mul":
            left = evaluate_expression(node[1], state, nonfinite_callback=nonfinite_callback)
            right = evaluate_expression(node[2], state, nonfinite_callback=nonfinite_callback)
            return multiply_double(left, right)
    except (ValueError, OverflowError):
        if nonfinite_callback is not None:
            nonfinite_callback()
        raise
    raise ValueError(f"unknown expression node {kind!r}")
