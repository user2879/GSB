from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .errors import fail
from .fixedpoint import (
    ceil_div,
    div_round,
    floor_div,
    normalize_error_terms,
    pow10,
)


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    position: int


class ExpressionSyntaxError(ValueError):
    def __init__(self, message: str, position: int):
        super().__init__(message)
        self.position = position


def tokenize(expression: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char.isspace():
            index += 1
            continue
        if char.isdigit() or (char == "." and index + 1 < len(expression) and expression[index + 1].isdigit()):
            start = index
            seen_dot = False
            while index < len(expression):
                current = expression[index]
                if current.isdigit():
                    index += 1
                elif current == "." and not seen_dot:
                    seen_dot = True
                    index += 1
                else:
                    break
            if index < len(expression) and expression[index] in "eE":
                exponent_start = index
                index += 1
                if index < len(expression) and expression[index] in "+-":
                    index += 1
                digit_start = index
                while index < len(expression) and expression[index].isdigit():
                    index += 1
                if digit_start == index:
                    raise ExpressionSyntaxError("malformed numeric exponent", exponent_start)
            tokens.append(Token("number", expression[start:index], start))
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < len(expression) and (expression[index].isalnum() or expression[index] == "_"):
                index += 1
            tokens.append(Token("identifier", expression[start:index], start))
            continue
        if char == '"':
            start = index
            index += 1
            value: list[str] = []
            while index < len(expression):
                if expression[index] == '"':
                    if index + 1 < len(expression) and expression[index + 1] == '"':
                        value.append('"')
                        index += 2
                        continue
                    index += 1
                    break
                value.append(expression[index])
                index += 1
            else:
                raise ExpressionSyntaxError("unterminated quoted identifier", start)
            tokens.append(Token("identifier", "".join(value), start))
            continue
        if char in "+-*/()":
            tokens.append(Token(char, char, index))
            index += 1
            continue
        raise ExpressionSyntaxError(f"unsupported character {char!r}", index)
    tokens.append(Token("eof", "", len(expression)))
    return tokens


class Node:
    def references(self) -> set[str]:
        raise NotImplementedError

    def evaluate(
        self,
        resolver: Callable[[str], "EvalValue"],
        *,
        target_scale: int,
        rounding: str,
    ) -> "EvalValue":
        raise NotImplementedError


@dataclass(frozen=True)
class NumberNode(Node):
    text: str

    def references(self) -> set[str]:
        return set()

    def evaluate(self, resolver, *, target_scale: int, rounding: str) -> "EvalValue":
        value = self.text
        lowered = value.lower()
        if "e" in lowered:
            mantissa, exponent_text = lowered.split("e", 1)
            exponent = int(exponent_text)
        else:
            mantissa = lowered
            exponent = 0
        if "." in mantissa:
            whole, fraction = mantissa.split(".", 1)
        else:
            whole, fraction = mantissa, ""
        digits = (whole or "0") + fraction
        scale = len(fraction) - exponent
        if scale < 0:
            digits += "0" * (-scale)
            scale = 0
        units = int(digits)
        return EvalValue(nominal=units, scale=scale, lo=units, hi=units, terms={})


@dataclass(frozen=True)
class ColumnNode(Node):
    name: str

    def references(self) -> set[str]:
        return {self.name}

    def evaluate(self, resolver, *, target_scale: int, rounding: str) -> "EvalValue":
        return resolver(self.name)


@dataclass(frozen=True)
class UnaryNode(Node):
    operator: str
    operand: Node

    def references(self) -> set[str]:
        return self.operand.references()

    def evaluate(self, resolver, *, target_scale: int, rounding: str) -> "EvalValue":
        value = self.operand.evaluate(
            resolver, target_scale=target_scale, rounding=rounding
        )
        if self.operator == "+":
            return value
        return EvalValue(
            nominal=-value.nominal,
            scale=value.scale,
            lo=-value.hi,
            hi=-value.lo,
            terms=dict(value.terms),
        )


@dataclass(frozen=True)
class BinaryNode(Node):
    operator: str
    left: Node
    right: Node

    def references(self) -> set[str]:
        return self.left.references() | self.right.references()

    def evaluate(self, resolver, *, target_scale: int, rounding: str) -> "EvalValue":
        left = self.left.evaluate(
            resolver, target_scale=target_scale, rounding=rounding
        )
        right = self.right.evaluate(
            resolver, target_scale=target_scale, rounding=rounding
        )
        if self.operator in {"+", "-"}:
            return _add_sub(left, right, self.operator)
        if self.operator == "*":
            return _multiply(left, right)
        if self.operator == "/":
            return _divide(
                left,
                right,
                target_scale=target_scale,
                rounding=rounding,
            )
        raise AssertionError(f"unexpected operator {self.operator}")


@dataclass
class EvalValue:
    nominal: int
    scale: int
    lo: int
    hi: int
    terms: dict[tuple[str, ...], int] = field(default_factory=dict)


def _merge_terms(
    left: dict[tuple[str, ...], int], right: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    merged = dict(left)
    for path, units in right.items():
        merged[path] = merged.get(path, 0) + units
    return merged


def _scaled_terms(
    terms: dict[tuple[str, ...], int], factor: int
) -> dict[tuple[str, ...], int]:
    return {path: units * factor for path, units in terms.items()}


def _align(value: EvalValue, scale: int) -> EvalValue:
    if value.scale == scale:
        return value
    if value.scale > scale:
        raise ValueError("cannot align to a coarser internal scale")
    factor = pow10(scale - value.scale)
    return EvalValue(
        nominal=value.nominal * factor,
        scale=scale,
        lo=value.lo * factor,
        hi=value.hi * factor,
        terms=_scaled_terms(value.terms, factor),
    )


def _add_sub(left: EvalValue, right: EvalValue, operator: str) -> EvalValue:
    scale = max(left.scale, right.scale)
    left_aligned = _align(left, scale)
    right_aligned = _align(right, scale)
    if operator == "+":
        return EvalValue(
            nominal=left_aligned.nominal + right_aligned.nominal,
            scale=scale,
            lo=left_aligned.lo + right_aligned.lo,
            hi=left_aligned.hi + right_aligned.hi,
            terms=_merge_terms(left_aligned.terms, right_aligned.terms),
        )
    return EvalValue(
        nominal=left_aligned.nominal - right_aligned.nominal,
        scale=scale,
        lo=left_aligned.lo - right_aligned.hi,
        hi=left_aligned.hi - right_aligned.lo,
        terms=_merge_terms(left_aligned.terms, right_aligned.terms),
    )


def _multiply(left: EvalValue, right: EvalValue) -> EvalValue:
    products = [
        left.lo * right.lo,
        left.lo * right.hi,
        left.hi * right.lo,
        left.hi * right.hi,
    ]
    terms: dict[tuple[str, ...], int] = {}
    for path, units in left.terms.items():
        terms[path] = terms.get(path, 0) + units * abs(right.nominal)
    for path, units in right.terms.items():
        terms[path] = terms.get(path, 0) + units * abs(left.nominal)
    return EvalValue(
        nominal=left.nominal * right.nominal,
        scale=left.scale + right.scale,
        lo=min(products),
        hi=max(products),
        terms=terms,
    )


def _ratio_to_units_floor(
    numerator: int, denominator: int, *, output_scale: int, input_scale_delta: int
) -> int:
    scaled_numerator = numerator * pow10(output_scale + input_scale_delta)
    return floor_div(scaled_numerator, denominator)


def _ratio_to_units_ceil(
    numerator: int, denominator: int, *, output_scale: int, input_scale_delta: int
) -> int:
    scaled_numerator = numerator * pow10(output_scale + input_scale_delta)
    return ceil_div(scaled_numerator, denominator)


def _divide(
    left: EvalValue,
    right: EvalValue,
    *,
    target_scale: int,
    rounding: str,
) -> EvalValue:
    if right.lo <= 0 <= right.hi:
        raise ZeroDivisionError("division interval contains zero")
    output_scale = max(target_scale + 4, left.scale - right.scale, 0) + 2
    input_scale_delta = right.scale - left.scale
    if output_scale + input_scale_delta < 0:
        output_scale = -input_scale_delta + 4
    nominal_numerator = left.nominal * pow10(output_scale + input_scale_delta)
    nominal = div_round(nominal_numerator, right.nominal, rounding)
    corners = []
    for numerator in (left.lo, left.hi):
        for denominator in (right.lo, right.hi):
            corners.append(
                _ratio_to_units_floor(
                    numerator,
                    denominator,
                    output_scale=output_scale,
                    input_scale_delta=input_scale_delta,
                )
            )
            corners.append(
                _ratio_to_units_ceil(
                    numerator,
                    denominator,
                    output_scale=output_scale,
                    input_scale_delta=input_scale_delta,
                )
            )
    terms: dict[tuple[str, ...], int] = {}
    denominator_abs = abs(right.nominal)
    for path, units in left.terms.items():
        contribution = ceil_div(
            units * pow10(output_scale + right.scale - left.scale),
            denominator_abs,
        )
        terms[path] = terms.get(path, 0) + contribution
    for path, units in right.terms.items():
        contribution = ceil_div(
            units * pow10(output_scale + right.scale) * abs(left.nominal),
            pow10(left.scale) * denominator_abs * denominator_abs,
        )
        terms[path] = terms.get(path, 0) + contribution
    return EvalValue(
        nominal=nominal,
        scale=output_scale,
        lo=min(corners),
        hi=max(corners),
        terms=terms,
    )


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.position = 0

    def peek(self) -> Token:
        return self.tokens[self.position]

    def take(self) -> Token:
        token = self.tokens[self.position]
        self.position += 1
        return token

    def parse(self) -> Node:
        node = self.parse_additive()
        token = self.peek()
        if token.kind != "eof":
            raise ExpressionSyntaxError("unexpected token after expression", token.position)
        return node

    def parse_additive(self) -> Node:
        node = self.parse_multiplicative()
        while self.peek().kind in {"+", "-"}:
            operator = self.take().text
            node = BinaryNode(operator, node, self.parse_multiplicative())
        return node

    def parse_multiplicative(self) -> Node:
        node = self.parse_unary()
        while self.peek().kind in {"*", "/"}:
            operator = self.take().text
            node = BinaryNode(operator, node, self.parse_unary())
        return node

    def parse_unary(self) -> Node:
        if self.peek().kind in {"+", "-"}:
            operator = self.take().text
            return UnaryNode(operator, self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Node:
        token = self.peek()
        if token.kind == "number":
            self.take()
            return NumberNode(token.text)
        if token.kind == "identifier":
            self.take()
            return ColumnNode(token.text)
        if token.kind == "(":
            self.take()
            node = self.parse_additive()
            closing = self.peek()
            if closing.kind != ")":
                raise ExpressionSyntaxError("expected closing parenthesis", closing.position)
            self.take()
            return node
        raise ExpressionSyntaxError("expected a number, column, or parenthesis", token.position)


@dataclass(frozen=True)
class ParsedExpression:
    text: str
    ast: Node
    references: tuple[str, ...]

    def evaluate(
        self,
        resolver: Callable[[str], EvalValue],
        *,
        target_scale: int,
        rounding: str,
    ) -> "ExpressionResult":
        value = self.ast.evaluate(
            resolver, target_scale=target_scale, rounding=rounding
        )
        if value.scale == target_scale:
            factor = 1
            quantized = value.nominal
            lo_target, hi_target = value.lo, value.hi
        elif value.scale < target_scale:
            factor = pow10(target_scale - value.scale)
            quantized = value.nominal * factor
            lo_target, hi_target = value.lo * factor, value.hi * factor
        else:
            factor = pow10(value.scale - target_scale)
            quantized = div_round(value.nominal, factor, rounding)
            lo_target = floor_div(value.lo, factor)
            hi_target = ceil_div(value.hi, factor)
        interval_bound = max(quantized - lo_target, hi_target - quantized, 0)
        scaled_terms: dict[tuple[str, ...], int] = {}
        for path, units in value.terms.items():
            if value.scale < target_scale:
                contribution = units * factor
            elif value.scale > target_scale:
                contribution = ceil_div(units, factor)
            else:
                contribution = units
            if contribution:
                scaled_terms[path] = scaled_terms.get(path, 0) + contribution
        terms, error_bound = normalize_error_terms(
            scaled_terms,
            bound=interval_bound,
            residual_path=("rounding",),
        )
        return ExpressionResult(
            units=int(quantized),
            scale=target_scale,
            error_bound=int(error_bound),
            interval_bound_units=int(interval_bound),
            terms=terms,
            term_units={tuple(term["path"]): int(term["units"]) for term in terms},
            references=self.references,
        )


@dataclass(frozen=True)
class ExpressionResult:
    units: int
    scale: int
    error_bound: int
    interval_bound_units: int
    terms: list[dict[str, object]]
    term_units: dict[tuple[str, ...], int]
    references: tuple[str, ...]

    def paths_payload(self) -> list[dict[str, object]]:
        return [dict(term) for term in self.terms]


def parse_expression(
    expression: str,
    *,
    step_id: str | None,
    field_path: str = "expression",
) -> ParsedExpression:
    if not isinstance(expression, str) or not expression.strip():
        fail(
            "E_MIGRATE",
            "backfill expression must be a non-empty string",
            step_id=step_id,
            field_path=field_path,
        )
    try:
        tokens = tokenize(expression)
        ast = Parser(tokens).parse()
    except ExpressionSyntaxError as exc:
        fail(
            "E_MIGRATE",
            f"invalid numeric expression: {exc}",
            step_id=step_id,
            field_path=field_path,
            position=exc.position,
        )
    references = tuple(sorted(ast.references()))
    return ParsedExpression(text=expression, ast=ast, references=references)


def collect_references(expression: str) -> tuple[str, ...]:
    return parse_expression(expression, step_id=None).references
