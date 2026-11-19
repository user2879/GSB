"""Expression parser, canonical renderer, and structural digest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .canonical import digest_value
from .errors import AppError


_TOKEN_RE = re.compile(
    r"\s*(?:(?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op><=|>=|==|!=|&&|\|\||[+\-*/%<>!?:(),]))"
)


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


@dataclass(frozen=True)
class Literal:
    value: int | float | bool | None
    literal_kind: str


@dataclass(frozen=True)
class ColumnRef:
    name: str


@dataclass(frozen=True)
class Unary:
    op: str
    operand: Any


@dataclass(frozen=True)
class Binary:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class Ternary:
    condition: Any
    when_true: Any
    when_false: Any


_BINARY_ALIASES = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "%": "%",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "==": "==",
    "!=": "!=",
    "and": "and",
    "or": "or",
}
_COMPARISON_OPS = {"<", "<=", ">", ">=", "==", "!="}
_PRECEDENCE = {
    "ternary": 1,
    "or": 2,
    "and": 3,
    "not": 4,
    "comparison": 5,
    "add": 6,
    "mul": 7,
    "unary": 8,
    "atom": 9,
}


def _tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if not match:
            if text[pos:].strip() == "":
                break
            raise AppError("E_DATA", f"表达式词法错误，位置 {pos}: {text[pos:pos + 12]!r}")
        kind = match.lastgroup or ""
        value = match.group(kind)
        tokens.append(Token(kind, value, match.start(kind)))
        pos = match.end()
    tokens.append(Token("eof", "", len(text)))
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.current
        self.index += 1
        return token

    def accept(self, text: str) -> bool:
        current = self.current
        if current.text == text or (current.kind == "ident" and current.text.lower() == text):
            self.advance()
            return True
        return False

    def expect(self, text: str) -> Token:
        if self.accept(text):
            return self.tokens[self.index - 1]
        raise AppError("E_DATA", f"表达式语法错误，期望 {text!r}，实际 {self.current.text!r}，位置 {self.current.pos}")

    def parse(self) -> Any:
        node = self.parse_ternary()
        if self.current.kind != "eof":
            raise AppError("E_DATA", f"表达式有多余内容 {self.current.text!r}，位置 {self.current.pos}")
        return node

    def parse_ternary(self) -> Any:
        condition = self.parse_or()
        if self.accept("?"):
            when_true = self.parse_ternary()
            self.expect(":")
            when_false = self.parse_ternary()
            return Ternary(condition, when_true, when_false)
        return condition

    def parse_or(self) -> Any:
        node = self.parse_and()
        while self.accept("or") or self.accept("||"):
            node = Binary("or", node, self.parse_and())
        return node

    def parse_and(self) -> Any:
        node = self.parse_not()
        while self.accept("and") or self.accept("&&"):
            node = Binary("and", node, self.parse_not())
        return node

    def parse_not(self) -> Any:
        if self.accept("not") or self.accept("!"):
            return Unary("not", self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> Any:
        node = self.parse_additive()
        while self.current.text in _COMPARISON_OPS:
            op = self.advance().text
            node = Binary(op, node, self.parse_additive())
        return node

    def parse_additive(self) -> Any:
        node = self.parse_multiplicative()
        while self.current.text in {"+", "-"}:
            op = self.advance().text
            node = Binary(op, node, self.parse_multiplicative())
        return node

    def parse_multiplicative(self) -> Any:
        node = self.parse_unary()
        while self.current.text in {"*", "/", "%"}:
            op = self.advance().text
            node = Binary(op, node, self.parse_unary())
        return node

    def parse_unary(self) -> Any:
        if self.accept("+"):
            return Unary("+", self.parse_unary())
        if self.accept("-"):
            return Unary("-", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Any:
        token = self.current
        if token.kind == "number":
            self.advance()
            if any(ch in token.text for ch in ".eE"):
                return Literal(float(token.text), "float")
            return Literal(int(token.text), "integer")
        if token.kind == "ident":
            self.advance()
            lowered = token.text.lower()
            if lowered == "true":
                return Literal(True, "boolean")
            if lowered == "false":
                return Literal(False, "boolean")
            if lowered == "null":
                return Literal(None, "null")
            if self.accept("("):
                if self.current.text == ")":
                    raise AppError("E_DATA", f"函数调用缺少参数，位置 {token.pos}")
                if lowered != "if":
                    raise AppError("E_DATA", f"不支持的函数 {token.text!r}，位置 {token.pos}")
                args = [self.parse_ternary()]
                while self.accept(","):
                    args.append(self.parse_ternary())
                self.expect(")")
                if len(args) != 3:
                    raise AppError("E_DATA", f"if() 需要 3 个参数，实际 {len(args)}", {"position": token.pos})
                return Ternary(args[0], args[1], args[2])
            return ColumnRef(token.text)
        if self.accept("("):
            value = self.parse_ternary()
            self.expect(")")
            return value
        raise AppError("E_DATA", f"表达式语法错误，位置 {token.pos}: {token.text!r}")


def _literal_ast(node: Literal) -> list[Any]:
    if node.literal_kind == "null":
        return ["literal", "null", None]
    if node.literal_kind == "float":
        value: Any = float(node.value)  # type: ignore[arg-type]
        if value == 0:
            value = 0.0
        return ["literal", "float", value]
    if node.literal_kind == "integer":
        return ["literal", "integer", int(node.value)]  # type: ignore[arg-type]
    return ["literal", "boolean", bool(node.value)]


def _literal_text(node: Literal) -> str:
    if node.literal_kind == "null":
        return "null"
    if node.literal_kind == "boolean":
        return "true" if node.value else "false"
    if node.literal_kind == "float":
        value = float(node.value)  # type: ignore[arg-type]
        text = repr(value)
        return "0.0" if text == "-0.0" else text
    return str(node.value)


def ast_payload(node: Any) -> list[Any]:
    if isinstance(node, Literal):
        return _literal_ast(node)
    if isinstance(node, ColumnRef):
        return ["column", node.name]
    if isinstance(node, Unary):
        return ["unary", node.op, ast_payload(node.operand)]
    if isinstance(node, Binary):
        return ["binary", node.op, ast_payload(node.left), ast_payload(node.right)]
    if isinstance(node, Ternary):
        return ["ternary", ast_payload(node.condition), ast_payload(node.when_true), ast_payload(node.when_false)]
    raise TypeError(f"unknown AST node {type(node)!r}")


def _node_precedence(node: Any) -> int:
    if isinstance(node, Ternary):
        return _PRECEDENCE["ternary"]
    if isinstance(node, Binary):
        if node.op == "or":
            return _PRECEDENCE["or"]
        if node.op == "and":
            return _PRECEDENCE["and"]
        if node.op in _COMPARISON_OPS:
            return _PRECEDENCE["comparison"]
        if node.op in {"+", "-"}:
            return _PRECEDENCE["add"]
        return _PRECEDENCE["mul"]
    if isinstance(node, Unary):
        return _PRECEDENCE["not"] if node.op == "not" else _PRECEDENCE["unary"]
    return _PRECEDENCE["atom"]


def _render(node: Any, parent_prec: int = 0, side: str = "none", *, parent_is_binary: bool = False) -> str:
    prec = _node_precedence(node)
    if isinstance(node, Literal):
        text = _literal_text(node)
    elif isinstance(node, ColumnRef):
        text = node.name
    elif isinstance(node, Unary):
        if node.op == "not":
            operand = _render(node.operand, 0, "right")
            if not isinstance(node.operand, (Literal, ColumnRef)) and not (
                isinstance(node.operand, Unary) and node.operand.op == "not"
            ):
                operand = f"({operand})"
            text = f"not {operand}"
        else:
            operand = _render(node.operand, prec, "right")
            text = f"{node.op}{operand}"
    elif isinstance(node, Binary):
        left = _render(node.left, prec, "left", parent_is_binary=True)
        right = _render(node.right, prec, "right", parent_is_binary=True)
        text = f"{left} {node.op} {right}"
    elif isinstance(node, Ternary):
        condition = _render(node.condition, _PRECEDENCE["or"], "none")
        when_true = _render(node.when_true, 1, "middle")
        when_false = _render(node.when_false, 1, "right")
        text = f"{condition} ? {when_true} : {when_false}"
    else:
        raise TypeError(f"unknown AST node {type(node)!r}")

    needs = prec < parent_prec
    if parent_is_binary and prec == parent_prec and side == "right":
        needs = True
    if isinstance(node, Ternary) and parent_prec == 1 and side in {"middle", "right"}:
        needs = False
    return f"({text})" if needs else text


def _collect_columns(node: Any, output: set[str]) -> None:
    if isinstance(node, ColumnRef):
        output.add(node.name)
    elif isinstance(node, Unary):
        _collect_columns(node.operand, output)
    elif isinstance(node, Binary):
        _collect_columns(node.left, output)
        _collect_columns(node.right, output)
    elif isinstance(node, Ternary):
        _collect_columns(node.condition, output)
        _collect_columns(node.when_true, output)
        _collect_columns(node.when_false, output)


@dataclass(frozen=True)
class ParsedExpression:
    source: str
    root: Any
    normalized: str
    structural_digest: str
    columns: tuple[str, ...]

    def ast_payload(self) -> list[Any]:
        return ast_payload(self.root)


def parse_expression(text: str) -> ParsedExpression:
    if not isinstance(text, str) or not text.strip():
        raise AppError("E_DATA", "表达式不能为空")
    root = _Parser(text).parse()
    normalized = _render(root)
    payload = ast_payload(root)
    columns: set[str] = set()
    _collect_columns(root, columns)
    return ParsedExpression(
        source=text,
        root=root,
        normalized=normalized,
        structural_digest=digest_value(payload),
        columns=tuple(sorted(columns)),
    )


def normalize_expression(text: str) -> str:
    return parse_expression(text).normalized
