"""Expression parsing, normalization and digests.

Grammar (precedence low -> high):
  ternary   := or_expr ('?' ternary ':' ternary)?
  or_expr   := and_expr (('or'|'||') and_expr)*
  and_expr  := not_expr (('and'|'&&') not_expr)*
  not_expr  := ('not'|'!') not_expr | cmp_expr
  cmp_expr  := add_expr (('=='|'!='|'<'|'<='|'>'|'>=') add_expr)?
  add_expr  := mul_expr (('+'|'-') mul_expr)*
  mul_expr  := unary (('*'|'/'|'%') unary)*
  unary     := '-' unary | primary
  primary   := number | ident | 'if' '(' ternary ',' ternary ',' ternary ')'
             | '(' ternary ')'

Normalization renders a fully-parenthesized prefix s-expression, so any two
texts that differ only in whitespace or redundant parentheses produce the
same normalized string and therefore the same digest.
"""

import hashlib
import re

from .errors import PlanError

_TOKEN_RE = re.compile(r"""
    (?P<num>\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>\|\||&&|==|!=|<=|>=|[-+*/%<>()?!,:])
  | (?P<ws>\s+)
""", re.VERBOSE)


class Num:
    def __init__(self, value):
        self.value = value


class Col:
    def __init__(self, name):
        self.name = name


class Bin:
    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right


class Un:
    def __init__(self, op, expr):
        self.op, self.expr = op, expr


class Ternary:
    def __init__(self, cond, then, otherwise):
        self.cond, self.then, self.otherwise = cond, then, otherwise


def _tokenize(text):
    tokens, pos = [], 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise PlanError("unexpected character", expr=text, pos=pos)
        pos = m.end()
        if m.lastgroup == "ws":
            continue
        tokens.append((m.lastgroup, m.group()))
    tokens.append(("end", ""))
    return tokens


class _Parser:
    def __init__(self, text):
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0

    def peek(self):
        return self.tokens[self.i]

    def next(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, value):
        kind, tok = self.next()
        if tok != value:
            raise PlanError(f"expected {value!r}, got {tok!r}", expr=self.text)

    def parse(self):
        node = self.ternary()
        if self.peek()[0] != "end":
            raise PlanError("trailing tokens", expr=self.text, token=self.peek()[1])
        return node

    def ternary(self):
        node = self.or_expr()
        if self.peek()[1] == "?":
            self.next()
            then = self.ternary()
            self.expect(":")
            otherwise = self.ternary()
            node = Ternary(node, then, otherwise)
        return node

    def or_expr(self):
        node = self.and_expr()
        while self.peek()[1] in ("or", "||"):
            self.next()
            node = Bin("or", node, self.and_expr())
        return node

    def and_expr(self):
        node = self.not_expr()
        while self.peek()[1] in ("and", "&&"):
            self.next()
            node = Bin("and", node, self.not_expr())
        return node

    def not_expr(self):
        if self.peek()[1] in ("not", "!"):
            self.next()
            return Un("not", self.not_expr())
        return self.cmp_expr()

    def cmp_expr(self):
        node = self.add_expr()
        if self.peek()[1] in ("==", "!=", "<", "<=", ">", ">="):
            op = self.next()[1]
            node = Bin(op, node, self.add_expr())
        return node

    def add_expr(self):
        node = self.mul_expr()
        while self.peek()[1] in ("+", "-"):
            op = self.next()[1]
            node = Bin(op, node, self.mul_expr())
        return node

    def mul_expr(self):
        node = self.unary()
        while self.peek()[1] in ("*", "/", "%"):
            op = self.next()[1]
            node = Bin(op, node, self.unary())
        return node

    def unary(self):
        if self.peek()[1] == "-":
            self.next()
            return Un("neg", self.unary())
        return self.primary()

    def primary(self):
        kind, tok = self.next()
        if kind == "num":
            return Num(float(tok) if ("." in tok or "e" in tok or "E" in tok) else int(tok))
        if kind == "ident":
            if tok == "if" and self.peek()[1] == "(":
                self.next()
                cond = self.ternary()
                self.expect(",")
                then = self.ternary()
                self.expect(",")
                otherwise = self.ternary()
                self.expect(")")
                return Ternary(cond, then, otherwise)
            return Col(tok)
        if tok == "(":
            node = self.ternary()
            self.expect(")")
            return node
        raise PlanError(f"unexpected token {tok!r}", expr=self.text)


def parse(text):
    return _Parser(text).parse()


def _fmt_num(value):
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def normalize(node):
    if isinstance(node, Num):
        return _fmt_num(node.value)
    if isinstance(node, Col):
        return node.name
    if isinstance(node, Bin):
        return f"({node.op} {normalize(node.left)} {normalize(node.right)})"
    if isinstance(node, Un):
        return f"({node.op} {normalize(node.expr)})"
    if isinstance(node, Ternary):
        return f"(if {normalize(node.cond)} {normalize(node.then)} {normalize(node.otherwise)})"
    raise TypeError(node)


def normalize_text(text):
    return normalize(parse(text))


def expr_digest(text):
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()
