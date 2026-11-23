"""Minimal JSON parser that records a 1-based line number for every value.

Used so that E_SPEC validation errors can point at the exact line of the
first illegal field. Only the JSON grammar is supported (no comments).
"""

from . import E_SPEC
from .errors import DiagError


class JNode:
    __slots__ = ("value", "line")

    def __init__(self, value, line):
        self.value = value
        self.line = line

    def __repr__(self):
        return f"JNode(line={self.line}, value={self.value!r})"


_ESCAPES = {
    '"': '"', "\\": "\\", "/": "/",
    "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t",
}


class _Parser:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.line = 1

    def fail(self, msg):
        raise DiagError(E_SPEC, f"JSON parse error: {msg}", line=self.line)

    def _advance(self):
        ch = self.text[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
        return ch

    def _skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self._advance()

    def parse(self):
        self._skip_ws()
        node = self._value()
        self._skip_ws()
        if self.pos != len(self.text):
            self.fail("trailing characters after top-level value")
        return node

    def _value(self):
        if self.pos >= len(self.text):
            self.fail("unexpected end of input")
        ch = self.text[self.pos]
        if ch == "{":
            return self._object()
        if ch == "[":
            return self._array()
        if ch == '"':
            line = self.line
            return JNode(self._string(), line)
        if ch == "t":
            return self._literal("true", True)
        if ch == "f":
            return self._literal("false", False)
        if ch == "n":
            return self._literal("null", None)
        if ch == "-" or ch.isdigit():
            return self._number()
        self.fail(f"unexpected character {ch!r}")

    def _literal(self, word, value):
        line = self.line
        for expected in word:
            if self.pos >= len(self.text) or self._advance() != expected:
                self.fail(f"invalid literal, expected {word!r}")
        return JNode(value, line)

    def _number(self):
        line = self.line
        start = self.pos
        if self.text[self.pos] == "-":
            self._advance()
        if self.pos >= len(self.text) or not self.text[self.pos].isdigit():
            self.fail("invalid number")
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self._advance()
        is_float = False
        if self.pos < len(self.text) and self.text[self.pos] == ".":
            is_float = True
            self._advance()
            if self.pos >= len(self.text) or not self.text[self.pos].isdigit():
                self.fail("invalid number fraction")
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self._advance()
        if self.pos < len(self.text) and self.text[self.pos] in "eE":
            is_float = True
            self._advance()
            if self.pos < len(self.text) and self.text[self.pos] in "+-":
                self._advance()
            if self.pos >= len(self.text) or not self.text[self.pos].isdigit():
                self.fail("invalid number exponent")
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self._advance()
        raw = self.text[start:self.pos]
        return JNode(float(raw) if is_float else int(raw), line)

    def _string(self):
        # opening quote
        self._advance()
        out = []
        while True:
            if self.pos >= len(self.text):
                self.fail("unterminated string")
            ch = self._advance()
            if ch == '"':
                return "".join(out)
            if ch == "\\":
                if self.pos >= len(self.text):
                    self.fail("unterminated escape")
                esc = self._advance()
                if esc == "u":
                    hex_digits = ""
                    for _ in range(4):
                        if self.pos >= len(self.text):
                            self.fail("unterminated \\u escape")
                        hex_digits += self._advance()
                    try:
                        out.append(chr(int(hex_digits, 16)))
                    except ValueError:
                        self.fail("invalid \\u escape")
                elif esc in _ESCAPES:
                    out.append(_ESCAPES[esc])
                else:
                    self.fail(f"invalid escape \\{esc}")
            else:
                out.append(ch)

    def _object(self):
        line = self.line
        self._advance()  # {
        obj = {}
        self._skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == "}":
            self._advance()
            return JNode(obj, line)
        while True:
            self._skip_ws()
            if self.pos >= len(self.text) or self.text[self.pos] != '"':
                self.fail("expected object key string")
            key = self._string()
            self._skip_ws()
            if self.pos >= len(self.text) or self._advance() != ":":
                self.fail("expected ':' after object key")
            self._skip_ws()
            obj[key] = self._value()
            self._skip_ws()
            if self.pos >= len(self.text):
                self.fail("unterminated object")
            ch = self._advance()
            if ch == "}":
                return JNode(obj, line)
            if ch != ",":
                self.fail("expected ',' or '}' in object")

    def _array(self):
        line = self.line
        self._advance()  # [
        items = []
        self._skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == "]":
            self._advance()
            return JNode(items, line)
        while True:
            self._skip_ws()
            items.append(self._value())
            self._skip_ws()
            if self.pos >= len(self.text):
                self.fail("unterminated array")
            ch = self._advance()
            if ch == "]":
                return JNode(items, line)
            if ch != ",":
                self.fail("expected ',' or ']' in array")


def parse(text):
    """Parse JSON text into a JNode tree (dict/list/scalars wrapped)."""
    return _Parser(text).parse()


def parse_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        return parse(fh.read())
