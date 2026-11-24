"""JSON parser that remembers the source line of every object/key/element.

Validation errors must point at the *first* offending field and its line in
the original sample file. The stdlib ``json`` module does not retain source
positions, so batchdiag ships a small recursive-descent parser that mirrors
the standard grammar and builds a :class:`LineMap` alongside the value.
"""

from .errors import FieldError


class LineMap:
    """Maps dotted/key-tuple paths to 1-based source line numbers."""

    def __init__(self):
        self._lines = {}

    def record(self, path, line):
        self._lines[tuple(path)] = line

    def line_of(self, *path):
        return self._lines.get(tuple(path))


class _Tokenizer:
    def __init__(self, text, source):
        self.text = text
        self.source = source
        self.pos = 0
        self.line = 1

    def _advance(self, chunk):
        self.line += chunk.count("\n")
        self.pos += len(chunk)

    def skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self._advance(self.text[self.pos])

    def expect(self, literal):
        self.skip_ws()
        if not self.text.startswith(literal, self.pos):
            raise FieldError(f"expected {literal!r}", line=self.line,
                             source=self.source)
        self._advance(literal)

    def next_token(self):
        self.skip_ws()
        if self.pos >= len(self.text):
            raise FieldError("unexpected end of document", line=self.line,
                             source=self.source)
        ch = self.text[self.pos]
        if ch in "{[:}],":
            self._advance(ch)
            return ch
        if ch == '"':
            return self._string()
        if ch == "-" or ch.isdigit():
            return self._number()
        for word in ("true", "false", "null"):
            if self.text.startswith(word, self.pos):
                self._advance(word)
                return word
        raise FieldError(f"unexpected character {ch!r}", line=self.line,
                         source=self.source)

    def _string(self):
        start_line = self.line
        self._advance('"')
        chars = []
        while True:
            if self.pos >= len(self.text):
                raise FieldError("unterminated string", line=start_line,
                                 source=self.source)
            ch = self.text[self.pos]
            if ch == '"':
                self._advance('"')
                return "str", "".join(chars)
            if ch == "\\":
                esc = self.text[self.pos:self.pos + 2]
                self._advance(esc)
                mapping = {'"': '"', "\\": "\\", "/": "/", "b": "\b",
                           "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
                if esc == "\\u":
                    hex4 = self.text[self.pos:self.pos + 4]
                    if len(hex4) != 4 or any(
                            c not in "0123456789abcdefABCDEF"
                            for c in hex4):
                        raise FieldError("bad \\u escape", line=self.line,
                                         source=self.source)
                    chars.append(chr(int(hex4, 16)))
                    self._advance(hex4)
                else:
                    chars.append(mapping.get(esc, ""))
            else:
                chars.append(ch)
                self._advance(ch)

    def _number(self):
        start = self.pos
        start_line = self.line
        if self.peek() == "-":
            self._advance("-")
        if self.peek() == "0":
            self._advance("0")
        elif self.peek().isdigit():
            while self.peek().isdigit():
                self._advance(self.peek())
        else:
            raise FieldError("bad number", line=start_line,
                             source=self.source)
        if self.peek() == ".":
            self._advance(".")
            if not self.peek().isdigit():
                raise FieldError("bad number: digits expected after '.'",
                                 line=self.line, source=self.source)
            while self.peek().isdigit():
                self._advance(self.peek())
        if self.peek() in ("e", "E"):
            self._advance(self.peek())
            if self.peek() in ("+", "-"):
                self._advance(self.peek())
            if not self.peek().isdigit():
                raise FieldError("bad number: digits expected in exponent",
                                 line=self.line, source=self.source)
            while self.peek().isdigit():
                self._advance(self.peek())
        raw = self.text[start:self.pos]
        try:
            if any(c in raw for c in ".eE"):
                return "float", float(raw)
            return "int", int(raw)
        except ValueError:
            raise FieldError(f"bad number {raw!r}", line=start_line,
                             source=self.source)

    def peek(self):
        return self.text[self.pos] if self.pos < len(self.text) else ""


def _scalar(token, tok):
    if isinstance(token, tuple):
        return token[1]
    if token in ("true", "false"):
        return token == "true"
    if token == "null":
        return None
    raise FieldError(f"unexpected token {token!r}", line=tok.line,
                     source=tok.source)


def _parse_token(token, path, tok, lines):
    if token == "{":
        return _parse_object(tok, path, lines)
    if token == "[":
        return _parse_array(tok, path, lines)
    return _scalar(token, tok)


def _parse_value(tok, path, lines):
    return _parse_token(tok.next_token(), path, tok, lines)


def _parse_object(tok, path, lines):
    result = {}
    lines.record(path, tok.line)
    token = tok.next_token()
    if token == "}":
        return result
    while True:
        if not (isinstance(token, tuple) and token[0] == "str"):
            raise FieldError("object key must be a string", line=tok.line,
                             source=tok.source)
        key = token[1]
        key_path = tuple(list(path) + [key])
        lines.record(key_path, tok.line)
        tok.expect(":")
        result[key] = _parse_value(tok, key_path, lines)
        token = tok.next_token()
        if token == ",":
            token = tok.next_token()
            continue
        if token == "}":
            return result
        raise FieldError("expected ',' or '}' in object", line=tok.line,
                         source=tok.source)


def _parse_array(tok, path, lines):
    result = []
    lines.record(path, tok.line)
    token = tok.next_token()
    if token == "]":
        return result
    while True:
        index_path = tuple(list(path) + [len(result)])
        lines.record(index_path, tok.line)
        result.append(_parse_token(token, index_path, tok, lines))
        token = tok.next_token()
        if token == ",":
            token = tok.next_token()
            continue
        if token == "]":
            return result
        raise FieldError("expected ',' or ']' in array", line=tok.line,
                         source=tok.source)


def loads(text: str, source: str = "<json>"):
    """Parse *text* and return ``(value, line_map)``."""
    tok = _Tokenizer(text, source)
    lines = LineMap()
    value = _parse_value(tok, (), lines)
    tok.skip_ws()
    if tok.pos != len(tok.text):
        raise FieldError("trailing data after document", line=tok.line,
                         source=source)
    return value, lines


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return loads(text, source=str(path))
