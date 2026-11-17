"""A dependency-free YAML subset parser for local workbench configuration.

Supported: nested block maps/lists, flow lists/maps, quoted and plain scalars,
comments, booleans, null, integers and floats. Anchors, aliases, tags and block
scalars are intentionally rejected so malformed inputs cannot be silently
accepted.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


class YamlLiteError(ValueError):
    pass


_KEY_RE = re.compile(r"^([^:#\[\]{},]+?):(?:\s*(.*))?$")


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()


def _split_flow(text: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == delimiter and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def _find_mapping_colon(text: str) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == ":" and depth == 0:
            return i
    return -1


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "" or text in ("~", "null", "Null", "NULL"):
        return None
    if text in ("true", "True", "TRUE"):
        return True
    if text in ("false", "False", "FALSE"):
        return False
    if text.startswith("[") or text.startswith("{"):
        negate = text.startswith("!")
        source = text[1:] if negate else text
        if negate:
            source = "!" + source
        try:
            return ast.literal_eval(_yaml_flow_to_python(source))
        except (ValueError, SyntaxError) as exc:
            raise YamlLiteError(f"invalid flow value: {text}") from exc
    if text.startswith('"') and text.endswith('"'):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise YamlLiteError(f"invalid double-quoted string: {text}") from exc
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1].replace("''", "'")
    if re.fullmatch(r"[-+]?[0-9]+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?", text):
        try:
            return float(text)
        except ValueError:
            pass
    if text.startswith(("&", "*", "!", "|", ">")):
        raise YamlLiteError(f"unsupported YAML construct: {text}")
    return text


def _yaml_flow_to_python(text: str) -> str:
    """Translate bare flow-map keys/scalars to Python literals without eval."""

    out: list[str] = []
    i = 0
    expecting_key = False
    stack: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch in "[{":
            stack.append(ch)
            out.append(ch)
            expecting_key = ch == "{"
            i += 1
            continue
        if ch in "]}":
            if stack:
                stack.pop()
            out.append(ch)
            expecting_key = False
            i += 1
            continue
        if ch == ":" and stack and stack[-1] == "{":
            out.append(":")
            expecting_key = False
            i += 1
            continue
        if ch == "," and stack:
            out.append(",")
            expecting_key = stack[-1] == "{"
            i += 1
            continue
        if ch.isspace():
            out.append(ch)
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            escaped = False
            while j < len(text):
                if escaped:
                    escaped = False
                elif text[j] == "\\" and quote == '"':
                    escaped = True
                elif text[j] == quote:
                    break
                j += 1
            out.append(text[i : j + 1])
            i = j + 1
            continue
        j = i
        while j < len(text) and text[j] not in ",:]}":
            j += 1
        token = text[i:j].strip()
        if not token:
            i = j
            continue
        parsed = _parse_plain_flow_scalar(token)
        if expecting_key and stack and stack[-1] == "{":
            out.append(repr(str(parsed)))
            expecting_key = False
        elif isinstance(parsed, str):
            out.append(repr(parsed))
        else:
            out.append(repr(parsed))
        i = j
    return "".join(out)


def _parse_plain_flow_scalar(token: str) -> Any:
    if token in ("true", "True"):
        return True
    if token in ("false", "False"):
        return False
    if token in ("null", "Null", "~"):
        return None
    if re.fullmatch(r"[-+]?[0-9]+", token):
        return int(token)
    if re.fullmatch(r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?", token):
        return float(token)
    return token.strip()


class _Parser:
    def __init__(self, text: str, source: str):
        self.source = source
        self.lines: list[tuple[int, str, int]] = []
        for number, raw in enumerate(text.splitlines(), 1):
            if "\t" in raw[: len(raw) - len(raw.lstrip())]:
                raise YamlLiteError(f"{source}:{number}: tabs are not allowed for indentation")
            cleaned = _strip_comment(raw)
            if not cleaned.strip() or cleaned.strip() in ("---", "..."):
                continue
            indent = len(cleaned) - len(cleaned.lstrip(" "))
            self.lines.append((indent, cleaned[indent:], number))

    def parse(self) -> Any:
        if not self.lines:
            return None
        value, index = self._block(0, self.lines[0][0])
        if index != len(self.lines):
            number = self.lines[index][2]
            raise YamlLiteError(f"{self.source}:{number}: unexpected indentation")
        return value

    def _block(self, index: int, indent: int) -> tuple[Any, int]:
        if index >= len(self.lines):
            return None, index
        if self.lines[index][1].startswith("- "):
            return self._sequence(index, indent)
        return self._mapping(index, indent)

    def _mapping(self, index: int, indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while index < len(self.lines):
            line_indent, content, number = self.lines[index]
            if line_indent < indent:
                break
            if line_indent > indent:
                raise YamlLiteError(f"{self.source}:{number}: unexpected indentation")
            if content.startswith("- "):
                break
            match = _KEY_RE.match(content)
            if not match:
                raise YamlLiteError(f"{self.source}:{number}: expected 'key: value'")
            key = str(_parse_scalar(match.group(1).strip()))
            if key in result:
                raise YamlLiteError(f"{self.source}:{number}: duplicate key {key!r}")
            value_text = match.group(2) or ""
            index += 1
            if value_text.strip() == "":
                if index < len(self.lines) and self.lines[index][0] > indent:
                    value, index = self._block(index, self.lines[index][0])
                else:
                    value = None
            else:
                value = _parse_scalar(value_text)
            result[key] = value
        return result, index

    def _sequence(self, index: int, indent: int) -> tuple[list[Any], int]:
        result: list[Any] = []
        while index < len(self.lines):
            line_indent, content, number = self.lines[index]
            if line_indent < indent:
                break
            if line_indent != indent or not content.startswith("-"):
                break
            rest = content[1:]
            if rest and not rest.startswith(" "):
                raise YamlLiteError(f"{self.source}:{number}: expected space after '-'")
            rest = rest[1:] if rest.startswith(" ") else ""
            index += 1
            if rest == "":
                if index < len(self.lines) and self.lines[index][0] > indent:
                    value, index = self._block(index, self.lines[index][0])
                else:
                    value = None
                result.append(value)
                continue
            colon = _find_mapping_colon(rest)
            if colon >= 0:
                item: dict[str, Any] = {}
                key = str(_parse_scalar(rest[:colon].strip()))
                tail = rest[colon + 1 :].strip()
                if tail:
                    item[key] = _parse_scalar(tail)
                elif index < len(self.lines) and self.lines[index][0] > indent:
                    child_indent = self.lines[index][0]
                    if self.lines[index][1].startswith("- "):
                        item[key], index = self._sequence(index, child_indent)
                    else:
                        item[key], index = self._mapping(index, child_indent)
                else:
                    item[key] = None
                if index < len(self.lines) and self.lines[index][0] > indent:
                    continuation_indent = self.lines[index][0]
                    extra, index = self._mapping(index, continuation_indent)
                    item.update(extra)
                result.append(item)
            else:
                result.append(_parse_scalar(rest))
        return result, index


def loads(text: str, source: str = "<string>") -> Any:
    return _Parser(text, source).parse()


def load(path: str | Path) -> Any:
    file_path = Path(path)
    return loads(file_path.read_text(encoding="utf-8"), str(file_path))
