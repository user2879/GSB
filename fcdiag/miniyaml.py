"""A tiny deterministic YAML subset parser.

Only the features required by pipeline definitions are supported:

* nested mappings via two-space indentation
* lists of mappings (``- key: value``)
* scalars: integers, floats, booleans, null and quoted/plain strings

The parser rejects tabs and inconsistent indentation instead of silently
guessing, which keeps validation locators meaningful.
"""

from __future__ import annotations

from typing import Any, List, Tuple


class MiniYamlError(ValueError):
    def __init__(self, line_no: int, message: str):
        self.line_no = line_no
        super().__init__(f"line {line_no}: {message}")


def _scalar(text: str) -> Any:
    text = text.strip()
    if text == "" or text in ("null", "~"):
        return None
    if text == "[]":
        return []
    if text == "{}":
        return {}
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_scalar(part) for part in inner.split(",")] if inner else []
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _tokenize(text: str) -> List[Tuple[int, str, int]]:
    tokens: List[Tuple[int, str, int]] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise MiniYamlError(idx, "tabs are not allowed for indentation")
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        if indent % 2 != 0:
            raise MiniYamlError(idx, "indentation must use multiples of two spaces")
        tokens.append((indent, stripped.strip(), idx))
    return tokens


def parse_yaml(text: str) -> dict:
    """Parse the supported YAML subset into plain Python containers."""

    tokens = _tokenize(text)
    if not tokens:
        return {}
    value, pos = _parse_block(tokens, 0, tokens[0][0])
    if pos != len(tokens):
        indent, _, line_no = tokens[pos]
        raise MiniYamlError(line_no, f"unexpected indentation {indent}")
    return value


def _parse_block(tokens, pos: int, indent: int):
    if tokens[pos][1].startswith("- "):
        return _parse_list(tokens, pos, indent)
    result = {}
    while pos < len(tokens):
        cur_indent, content, line_no = tokens[pos]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise MiniYamlError(line_no, "unexpected deeper indentation")
        if content.startswith("- "):
            raise MiniYamlError(line_no, "list where mapping was expected")
        if ":" not in content:
            raise MiniYamlError(line_no, "mapping entry must contain ':'")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        pos += 1
        if raw_value:
            result[key] = _scalar(raw_value)
            continue
        if pos < len(tokens) and tokens[pos][0] > indent:
            child, pos = _parse_block(tokens, pos, tokens[pos][0])
            result[key] = child
        else:
            result[key] = None
    return result, pos


def _parse_list(tokens, pos: int, indent: int):
    items: List[Any] = []
    while pos < len(tokens):
        cur_indent, content, line_no = tokens[pos]
        if cur_indent != indent or not content.startswith("- "):
            break
        first = content[2:].strip()
        synthetic = [(indent + 2, first, line_no)]
        # Remaining keys of this list item sit two spaces deeper than '-'.
        item_indent = indent + 2
        end = pos + 1
        while end < len(tokens) and tokens[end][0] >= item_indent:
            synthetic.append(tokens[end])
            end += 1
        if ":" not in first:
            items.append(_scalar(first))
            pos = end
            continue
        item, consumed = _parse_block(synthetic, 0, item_indent)
        if consumed != len(synthetic):
            raise MiniYamlError(line_no, "malformed list item")
        items.append(item)
        pos = end
    return items, pos
