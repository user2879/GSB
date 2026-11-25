"""Deterministic minimal YAML subset loader/dumper.

Only the construction used by this project is supported: block mappings,
block sequences, nested indentation, ``#`` comments and inline flow
sequences such as ``[1, 2]``. No external dependency (PyYAML) is required.
"""
from __future__ import annotations

from typing import Any


class YamlError(ValueError):
    pass


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token == "" or token in ("~", "null", "Null", "NULL"):
        return None
    if token in ("true", "True", "TRUE"):
        return True
    if token in ("false", "False", "FALSE"):
        return False
    if token.startswith(('"', "'")) and token.endswith(('"', "'")) and len(token) >= 2:
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _parse_flow(token: str) -> list[Any]:
    inner = token.strip()
    if not (inner.startswith("[") and inner.endswith("]")):
        raise YamlError(f"unsupported inline value: {token}")
    body = inner[1:-1].strip()
    if not body:
        return []
    return [_parse_scalar(part) for part in _split_flow(body)]


def _split_flow(body: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = []
    in_str: str | None = None
    for ch in body:
        if in_str:
            current.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            current.append(ch)
        elif ch in "[{":
            depth += 1
            current.append(ch)
        elif ch in "]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return parts


def _parse_value(token: str) -> Any:
    token = token.strip()
    if token.startswith("["):
        return _parse_flow(token)
    return _parse_scalar(token)


def loads(text: str) -> dict[str, Any]:
    raw_lines = text.splitlines()
    lines: list[tuple[int, str]] = []
    for line in raw_lines:
        if not _quoted_hash(line):
            stripped = line.split("#", 1)[0].rstrip()
        else:
            stripped = line.rstrip()
        if not stripped.strip() or stripped.lstrip(" ").startswith("#"):
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))

    pos = 0

    def parse_block(indent: int) -> Any:
        nonlocal pos
        if pos >= len(lines):
            return None
        first = lines[pos][1]
        if first.startswith("- "):
            seq: list[Any] = []
            while pos < len(lines) and lines[pos][0] == indent and lines[pos][1].startswith("- "):
                item_text = lines[pos][1][2:].strip()
                pos += 1
                if item_text:
                    seq.append(_parse_value(item_text))
                else:
                    seq.append(parse_block(lines[pos][0]) if pos < len(lines) else None)
            return seq
        mapping: dict[str, Any] = {}
        while pos < len(lines) and lines[pos][0] == indent and not lines[pos][1].startswith("- "):
            key_text = lines[pos][1]
            if ":" not in key_text:
                raise YamlError(f"malformed mapping line: {key_text}")
            key, _, value = key_text.partition(":")
            key = key.strip()
            value = value.strip()
            pos += 1
            if value:
                mapping[key] = _parse_value(value)
            else:
                if pos < len(lines) and lines[pos][0] > indent:
                    mapping[key] = parse_block(lines[pos][0])
                else:
                    mapping[key] = None
        return mapping

    if not lines:
        return {}
    result = parse_block(lines[0][0])
    if pos != len(lines):
        raise YamlError("unparsed trailing YAML content")
    if not isinstance(result, dict):
        raise YamlError("top-level YAML document must be a mapping")
    return result


def _quoted_hash(line: str) -> bool:
    quote: str | None = None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#":
            return True
    return False


def dumps(data: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(data, dict):
        chunks = []
        for key, value in data.items():
            if isinstance(value, (dict, list)) and value:
                chunks.append(f"{pad}{key}:\n{dumps(value, indent + 2)}")
            elif isinstance(value, dict):
                chunks.append(f"{pad}{key}: {{}}\n")
            elif isinstance(value, list):
                chunks.append(f"{pad}{key}: []\n")
            else:
                chunks.append(f"{pad}{key}: {_dump_scalar(value)}\n")
        return "".join(chunks)
    if isinstance(data, list):
        chunks = []
        for value in data:
            if isinstance(value, (dict, list)):
                inner = dumps(value, indent + 2)
                inner_lines = inner.rstrip("\n").split("\n")
                head = inner_lines[0].lstrip(" ")
                chunks.append(f"{pad}- {head}\n")
                chunks.extend(f"{line}\n" for line in inner_lines[1:])
            else:
                chunks.append(f"{pad}- {_dump_scalar(value)}\n")
        return "".join(chunks)
    return f"{pad}{_dump_scalar(data)}\n"


def _dump_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)
