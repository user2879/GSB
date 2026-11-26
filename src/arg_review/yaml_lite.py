from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "":
        return None
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def loads(text: str) -> dict[str, Any]:
    """Parse the small indentation-based YAML subset used by this project."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))

    pos = 0

    def parse_block(indent: int) -> Any:
        nonlocal pos
        if pos >= len(lines):
            return None
        first = lines[pos][1]
        if first.startswith("- ") or first == "-":
            items: list[Any] = []
            while pos < len(lines) and lines[pos][0] == indent and (
                lines[pos][1].startswith("- ") or lines[pos][1] == "-"
            ):
                item_text = lines[pos][1]
                rest = item_text[1:].strip()
                if not rest:
                    pos += 1
                    if pos < len(lines) and lines[pos][0] > indent:
                        items.append(parse_block(lines[pos][0]))
                    else:
                        items.append(None)
                elif ":" in rest and not rest.startswith(('"', "'")):
                    key, value = rest.split(":", 1)
                    item: dict[str, Any] = {key.strip(): _parse_scalar(value)}
                    pos += 1
                    child_indent = indent + 2
                    while pos < len(lines) and lines[pos][0] > indent:
                        if lines[pos][0] < child_indent:
                            child_indent = lines[pos][0]
                        nested = parse_mapping(child_indent, allow_single=True)
                        if nested:
                            item.update(nested)
                    items.append(item)
                else:
                    items.append(_parse_scalar(rest))
                    pos += 1
            return items
        return parse_mapping(indent)

    def parse_mapping(indent: int, allow_single: bool = False) -> dict[str, Any]:
        nonlocal pos
        result: dict[str, Any] = {}
        while pos < len(lines):
            current_indent, content = lines[pos]
            if current_indent < indent:
                break
            if current_indent > indent:
                if allow_single:
                    return parse_block(current_indent)
                break
            if content.startswith("- ") or content == "-":
                break
            if ":" not in content:
                raise ValueError(f"Invalid YAML line: {content}")
            key, raw_value = content.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            pos += 1
            if raw_value:
                result[key] = _parse_scalar(raw_value)
            elif pos < len(lines) and lines[pos][0] > indent:
                result[key] = parse_block(lines[pos][0])
            else:
                result[key] = {}
        return result

    if not lines:
        return {}
    return parse_block(lines[0][0])


def load(path: str | Path) -> dict[str, Any]:
    return loads(Path(path).read_text(encoding="utf-8"))
