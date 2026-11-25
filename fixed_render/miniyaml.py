"""Small YAML subset loader with PyYAML when available.

The generated configuration and command files use only mappings, lists and
scalars.  Keeping this fallback lets the project run with just numpy/Pillow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return parse_yaml(text)


def parse_yaml(text: str) -> Any:
    raw_lines: list[tuple[int, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        raw_lines.append((indent, line.strip()))

    def scalar(token: str) -> Any:
        token = token.strip()
        if token in {"null", "~", "None"}:
            return None
        if token in {"true", "True", "yes"}:
            return True
        if token in {"false", "False", "no"}:
            return False
        if (token.startswith('"') and token.endswith('"')) or (
            token.startswith("'") and token.endswith("'")
        ):
            return token[1:-1]
        if token.startswith("[") and token.endswith("]"):
            inner = token[1:-1].strip()
            if not inner:
                return []
            return [scalar(item.strip()) for item in _split_inline(inner)]
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            return token

    def _split_inline(text_value: str) -> list[str]:
        items: list[str] = []
        start = 0
        depth = 0
        quote: str | None = None
        for index, char in enumerate(text_value):
            if char in {'"', "'"}:
                quote = None if quote == char else char if quote is None else quote
            elif quote is None and char in "[{":
                depth += 1
            elif quote is None and char in "]}":
                depth -= 1
            elif quote is None and char == "," and depth == 0:
                items.append(text_value[start:index])
                start = index + 1
        items.append(text_value[start:])
        return items

    def block(index: int, indent: int) -> tuple[Any, int]:
        if raw_lines[index][1].startswith("- "):
            result: list[Any] = []
            while index < len(raw_lines) and raw_lines[index][0] == indent:
                content = raw_lines[index][1]
                if not content.startswith("- "):
                    break
                item_text = content[2:].strip()
                if ":" in item_text and not item_text.startswith(("[", '"', "'")):
                    key, value = item_text.split(":", 1)
                    item: dict[str, Any] = {key.strip(): scalar(value) if value.strip() else {}}
                    index += 1
                    while index < len(raw_lines) and raw_lines[index][0] > indent:
                        child, index = block(index, raw_lines[index][0])
                        if isinstance(child, dict):
                            item.update(child)
                        else:
                            item[key.strip()] = child
                        break
                    result.append(item)
                elif item_text:
                    result.append(scalar(item_text))
                    index += 1
                else:
                    index += 1
                    child, index = block(index, raw_lines[index][0])
                    result.append(child)
            return result, index

        mapping: dict[str, Any] = {}
        while index < len(raw_lines) and raw_lines[index][0] == indent:
            content = raw_lines[index][1]
            if content.startswith("- ") or ":" not in content:
                break
            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                mapping[key] = scalar(value)
                index += 1
            else:
                index += 1
                if index < len(raw_lines) and raw_lines[index][0] > indent:
                    child, index = block(index, raw_lines[index][0])
                    mapping[key] = child
                else:
                    mapping[key] = None
        return mapping, index

    if not raw_lines:
        return {}
    result, _ = block(0, raw_lines[0][0])
    return result


def dump_yaml(value: Any, indent: int = 0) -> str:
    """Tiny deterministic YAML emitter for generated sample/config files."""
    pad = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(dump_yaml(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{pad}{key}: {_emit_scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                first = True
                for key, child in item.items():
                    prefix = f"{pad}- " if first else f"{pad}  "
                    if isinstance(child, (dict, list)):
                        lines.append(f"{prefix}{key}:")
                        lines.append(dump_yaml(child, indent + 4).rstrip("\n"))
                    else:
                        lines.append(f"{prefix}{key}: {_emit_scalar(child)}")
                    first = False
            elif isinstance(item, list):
                lines.append(f"{pad}-")
                lines.append(dump_yaml(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{pad}- {_emit_scalar(item)}")
    else:
        lines.append(pad + _emit_scalar(value))
    return "\n".join(lines) + "\n"


def _emit_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if any(char in value for char in ":#\n") or value == "":
            return '"' + value.replace('"', '\\"') + '"'
        return value
    return str(value)
