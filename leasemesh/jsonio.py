from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .errors import E_SPEC, SpecError


@dataclass
class JsonDocument:
    path: Path
    data: Any
    text: str
    lines: dict[str, int]

    def line(self, field: str) -> Optional[int]:
        candidates = [field]
        if field.startswith("$."):
            candidates.append(field[2:])
        else:
            candidates.append("$." + field)
        for candidate in candidates:
            if candidate in self.lines:
                return self.lines[candidate]
        parent = field.rsplit(".", 1)[0] if "." in field else ""
        while parent:
            if parent in self.lines:
                return self.lines[parent]
            parent = parent.rsplit(".", 1)[0] if "." in parent else ""
        return None


def build_line_map(text: str) -> dict[str, int]:
    """Map dotted/bracketed JSON paths to source line numbers."""
    line_starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            line_starts.append(index + 1)

    def line_at(position: int) -> int:
        return bisect.bisect_right(line_starts, position)

    decoder = json.JSONDecoder()
    lines: dict[str, int] = {}

    def skip_ws(position: int) -> int:
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        return position

    def parse_value(position: int, path: str) -> int:
        position = skip_ws(position)
        if position >= len(text):
            raise SpecError("JSON ended unexpectedly", field=path or "$", line=line_at(position))
        char = text[position]

        if char == "{":
            position += 1
            while True:
                position = skip_ws(position)
                if position >= len(text):
                    raise SpecError("unterminated object", field=path or "$", line=line_at(position))
                if text[position] == "}":
                    return position + 1

                key, key_end = decoder.raw_decode(text, position)
                key_path = f"{path}.{key}" if path else str(key)
                key_line = line_at(position)
                if key_path in lines:
                    raise SpecError(
                        f"duplicate JSON key: {key}",
                        field=key_path,
                        line=key_line,
                    )
                lines[key_path] = key_line
                if not isinstance(key, str):
                    raise SpecError("object key must be a string", field=key_path, line=key_line)

                position = skip_ws(key_end)
                if position >= len(text) or text[position] != ":":
                    raise SpecError("expected ':' after object key", field=key_path, line=key_line)
                position = parse_value(position + 1, key_path)
                position = skip_ws(position)
                if position < len(text) and text[position] == ",":
                    position += 1
                    continue
                if position < len(text) and text[position] == "}":
                    return position + 1
                raise SpecError("expected ',' or '}'", field=key_path, line=line_at(position))

        if char == "[":
            position += 1
            element_index = 0
            while True:
                position = skip_ws(position)
                if position >= len(text):
                    raise SpecError("unterminated array", field=path or "$", line=line_at(position))
                if text[position] == "]":
                    return position + 1
                element_path = f"{path}[{element_index}]"
                lines[element_path] = line_at(position)
                position = parse_value(position, element_path)
                position = skip_ws(position)
                if position < len(text) and text[position] == ",":
                    position += 1
                    element_index += 1
                    continue
                if position < len(text) and text[position] == "]":
                    return position + 1
                raise SpecError(
                    "expected ',' or ']'",
                    field=f"{path}[{element_index}]",
                    line=line_at(position),
                )

        _, end = decoder.raw_decode(text, position)
        return end

    parse_value(0, "")
    return lines


def load_json_document(path: str | Path) -> JsonDocument:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read JSON file {path}: {exc}", field="path") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}",
            field="$",
            line=exc.lineno,
            details={"column": exc.colno},
        ) from exc
    lines = build_line_map(text)
    return JsonDocument(path=path, data=data, text=text, lines=lines)


def first_problem(problems: list[SpecError]) -> Optional[SpecError]:
    if not problems:
        return None
    return min(problems, key=lambda item: (item.line or 10**12, item.field or ""))
