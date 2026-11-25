from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def line_at(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def parse_with_lines(text: str) -> tuple[Any, dict[str, int]]:
    """Parse JSON and map JSON-pointer-like paths to source line numbers.

    The parser is intentionally recursive over JSON values.  It uses the stdlib
    decoder for scalars/strings but tracks positions itself, which keeps E_SPEC
    diagnostics tied to the first offending field in the original file.
    """

    decoder = json.JSONDecoder()
    lines: dict[str, int] = {}
    n = len(text)

    def skip_ws(pos: int) -> int:
        while pos < n and text[pos] in " \t\r\n":
            pos += 1
        return pos

    def parse(pos: int, path: str) -> tuple[Any, int]:
        pos = skip_ws(pos)
        if pos >= n:
            raise json.JSONDecodeError("Expecting value", text, pos)
        lines[path or "$"] = line_at(text, pos)
        ch = text[pos]

        if ch == "{":
            obj: dict[str, Any] = {}
            pos += 1
            pos = skip_ws(pos)
            if pos < n and text[pos] == "}":
                return obj, pos + 1
            while True:
                pos = skip_ws(pos)
                key, key_end = decoder.raw_decode(text, pos)
                if not isinstance(key, str):
                    raise json.JSONDecodeError("Object key must be a string", text, pos)
                key_path = f"{path}.{key}" if path else f"$.{key}"
                lines[key_path] = line_at(text, pos)
                pos = skip_ws(key_end)
                if pos >= n or text[pos] != ":":
                    raise json.JSONDecodeError("Expecting ':' delimiter", text, pos)
                pos += 1
                value, pos = parse(pos, key_path)
                obj[key] = value
                pos = skip_ws(pos)
                if pos >= n:
                    raise json.JSONDecodeError("Unterminated object", text, pos)
                if text[pos] == ",":
                    pos += 1
                    continue
                if text[pos] == "}":
                    return obj, pos + 1
                raise json.JSONDecodeError("Expecting ',' or '}'", text, pos)

        if ch == "[":
            arr: list[Any] = []
            pos += 1
            pos = skip_ws(pos)
            if pos < n and text[pos] == "]":
                return arr, pos + 1
            index = 0
            while True:
                item_path = f"{path}[{index}]"
                value, pos = parse(pos, item_path)
                arr.append(value)
                pos = skip_ws(pos)
                if pos >= n:
                    raise json.JSONDecodeError("Unterminated array", text, pos)
                if text[pos] == ",":
                    pos += 1
                    index += 1
                    continue
                if text[pos] == "]":
                    return arr, pos + 1
                raise json.JSONDecodeError("Expecting ',' or ']'", text, pos)

        value, end = decoder.raw_decode(text, pos)
        return value, end

    value, end = parse(0, "$")
    end = skip_ws(end)
    if end != n:
        raise json.JSONDecodeError("Extra data", text, end)
    return value, lines


def load_json_file(path: str | Path) -> tuple[Any, dict[str, int], str]:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        from .errors import SpecError

        raise SpecError(f"cannot read JSON file {p.name}: {exc}", "$", 1) from exc
    try:
        value, lines = parse_with_lines(text)
    except json.JSONDecodeError as exc:
        from .errors import SpecError

        raise SpecError(
            f"invalid JSON in {p.name}: {exc.msg}",
            field="$",
            line=exc.lineno,
        ) from exc
    return value, lines, text


def load_jsonl_file(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    p = Path(path)
    records: list[dict[str, Any]] = []
    first_bad_line = 0
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            from .errors import SpecError

            first_bad_line = lineno
            raise SpecError(
                f"invalid JSON line in {p.name}: {exc.msg}",
                field="$",
                line=lineno,
            ) from exc
        if not isinstance(value, dict):
            from .errors import SpecError

            raise SpecError("JSONL record must be an object", field="$", line=lineno)
        value["__line__"] = lineno
        records.append(value)
    return records, first_bad_line
