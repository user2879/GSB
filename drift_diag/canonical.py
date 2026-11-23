"""Canonical JSON, digests, and strict local-file parsing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .errors import format_error


def canonical_json(value: Any) -> str:
    """Return byte-stable JSON (UTF-8 data, sorted keys, compact separators)."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_digest(value: Any) -> str:
    return sha256_text(canonical_json(value))


def stable_int(*parts: Any, bits: int = 60) -> int:
    """Derive a positive integer that only changes when its inputs change."""

    raw = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).hexdigest()
    return int(raw[: (bits + 3) // 4], 16) & ((1 << bits) - 1)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_strict(text: str, *, source: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise format_error(
            f"invalid JSON in {source}: {exc.msg}",
            item=source,
            field="<json>",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc
    except ValueError as exc:
        raise format_error(str(exc), item=source, field="<json>") from exc


def load_json_strict(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise format_error(f"cannot read {file_path}: {exc}", item=str(file_path), field="<file>") from exc
    return parse_json_strict(text, source=str(file_path))
