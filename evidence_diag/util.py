from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        return str(value)
    normalized = value.normalize()
    if normalized == 0:
        return "0"
    return format(normalized, "f")


def unit_decimal(units: int, scale: int) -> str:
    return decimal_text(Decimal(units) / Decimal(scale))


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle, parse_float=Decimal)


def load_jsonl(path: str | Path) -> list[Any]:
    values: list[Any] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values.append(json.loads(stripped, parse_float=Decimal))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: {exc.msg}") from exc
    return values


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
    )


def deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted(set(values))
