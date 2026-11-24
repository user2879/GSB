from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .errors import SpecError


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_hex(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def stable_seed64(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def normalize_decimal(value: Any, field: str) -> tuple[str, Decimal]:
    if isinstance(value, bool) or value is None:
        raise SpecError(field, "must be a finite decimal number")
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, int):
        decimal_value = Decimal(value)
    elif isinstance(value, float):
        decimal_value = Decimal(repr(value))
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise SpecError(field, "must not be an empty decimal string")
        try:
            decimal_value = Decimal(text)
        except InvalidOperation as exc:
            raise SpecError(field, "must be a finite decimal number") from exc
    else:
        raise SpecError(field, "must be an integer, number, or decimal string")
    if not decimal_value.is_finite():
        raise SpecError(field, "must be finite")
    text = format(decimal_value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return text, Decimal(text)


def read_json_strict(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise SpecError("$", f"non-finite JSON number {token!r} is not allowed")

    def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SpecError("$", f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError("$", f"cannot read specification: {exc}") from exc
    try:
        return json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=no_duplicate_object,
        )
    except SpecError:
        raise
    except json.JSONDecodeError as exc:
        raise SpecError(
            "$",
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ) from exc

