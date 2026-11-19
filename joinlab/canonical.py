from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Sequence


def normalize_float(value: float) -> float:
    if value == 0.0:
        return 0.0
    return float(value)


def normalize_json(value: Any) -> Any:
    """Return a JSON-safe value with stable treatment of signed zero and floats."""

    if isinstance(value, float):
        return normalize_float(value)
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalize_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def value_token(value: Any, value_type: str | None = None) -> str:
    """A total, deterministic token used for row sorting and hashing."""

    if value is None:
        return "n"
    if value_type == "float" or isinstance(value, float):
        return "f:" + float(value).hex()
    return "i:" + str(int(value))


def row_tokens(row: Sequence[Any], column_types: Sequence[str]) -> tuple[str, ...]:
    return tuple(value_token(v, t) for v, t in zip(row, column_types))


def row_id(table_name: str, tokens: Sequence[str], occurrence: int) -> str:
    material = canonical_json(
        {"table": table_name, "values": list(tokens), "occurrence": occurrence}
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def canonical_rows(
    table_name: str,
    rows: Iterable[Sequence[Any]],
    column_types: Sequence[str],
) -> list[dict[str, Any]]:
    """Sort rows independently of source order and assign stable row identifiers."""

    token_rows = [(row_tokens(row, column_types), list(row)) for row in rows]
    token_rows.sort(key=lambda pair: pair[0])
    occurrence: dict[tuple[str, ...], int] = {}
    result: list[dict[str, Any]] = []
    for tokens, values in token_rows:
        seen = occurrence.get(tokens, 0)
        occurrence[tokens] = seen + 1
        rid = row_id(table_name, tokens, seen)
        result.append({"__rid": rid, "__tokens": tokens, "values": values})
    return result


def fixed_floor(numerator: int, denominator: int, scale: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator * scale) // denominator


def fixed_text(fixed: int, scale: int) -> str:
    sign = "-" if fixed < 0 else ""
    absolute = abs(fixed)
    whole, fraction = divmod(absolute, scale)
    return f"{sign}{whole}.{fraction:0{len(str(scale)) - 1}d}"


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

