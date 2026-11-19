"""Canonical JSON and digest helpers used by every persisted structure."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


class CanonicalError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise CanonicalError(f"non-finite JSON number is not allowed: {value}")


def loads_strict(text: str) -> Any:
    """Parse JSON while rejecting NaN/Infinity and duplicate object keys."""

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=pairs_hook, parse_constant=_reject_constant)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalError(f"value is not canonical-JSON encodable: {exc}") from exc


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )


def number_from_json_text(text: str) -> int | float:
    value = loads_strict(text)
    if not is_number(value):
        raise CanonicalError("JSON text is not a finite number")
    return value


def pk_sort_key(pk: Any) -> tuple[int, Any]:
    # Numeric primary keys sort numerically; string keys sort lexically.
    if is_number(pk):
        return (0, pk)
    if isinstance(pk, str):
        return (1, pk)
    raise CanonicalError("primary key must be a finite number or a string")


def entry_sort_key(value: int | float, pk: Any) -> tuple[Any, ...]:
    return (value, *pk_sort_key(pk), canonical_json(pk))
