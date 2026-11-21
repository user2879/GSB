from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from typing import Any, Iterable


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def labelled_digest(label: str, value: Any) -> str:
    return hashlib.sha256((label + "\x00" + canonical_json(value)).encode("utf-8")).hexdigest()


def ordered_set_digest(values: Iterable[str]) -> str:
    ordered = sorted(set(values))
    return hashlib.sha256(("\n".join(ordered)).encode("utf-8")).hexdigest()
