from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any


def decimal_text(value: Decimal | str | int | float) -> str:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    if not value.is_finite():
        return str(value)
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def _normalise(value: Any) -> Any:
    if isinstance(value, Decimal):
        return decimal_text(value)
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return repr(value)
        return float(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalise(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(domain: str, value: Any) -> str:
    payload = domain.encode("utf-8") + b"\x00" + canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bytes_fingerprint(domain: str, payload: bytes) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def read_canonical_hash(path: Path) -> str:
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return bytes_fingerprint("file-bytes-v1", path.read_bytes())
        return fingerprint("json-file-v1", data)
    return bytes_fingerprint("file-bytes-v1", path.read_bytes())

