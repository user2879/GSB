from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def normalize_json(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be fingerprinted")
        return 0.0 if value == 0.0 else value
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value: Any) -> str:
    return "sha256:" + sha256_text(canonical_json(value))


def short_fingerprint(value: Any, length: int = 16) -> str:
    return fingerprint(value).split(":", 1)[1][:length]
