"""内容指纹工具：所有指纹都基于规范化 JSON 的 sha256，保证可复现。"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def fingerprint_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def short_key(*parts: Any) -> str:
    return fingerprint(list(parts))[:16]
