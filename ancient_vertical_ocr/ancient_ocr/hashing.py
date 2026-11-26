"""确定性指纹与缓存键派生（blake2b，规范 JSON，无随机盐）。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    else:
        raw = canonical_json(payload).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def cache_key(stage: str, *parts: Any) -> str:
    material = [stage]
    for part in parts:
        if isinstance(part, (bytes, bytearray)):
            material.append(hashlib.blake2b(bytes(part), digest_size=8).hexdigest())
        else:
            material.append(str(part))
    return f"{stage}:{digest(material)}"


def stable_unit_key(filename: str, stage: str, index: int | str) -> str:
    return f"{filename}|{stage}|{index}"


def stable_sorted(keys: Iterable[str]) -> list[str]:
    return sorted(keys)
