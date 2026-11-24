"""确定性 JSON、哈希与整数定点工具。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

FIXED_SCALE = 1_000_000


def canonical_json(obj: Any) -> str:
    """固定键序、固定分隔符的 JSON 文本，用于哈希与快照。"""

    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_default
    )


def _default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"不可序列化的对象类型: {type(obj)!r}")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def short_ref(obj: Any) -> str:
    return sha256_hex(obj)[:16]


def file_fingerprint(path: str) -> dict:
    with open(path, "rb") as handle:
        data = handle.read()
    return {
        "path": os.path.basename(path),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fixed_div(numerator: int, denominator: int) -> int:
    """整数定点除法，向零截断，避免浮点参与资源指标。"""

    if denominator == 0:
        raise ZeroDivisionError("定点除法分母为零")
    quotient, remainder = divmod(abs(numerator), abs(denominator))
    result = quotient * FIXED_SCALE + remainder * FIXED_SCALE // abs(denominator)
    return result if (numerator < 0) == (denominator < 0) else -result


def atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
