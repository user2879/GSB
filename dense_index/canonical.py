from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_dumps(value: Any) -> str:
    """Serialize data in one cross-process, sort-key canonical form."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value)!r} is not canonically serializable")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value: Any, length: int = 16) -> str:
    digest = sha256_text(canonical_dumps(value))
    return digest[:length]


def full_fingerprint(value: Any) -> str:
    return "sha256:" + sha256_text(canonical_dumps(value))


def cache_key(*parts: Any) -> str:
    return "ck_" + fingerprint(list(parts), 24)


def stable_seed(value: Any) -> int:
    return int(fingerprint(value, 16), 16)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
