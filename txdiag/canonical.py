"""Canonical JSON, hashes, and deterministic structured comparison."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def structured_diff(left: Any, right: Any, path: str = "$", limit: int = 64) -> list[dict[str, Any]]:
    """Return stable path-oriented differences between two JSON values."""
    out: list[dict[str, Any]] = []

    def walk(a: Any, b: Any, where: str) -> None:
        if len(out) >= limit:
            return
        if type(a) is not type(b):
            out.append({"path": where, "left": a, "right": b})
            return
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                child = f"{where}.{key}"
                if key not in a:
                    out.append({"path": child, "left": "<missing>", "right": b[key]})
                elif key not in b:
                    out.append({"path": child, "left": a[key], "right": "<missing>"})
                else:
                    walk(a[key], b[key], child)
                if len(out) >= limit:
                    return
            return
        if isinstance(a, list):
            if len(a) != len(b):
                out.append({"path": f"{where}.length", "left": len(a), "right": len(b)})
            for idx, (av, bv) in enumerate(zip(a, b)):
                walk(av, bv, f"{where}[{idx}]")
                if len(out) >= limit:
                    return
            return
        if a != b:
            out.append({"path": where, "left": a, "right": b})

    walk(left, right, path)
    return out


def first_diff_location(diffs: Iterable[dict[str, Any]]) -> tuple[str | None, str | None, str]:
    for item in diffs:
        path = str(item["path"])
        txid = _extract_after(path, "txid")
        row = _extract_after(path, "row")
        return txid, row, path
    return None, None, "$"


def _extract_after(path: str, key: str) -> str | None:
    marker = f".{key}="
    if marker in path:
        value = path.split(marker, 1)[1]
        for sep in (".", "[", "]"):
            value = value.split(sep, 1)[0]
        return value or None
    return None

