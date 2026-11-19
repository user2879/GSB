"""Canonical serialization and file helpers used for determinism."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import AppError


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AppError("E_DATA", f"值无法规范化为确定性 JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_text(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise AppError("E_DATA", f"值无法规范化为确定性 JSON: {exc}") from exc
    return canonical_json_bytes(value).decode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"不允许非有限数值 {value}")


def loads_strict(text: str, *, source: str = "<memory>") -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise AppError("E_DATA", f"JSON 解析失败: {source}: {exc}", {"source": source}) from exc


def load_json_file(path: str | os.PathLike[str]) -> Any:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise AppError("E_DATA", f"无法读取 JSON 文件: {p}: {exc}", {"path": str(p)}) from exc
    return loads_strict(text, source=str(p))


def write_json_atomic(path: str | os.PathLike[str], value: Any, *, pretty: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_text(value, pretty=pretty).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
