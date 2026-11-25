"""Strict JSON input and canonical serialization helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import DiagnosticError


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def parse_json_text(text: str, *, object_name: str, code: str = "E_SPEC") -> Any:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_invalid_constant,
        )
    except _DuplicateKeyError as exc:
        raise DiagnosticError(
            code=code,
            message=f"duplicate JSON key {exc.key!r}",
            object=object_name,
            field=exc.key,
        ) from exc
    except json.JSONDecodeError as exc:
        raise DiagnosticError(
            code=code,
            message=f"invalid JSON: {exc.msg}",
            object=object_name,
            field="<json>",
            line=exc.lineno,
            details={"column": exc.colno},
        ) from exc
    except ValueError as exc:
        raise DiagnosticError(
            code=code,
            message=str(exc),
            object=object_name,
            field="<json>",
        ) from exc
    return value


def load_json_file(
    path: str | Path,
    *,
    object_name: str | None = None,
    code: str = "E_SPEC",
) -> Any:
    source = Path(path)
    label = object_name or str(source)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise DiagnosticError(
            code=code,
            message=f"cannot read JSON file: {exc.strerror or exc}",
            object=label,
            field="<file>",
        ) from exc
    return parse_json_text(text, object_name=label, code=code)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_canonical(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))
