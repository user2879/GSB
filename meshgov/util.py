from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import RuleError


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def mono_ms() -> float:
    return time.monotonic_ns() / 1_000_000.0


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def json_pretty_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuleError("duplicate JSON object key", f"$.{key}")
        result[key] = value
    return result


def load_json_strict(raw: bytes, source: str) -> Any:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuleError(
            "JSON file must be UTF-8 encoded",
            "$",
            {"source": source, "reason": str(exc)},
        ) from exc
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except RuleError:
        raise
    except json.JSONDecodeError as exc:
        raise RuleError(
            "malformed JSON",
            f"$[line={exc.lineno},column={exc.colno}]",
            {"source": source, "reason": exc.msg},
        ) from exc


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    read_only: bool = False,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json_pretty_bytes(value)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        if read_only:
            try:
                target.chmod(0o444)
            except OSError:
                pass
    finally:
        if os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    return sha256_bytes(payload)


def require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuleError("expected a JSON object", path)
    return value


def require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuleError("expected a JSON array", path)
    return value


def require_string(value: Any, path: str, *, non_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise RuleError("expected a string", path)
    if non_empty and not value.strip():
        raise RuleError("string must not be empty", path)
    return value


def require_int(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuleError("expected an integer", path)
    if minimum is not None and value < minimum:
        raise RuleError(f"must be >= {minimum}", path)
    if maximum is not None and value > maximum:
        raise RuleError(f"must be <= {maximum}", path)
    return value
