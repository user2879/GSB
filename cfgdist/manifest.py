"""Encode/decode and strict validation for manifest format versions v1..v3.

Format summary
--------------
v1: {"format_version": 1, "entries": [{"key", "value"}]}
v2: per-entry "node" and "config_version" are added.
v3: per-entry "strategy" and top-level "derivations" are added.

Any structural problem rejects the *whole* document with E_FORMAT and a
JSON-pointer-like field path plus a human-readable reason.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import BizError, ErrorCode

OBJECT_MARKER = "__raw_object__"
SUPPORTED_VALUE_TYPES = (int, float, str, bool)
SUPPORTED_OPS = ("sum", "mean", "min", "max", "count")
DEFAULT_STRATEGY = {
    "mode": "targeted",
    "retries": 2,
    "timeout_ms": 2000,
}

_TOP_LEVEL_KEYS = {
    1: {"format_version", "entries"},
    2: {"format_version", "entries"},
    3: {"format_version", "entries", "derivations"},
}
_ENTRY_KEYS = {
    1: {"key", "value"},
    2: {"key", "value", "node", "config_version"},
    3: {"key", "value", "node", "config_version", "strategy"},
}


def format_error(path: str, reason: str) -> BizError:
    return BizError(
        ErrorCode.E_FORMAT,
        f"manifest validation failed at {path}: {reason}",
        {"field_path": path, "reason": reason},
    )


def _pairs_hook(pairs):
    # Preserve duplicate keys and object/array distinction for later walking.
    return (OBJECT_MARKER, pairs)


def _walk(node: Any, path: str) -> Any:
    if isinstance(node, tuple) and len(node) == 2 and node[0] == OBJECT_MARKER:
        result = {}
        seen: set[str] = set()
        for key, value in node[1]:
            if not isinstance(key, str):
                raise format_error(path, f"object key must be string, got {type(key).__name__}")
            child_path = f"{path}.{key}" if path else key
            if key in seen:
                raise format_error(child_path, "duplicate key")
            seen.add(key)
            result[key] = _walk(value, child_path)
        return result
    if isinstance(node, list):
        return [_walk(item, f"{path}[{index}]") for index, item in enumerate(node)]
    return node


def parse_manifest(text: str, source: str = "<memory>") -> dict:
    try:
        raw = json.loads(text, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as exc:
        raise format_error(
            f"$[{exc.lineno}:{exc.colno}]",
            f"invalid JSON syntax: {exc.msg}",
        ) from exc
    manifest = _walk(raw, "$")
    return validate_manifest(manifest, source)


def load_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise format_error(str(manifest_path), f"cannot read manifest: {exc}") from exc
    manifest = parse_manifest(text, str(manifest_path))
    manifest["__source__"] = str(manifest_path)
    return manifest


def load_json_with_duplicate_check(text: str, source: str = "<memory>") -> dict:
    """Parse JSON and reject duplicate keys, without version-specific rules.

    Used for intermediate working documents whose entries can legitimately
    contain fields from a newer format while the document is in progress.
    """
    try:
        raw = json.loads(text, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as exc:
        raise format_error(
            f"$[{exc.lineno}:{exc.colno}]",
            f"invalid JSON syntax: {exc.msg}",
        ) from exc
    return _walk(raw, "$")


def _ensure(value: Any, path: str, expected: type | tuple, type_name: str):
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise format_error(path, f"expected {type_name}, got {type(value).__name__}")


def _validate_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (dict, list, tuple)):
        raise format_error(
            path,
            f"unsupported value type {type(value).__name__}; "
            "only int/float/str/bool scalars are allowed",
        )
    if not isinstance(value, SUPPORTED_VALUE_TYPES):
        raise format_error(path, f"unsupported value type {type(value).__name__}")


def _validate_strategy(strategy: Any, path: str) -> None:
    if not isinstance(strategy, dict):
        raise format_error(path, f"expected object, got {type(strategy).__name__}")
    allowed = {"mode", "retries", "timeout_ms"}
    for key in strategy:
        if key not in allowed:
            raise format_error(f"{path}.{key}", "unknown strategy field")
    mode = strategy.get("mode")
    if mode not in ("broadcast", "targeted"):
        raise format_error(f"{path}.mode", "must be 'broadcast' or 'targeted'")
    retries = strategy.get("retries")
    _ensure(retries, f"{path}.retries", int, "integer")
    if retries < 0:
        raise format_error(f"{path}.retries", "must be >= 0")
    timeout_ms = strategy.get("timeout_ms")
    _ensure(timeout_ms, f"{path}.timeout_ms", int, "integer")
    if timeout_ms <= 0:
        raise format_error(f"{path}.timeout_ms", "must be > 0")


def validate_manifest(manifest: Any, source: str = "<memory>") -> dict:
    if not isinstance(manifest, dict):
        raise format_error("$", f"top-level must be object, got {type(manifest).__name__}")

    if "format_version" not in manifest:
        raise format_error("$.format_version", "required field missing")
    version = manifest["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise format_error("$.format_version", f"expected integer, got {type(version).__name__}")
    if version not in _TOP_LEVEL_KEYS:
        raise format_error(
            "$.format_version",
            f"unknown format version {version}; supported versions are 1, 2, 3",
        )

    for key in manifest:
        if key == "__source__":
            continue
        if key not in _TOP_LEVEL_KEYS[version]:
            raise format_error(f"$.{key}", f"field not allowed in format v{version}")

    if "entries" not in manifest:
        raise format_error("$.entries", "required field missing")
    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise format_error("$.entries", "must be a non-empty array")

    allowed_entry_keys = _ENTRY_KEYS[version]
    seen_keys: set[str] = set()
    for index, entry in enumerate(entries):
        entry_path = f"$.entries[{index}]"
        if not isinstance(entry, dict):
            raise format_error(entry_path, f"expected object, got {type(entry).__name__}")
        for field in entry:
            if field not in allowed_entry_keys:
                raise format_error(
                    f"{entry_path}.{field}",
                    f"field not allowed in format v{version}",
                )
        if "key" not in entry:
            raise format_error(f"{entry_path}.key", "required field missing")
        key = entry["key"]
        if not isinstance(key, str) or not key:
            raise format_error(f"{entry_path}.key", "must be a non-empty string")
        if key in seen_keys:
            raise format_error(f"{entry_path}.key", f"duplicate entry key '{key}'")
        seen_keys.add(key)
        if "value" not in entry:
            raise format_error(f"{entry_path}.value", "required field missing")
        _validate_value(entry["value"], f"{entry_path}.value")

        if version >= 2:
            if "node" not in entry:
                raise format_error(f"{entry_path}.node", "required field missing in v2+")
            if not isinstance(entry["node"], str) or not entry["node"]:
                raise format_error(f"{entry_path}.node", "must be a non-empty string")
            if "config_version" not in entry:
                raise format_error(
                    f"{entry_path}.config_version",
                    "required field missing in v2+",
                )
            cv = entry["config_version"]
            if isinstance(cv, bool) or not isinstance(cv, int):
                raise format_error(
                    f"{entry_path}.config_version",
                    f"expected integer, got {type(cv).__name__}",
                )
        if version >= 3 and "strategy" in entry:
            _validate_strategy(entry["strategy"], f"{entry_path}.strategy")

    if version >= 3:
        derivations = manifest.get("derivations", [])
        if not isinstance(derivations, list):
            raise format_error("$.derivations", "must be an array")
        known_keys = seen_keys
        seen_names: set[str] = set()
        for index, derivation in enumerate(derivations):
            dpath = f"$.derivations[{index}]"
            if not isinstance(derivation, dict):
                raise format_error(dpath, f"expected object, got {type(derivation).__name__}")
            for field in derivation:
                if field not in {"name", "inputs", "op", "target_node"}:
                    raise format_error(f"{dpath}.{field}", "unknown derivation field")
            name = derivation.get("name")
            if not isinstance(name, str) or not name:
                raise format_error(f"{dpath}.name", "must be a non-empty string")
            if name in seen_names:
                raise format_error(f"{dpath}.name", f"duplicate derivation name '{name}'")
            seen_names.add(name)
            inputs = derivation.get("inputs")
            if not isinstance(inputs, list) or not inputs:
                raise format_error(f"{dpath}.inputs", "must be a non-empty array of strings")
            for input_index, input_key in enumerate(inputs):
                if not isinstance(input_key, str) or not input_key:
                    raise format_error(f"{dpath}.inputs[{input_index}]", "must be a non-empty string")
                if input_key not in known_keys:
                    raise format_error(
                        f"{dpath}.inputs[{input_index}]",
                        f"input '{input_key}' does not reference any entry key",
                    )
            op = derivation.get("op")
            if op not in SUPPORTED_OPS:
                raise format_error(f"{dpath}.op", f"unsupported op '{op}'")
            target_node = derivation.get("target_node")
            if not isinstance(target_node, str) or not target_node:
                raise format_error(f"{dpath}.target_node", "must be a non-empty string")

    manifest["format_version"] = version
    return manifest


def canonical_json(payload: Any) -> str:
    """Stable serialization used for digests and plan comparison."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest_to_json(manifest: dict) -> str:
    public = {key: value for key, value in manifest.items() if not key.startswith("__")}
    return json.dumps(public, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def atomic_write(path: str | Path, text: str, readonly: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    if readonly:
        os.chmod(target, 0o444)
    return target
