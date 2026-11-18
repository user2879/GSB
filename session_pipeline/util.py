from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


class PipelineValidationError(ValueError):
    """A deterministic, user-facing validation failure."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def public_event(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if not key.startswith("_")}


def event_content_hash(event: dict[str, Any]) -> str:
    return stable_hash(public_event(event))


def slot_event_hash(event: dict[str, Any], update: dict[str, Any]) -> str:
    return stable_hash(
        {
            "session_id": event.get("session_id"),
            "turn": event.get("turn"),
            "role": event.get("role"),
            "slot_update": update,
        }
    )


def transition_event_hash(event: dict[str, Any]) -> str:
    return stable_hash(
        {
            "session_id": event.get("session_id"),
            "turn": event.get("turn"),
            "role": event.get("role"),
            "text": event.get("text", ""),
            "requested_target": event.get("requested_target")
            or event.get("transition_target"),
            "slot_updates": event.get("slot_updates", []),
        }
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_raw(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    {
                        "code": "E_INPUT_INVALID",
                        "phase": "loading",
                        "session_id": None,
                        "turn": None,
                        "line": line_number,
                        "exception_type": type(exc).__name__,
                        "message": f"invalid JSONL line: {exc.msg}",
                    }
                )
                continue
            if not isinstance(value, dict):
                failures.append(
                    {
                        "code": "E_INPUT_INVALID",
                        "phase": "loading",
                        "session_id": None,
                        "turn": None,
                        "line": line_number,
                        "exception_type": "TypeError",
                        "message": "JSONL entry must be an object",
                    }
                )
                continue
            value["_line"] = line_number
            events.append(value)
    return events, failures


def parse_decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise PipelineValidationError(f"{field} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PipelineValidationError(f"{field} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise PipelineValidationError(f"{field} must be a finite decimal")
    return parsed


def quantize_decimal(value: Decimal, places: int) -> Decimal:
    exponent = Decimal(1).scaleb(-places)
    return value.quantize(exponent, rounding=ROUND_HALF_UP)


def decimal_string(value: Decimal, places: int | None = None) -> str:
    if places is not None:
        value = quantize_decimal(value, places)
        return format(value, f".{places}f")
    normalized = value.normalize()
    if normalized == 0:
        return "0"
    return format(normalized, "f")


def fixed_to_string(fixed: int, decimals: int) -> str:
    scale = 10**decimals
    return format(Decimal(fixed) / Decimal(scale), f".{decimals}f")


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


DEFAULT_BUSINESS_CODES = {
    "input_invalid": "E_INPUT_INVALID",
    "transition_rejected": "E_TRANSITION_REJECTED",
    "slot_value_invalid": "E_SLOT_VALUE_INVALID",
    "session_quota_exceeded": "E_SESSION_QUOTA_EXCEEDED",
    "state_budget_exhausted": "E_STATE_BUDGET_EXHAUSTED",
    "internal_error": "E_INTERNAL_ERROR",
}


def business_code(config: dict[str, Any], key: str) -> str:
    return config.get("business_codes", {}).get(key, DEFAULT_BUSINESS_CODES.get(key, key.upper()))

