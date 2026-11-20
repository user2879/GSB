from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .models import (
    CURRENT_FORMAT_VERSION,
    DEFAULT_CONCURRENCY,
    DEFAULT_ISOLATION_LEVEL,
    DEFAULT_LOCK_WAIT_BUDGET_MS,
    DEFAULT_OPERATION_TIMEOUT_MS,
    EditScript,
    ISOLATION_LEVELS,
    Operation,
    SessionSpec,
)


_TOP_LEVEL_FIELDS = {
    "format_version",
    "script_id",
    "initial_rows",
    "description",
    "sessions",
    "source_format_version",
    "upgrade_changes",
}
_SESSION_FIELDS = {
    1: {"session_id", "operations", "lock_order"},
    2: {
        "session_id",
        "isolation_level",
        "lock_wait_budget_ms",
        "operations",
        "lock_order",
    },
    3: {
        "session_id",
        "isolation_level",
        "lock_wait_budget_ms",
        "concurrency",
        "operation_timeout_ms",
        "operations",
        "lock_order",
    },
}
_OPERATION_FIELDS = {"op_no", "type", "row_key", "value"}


def _error(
    message: str,
    path: str,
    session_id: str | None = None,
    op_no: int | None = None,
    details: dict[str, Any] | None = None,
) -> BusinessError:
    return BusinessError(
        "E_SCRIPT",
        message,
        path=path,
        session_id=session_id,
        op_no=op_no,
        details=details or {},
    )


def _require_object(value: Any, path: str, session_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error("expected an object", path, session_id)
    return value


def _is_int(value: Any) -> bool:
    return type(value) is int


def _require_int(value: Any, path: str, session_id: str | None, *, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise _error("expected an integer", path, session_id)
    if minimum is not None and value < minimum:
        raise _error(f"expected an integer >= {minimum}", path, session_id)
    return value


def _require_string(value: Any, path: str, session_id: str | None, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise _error("expected a string", path, session_id)
    if nonempty and not value.strip():
        raise _error("expected a non-empty string", path, session_id)
    return value


def _reject_unknown_fields(
    obj: dict[str, Any],
    allowed: set[str],
    path: str,
    session_id: str | None = None,
) -> None:
    for key in sorted(set(obj) - allowed):
        raise _error(f"unknown field: {key}", f"{path}.{key}", session_id)


def _validate_json_value(value: Any, path: str, session_id: str | None) -> None:
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise _error("non-finite numbers are not valid JSON values", path, session_id)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]", session_id)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _error("object keys must be strings", path, session_id)
            _validate_json_value(item, f"{path}.{key}", session_id)
        return
    raise _error(f"unsupported JSON value type: {type(value).__name__}", path, session_id)


def _derive_lock_order(operations: list[Operation]) -> tuple[str, ...]:
    seen: list[str] = []
    for operation in operations:
        if operation.row_key not in seen:
            seen.append(operation.row_key)
    return tuple(seen)


def _parse_operation(
    raw: Any,
    *,
    version: int,
    session_id: str,
    session_path: str,
    index: int,
) -> Operation:
    path = f"{session_path}.operations[{index}]"
    obj = _require_object(raw, path, session_id)
    _reject_unknown_fields(obj, _OPERATION_FIELDS, path, session_id)
    for field_name in ("op_no", "type", "row_key"):
        if field_name not in obj:
            raise _error(f"missing required field: {field_name}", f"{path}.{field_name}", session_id)

    op_no = _require_int(obj["op_no"], f"{path}.op_no", session_id, minimum=1)
    kind = _require_string(obj["type"], f"{path}.type", session_id)
    if kind not in ("read", "write"):
        raise _error("type must be 'read' or 'write'", f"{path}.type", session_id, op_no)
    row_key = _require_string(obj["row_key"], f"{path}.row_key", session_id, nonempty=True)

    if kind == "write":
        if "value" not in obj:
            raise _error("missing required field: value", f"{path}.value", session_id, op_no)
        _validate_json_value(obj["value"], f"{path}.value", session_id)
        return Operation(op_no, kind, row_key, obj["value"], True)

    if "value" in obj:
        raise _error("read operations must not declare a value", f"{path}.value", session_id, op_no)
    return Operation(op_no, kind, row_key, None, False)


def _parse_session(raw: Any, *, version: int, index: int) -> SessionSpec:
    path = f"$.sessions[{index}]"
    obj = _require_object(raw, path)
    sid = obj.get("session_id")
    if sid is None:
        raise _error("missing required field: session_id", f"{path}.session_id")
    sid = _require_string(sid, f"{path}.session_id", None)
    _reject_unknown_fields(obj, _SESSION_FIELDS[version], path, sid)

    if "operations" not in obj:
        raise _error("missing required field: operations", f"{path}.operations", sid)
    operations_raw = obj["operations"]
    if not isinstance(operations_raw, list) or not operations_raw:
        raise _error("expected a non-empty array", f"{path}.operations", sid)
    operations = [
        _parse_operation(
            operation,
            version=version,
            session_id=sid,
            session_path=path,
            index=operation_index,
        )
        for operation_index, operation in enumerate(operations_raw)
    ]
    operation_numbers = [operation.op_no for operation in operations]
    if len(set(operation_numbers)) != len(operation_numbers):
        raise _error("op_no values must be unique within a session", f"{path}.operations", sid)
    if operation_numbers != sorted(operation_numbers):
        raise _error("operations must be listed in ascending op_no order", f"{path}.operations", sid)

    derived_order = _derive_lock_order(operations)
    if "lock_order" in obj:
        order_raw = obj["lock_order"]
        if not isinstance(order_raw, list) or any(not isinstance(item, str) or not item for item in order_raw):
            raise _error("lock_order must be an array of non-empty strings", f"{path}.lock_order", sid)
        if len(set(order_raw)) != len(order_raw):
            raise _error("lock_order entries must be unique", f"{path}.lock_order", sid)
        if tuple(order_raw) != derived_order:
            raise _error(
                "lock_order must equal the first-use order of row keys in operations",
                f"{path}.lock_order",
                sid,
            )
        lock_order = tuple(order_raw)
    else:
        lock_order = derived_order

    isolation = DEFAULT_ISOLATION_LEVEL
    budget = DEFAULT_LOCK_WAIT_BUDGET_MS
    concurrency = DEFAULT_CONCURRENCY
    operation_timeout = DEFAULT_OPERATION_TIMEOUT_MS

    if version >= 2:
        for field_name in ("isolation_level", "lock_wait_budget_ms"):
            if field_name not in obj:
                raise _error(f"missing required field: {field_name}", f"{path}.{field_name}", sid)
        isolation = _require_string(obj["isolation_level"], f"{path}.isolation_level", sid)
        if isolation not in ISOLATION_LEVELS:
            raise _error(
                f"isolation_level must be one of {', '.join(ISOLATION_LEVELS)}",
                f"{path}.isolation_level",
                sid,
            )
        budget = _require_int(
            obj["lock_wait_budget_ms"],
            f"{path}.lock_wait_budget_ms",
            sid,
            minimum=0,
        )

    if version >= 3:
        for field_name in ("concurrency", "operation_timeout_ms"):
            if field_name not in obj:
                raise _error(f"missing required field: {field_name}", f"{path}.{field_name}", sid)
        concurrency = _require_int(obj["concurrency"], f"{path}.concurrency", sid, minimum=1)
        operation_timeout = _require_int(
            obj["operation_timeout_ms"],
            f"{path}.operation_timeout_ms",
            sid,
            minimum=1,
        )

    return SessionSpec(
        session_id=sid,
        operations=tuple(operations),
        isolation_level=isolation,
        lock_wait_budget_ms=budget,
        concurrency=concurrency,
        operation_timeout_ms=operation_timeout,
        lock_order=lock_order,
    )


def _upgrade_changes(
    document: dict[str, Any],
    version: int,
    source_version: int,
    sessions: tuple[SessionSpec, ...],
) -> tuple[str, ...]:
    if source_version >= CURRENT_FORMAT_VERSION:
        return ()
    changes = [f"format_version:{source_version}->{CURRENT_FORMAT_VERSION}"]
    for raw_session, session in zip(document["sessions"], sessions):
        sid = session.session_id
        if version == 1:
            changes.extend(
                [
                    f"{sid}.isolation_level=default:{DEFAULT_ISOLATION_LEVEL}",
                    f"{sid}.lock_wait_budget_ms=default:{DEFAULT_LOCK_WAIT_BUDGET_MS}",
                ]
            )
        if version <= 2:
            changes.extend(
                [
                    f"{sid}.concurrency=default:{DEFAULT_CONCURRENCY}",
                    f"{sid}.operation_timeout_ms=default:{DEFAULT_OPERATION_TIMEOUT_MS}",
                ]
            )
        if "lock_order" not in raw_session:
            changes.append(f"{sid}.lock_order=derived")
    return tuple(changes)


def normalize_document(document: Any) -> EditScript:
    obj = _require_object(document, "$")
    _reject_unknown_fields(obj, _TOP_LEVEL_FIELDS, "$")

    if "format_version" not in obj:
        raise _error("missing required field: format_version", "$.format_version")
    version = obj["format_version"]
    if not _is_int(version):
        raise _error("format_version must be an integer", "$.format_version")
    if version not in (1, 2, 3):
        raise _error(
            "unknown script format_version; supported versions are 1, 2 and 3",
            "$.format_version",
            details={"format_version": version},
        )

    for field_name in ("script_id", "initial_rows", "sessions"):
        if field_name not in obj:
            raise _error(f"missing required field: {field_name}", f"$.{field_name}")

    script_id = _require_string(obj["script_id"], "$.script_id", None)
    initial_rows_raw = _require_object(obj["initial_rows"], "$.initial_rows")
    initial_rows: dict[str, Any] = {}
    for key in sorted(initial_rows_raw):
        if not isinstance(key, str) or not key.strip():
            raise _error("initial_rows keys must be non-empty strings", f"$.initial_rows.{key}")
        _validate_json_value(initial_rows_raw[key], f"$.initial_rows.{key}", "<initial_rows>")
        initial_rows[key] = initial_rows_raw[key]

    sessions_raw = obj["sessions"]
    if not isinstance(sessions_raw, list) or not sessions_raw:
        raise _error("expected a non-empty array", "$.sessions")
    sessions = [_parse_session(raw, version=version, index=index) for index, raw in enumerate(sessions_raw)]
    session_ids = [session.session_id for session in sessions]
    if len(set(session_ids)) != len(session_ids):
        duplicate = next(sid for sid in session_ids if session_ids.count(sid) > 1)
        raise _error("session_id values must be unique", "$.sessions", duplicate)

    description = None
    if "description" in obj:
        description = _require_string(obj["description"], "$.description", None, nonempty=False)

    source_version = version
    if "source_format_version" in obj:
        source_version = _require_int(
            obj["source_format_version"],
            "$.source_format_version",
            None,
        )
        if source_version not in (1, 2, 3) or source_version > version:
            raise _error(
                "source_format_version must be one of 1, 2, 3 and not exceed format_version",
                "$.source_format_version",
            )

    supplied_changes: tuple[str, ...] | None = None
    if "upgrade_changes" in obj:
        raw_changes = obj["upgrade_changes"]
        if not isinstance(raw_changes, list) or any(not isinstance(item, str) for item in raw_changes):
            raise _error("upgrade_changes must be an array of strings", "$.upgrade_changes")
        supplied_changes = tuple(raw_changes)

    return EditScript(
        script_id=script_id,
        source_version=source_version,
        normalized_version=CURRENT_FORMAT_VERSION,
        initial_rows=initial_rows,
        sessions=tuple(sessions),
        description=description,
        upgrade_changes=(
            supplied_changes
            if supplied_changes is not None
            else _upgrade_changes(obj, version, source_version, tuple(sessions))
        ),
    )


def load_script(path: str | Path) -> EditScript:
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError(
            "E_SCRIPT",
            f"cannot read script file: {exc}",
            path="$",
            details={"file": str(source)},
        ) from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BusinessError(
            "E_SCRIPT",
            f"invalid JSON: {exc.msg}",
            path="$",
            details={"line": exc.lineno, "column": exc.colno, "file": str(source)},
        ) from exc
    return normalize_document(document)


def upgrade_document(document: Any) -> dict[str, Any]:
    return normalize_document(document).to_dict(include_source=True)
