"""JSON configuration loading with strict, whole-document validation.

A rejected document raises a single :class:`BusinessError` (``E_CONFIG``) that
carries every offending field path, so callers can report all problems at once
and never run with a partially accepted configuration.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .errors import BusinessError, ErrorCode

READ_WRITE = "READ_WRITE"
READ_ONLY = "READ_ONLY"
CONN_TYPES = (READ_WRITE, READ_ONLY)
_CONN_ALIASES = {
    "READ_WRITE": READ_WRITE,
    "READ_ONLY": READ_ONLY,
    "RW": READ_WRITE,
    "RO": READ_ONLY,
}

_FORBIDDEN_SQL_KEYWORDS = {
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ALTER", "DROP",
    "CREATE", "DELETE", "REPLACE", "TRIGGER", "VIEW", "SAVEPOINT",
    "RELEASE", "ROLLBACK", "COMMIT", "BEGIN", "TRANSACTION",
}
_TABLE_RE = re.compile(
    r"(?:\bFROM|\bJOIN|\bINTO|\bUPDATE)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PoolConfig:
    max_connections: int
    workload_quotas: dict[str, int]
    hold_timeout_ms: int
    acquire_timeout_ms: int
    param_version: int = 1


@dataclass(frozen=True)
class Task:
    task_id: str
    workload: str
    conn_type: str
    estimated_ms: int
    sql: str


@dataclass(frozen=True)
class WorkbenchConfig:
    pool: PoolConfig
    tasks: list[Task]
    allowed_tables: tuple[str, ...]
    sample_data_version: str = "1"


def _is_positive_int(value: Any) -> bool:
    # bool is a subclass of int; it must not be accepted as a number.
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _normalize_conn_type(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    return _CONN_ALIASES.get(raw.strip().upper())


def validate_sql(sql: Any, allowed_tables: Sequence[str]) -> tuple[str | None, str | None, list[str]]:
    """Return (normalized statement kind, error message, referenced tables)."""
    if not isinstance(sql, str) or not sql.strip():
        return None, "sql must be a non-empty string", []
    text = sql.strip().rstrip(";").strip()
    if ";" in text:
        return None, "multiple statements are not allowed", []
    keyword_match = re.match(r"[A-Za-z_]+", text)
    if keyword_match is None:
        return None, "statement must start with SELECT/INSERT/UPDATE", []
    kind = keyword_match.group(0).upper()
    if kind not in ("SELECT", "INSERT", "UPDATE"):
        return None, f"only SELECT/INSERT/UPDATE are allowed, got {kind}", []
    upper_tokens = set(re.findall(r"[A-Za-z_]+", text.upper()))
    forbidden = sorted(upper_tokens & _FORBIDDEN_SQL_KEYWORDS)
    if forbidden:
        return None, f"forbidden SQL keyword(s): {', '.join(forbidden)}", []
    tables = _TABLE_RE.findall(text)
    if not tables:
        return None, "could not resolve the target sample table", []
    allowed = set(allowed_tables)
    unknown = [table for table in tables if table not in allowed]
    if unknown:
        return None, f"table(s) not in sample allow-list: {', '.join(sorted(set(unknown)))}", []
    return kind, None, tables


def _validate_pool(raw: Any, errors: list[tuple[str, str]]) -> PoolConfig | None:
    if not isinstance(raw, dict):
        errors.append(("pool", "pool section must be an object"))
        return None
    max_connections = raw.get("max_connections")
    if not _is_positive_int(max_connections):
        errors.append(("pool.max_connections", "must be a positive integer"))
    hold_timeout_ms = raw.get("hold_timeout_ms")
    if not _is_positive_int(hold_timeout_ms):
        errors.append(("pool.hold_timeout_ms", "must be a positive integer"))
    acquire_timeout_ms = raw.get("acquire_timeout_ms")
    if not _is_positive_int(acquire_timeout_ms):
        errors.append(("pool.acquire_timeout_ms", "must be a positive integer"))
    param_version = raw.get("param_version", 1)
    if not _is_positive_int(param_version):
        errors.append(("pool.param_version", "must be a positive integer"))
    quotas_raw = raw.get("workload_quotas")
    quotas: dict[str, int] = {}
    if not isinstance(quotas_raw, dict) or not quotas_raw:
        errors.append(("pool.workload_quotas", "must be a non-empty object of workload -> quota"))
    else:
        for name, quota in quotas_raw.items():
            path = f"pool.workload_quotas.{name}"
            if not isinstance(name, str) or not name.strip():
                errors.append(("pool.workload_quotas", "workload names must be non-empty strings"))
            elif not _is_positive_int(quota):
                errors.append((path, "quota must be a positive integer"))
            else:
                quotas[name] = quota
        if quotas and sum(quotas.values()) > (max_connections or 0):
            errors.append((
                "pool.workload_quotas.sum",
                f"sum of quotas ({sum(quotas.values())}) exceeds max_connections "
                f"({max_connections})",
            ))
    if errors:
        return None
    return PoolConfig(
        max_connections=max_connections,
        workload_quotas=quotas,
        hold_timeout_ms=hold_timeout_ms,
        acquire_timeout_ms=acquire_timeout_ms,
        param_version=param_version,
    )


def _validate_tasks(
    raw: Any,
    pool: PoolConfig | None,
    allowed_tables: Sequence[str],
    errors: list[tuple[str, str]],
) -> list[Task]:
    if not isinstance(raw, list) or not raw:
        errors.append(("tasks", "must be a non-empty list"))
        return []
    tasks: list[Task] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        prefix = f"tasks[{index}]"
        if not isinstance(item, dict):
            errors.append((prefix, "task must be an object"))
            continue
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            errors.append((f"{prefix}.task_id", "must be a non-empty string"))
        elif task_id in seen_ids:
            errors.append((f"{prefix}.task_id", f"duplicate task_id {task_id!r}"))
        else:
            seen_ids.add(task_id)
        workload = item.get("workload")
        if not isinstance(workload, str) or not workload.strip():
            errors.append((f"{prefix}.workload", "must be a non-empty string"))
        elif pool is not None and workload not in pool.workload_quotas:
            errors.append((
                f"{prefix}.workload",
                f"workload {workload!r} has no quota in pool.workload_quotas",
            ))
        conn_type = _normalize_conn_type(item.get("conn_type"))
        if conn_type is None:
            errors.append((f"{prefix}.conn_type", "must be READ_WRITE or READ_ONLY"))
        estimated_ms = item.get("estimated_ms")
        if not _is_positive_int(estimated_ms):
            errors.append((f"{prefix}.estimated_ms", "must be a positive integer"))
        kind, sql_error, _tables = validate_sql(item.get("sql"), allowed_tables)
        if sql_error is not None:
            errors.append((f"{prefix}.sql", sql_error))
        elif kind in ("INSERT", "UPDATE") and conn_type == READ_ONLY:
            errors.append((f"{prefix}.conn_type", f"{kind} requires a READ_WRITE connection"))
        if errors and any(path.startswith(prefix) for path, _ in errors):
            continue
        tasks.append(Task(
            task_id=task_id,
            workload=workload,
            conn_type=conn_type,  # type: ignore[arg-type]
            estimated_ms=estimated_ms,
            sql=item["sql"].strip().rstrip(";").strip(),
        ))
    return tasks


def config_from_dict(data: Any, allowed_tables: Sequence[str]) -> WorkbenchConfig:
    errors: list[tuple[str, str]] = []
    if not isinstance(data, dict):
        raise BusinessError(
            ErrorCode.E_CONFIG,
            "configuration root must be a JSON object",
            field_paths=["<root>"],
        )
    pool = _validate_pool(data.get("pool"), errors)
    tasks = _validate_tasks(data.get("tasks"), pool, allowed_tables, errors)
    sample_data_version = data.get("sample_data_version", "1")
    if not isinstance(sample_data_version, str) or not sample_data_version.strip():
        errors.append(("sample_data_version", "must be a non-empty string"))
    if errors:
        raise BusinessError(
            ErrorCode.E_CONFIG,
            "configuration validation failed",
            field_paths=[path for path, _ in errors],
            details={"reasons": {path: reason for path, reason in errors}},
        )
    return WorkbenchConfig(
        pool=pool,  # type: ignore[arg-type]
        tasks=tasks,
        allowed_tables=tuple(allowed_tables),
        sample_data_version=sample_data_version,
    )


def load_config(path: str | Path, allowed_tables: Sequence[str]) -> WorkbenchConfig:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BusinessError(
            ErrorCode.E_CONFIG, f"config file not found: {path}", [str(path)]
        ) from exc
    except json.JSONDecodeError as exc:
        raise BusinessError(
            ErrorCode.E_CONFIG,
            f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}",
            [f"<json>:{exc.lineno}:{exc.colno}"],
        ) from exc
    return config_from_dict(data, allowed_tables)
