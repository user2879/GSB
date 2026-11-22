"""Structured error and failure helpers."""

from __future__ import annotations

from typing import Any


E_QUERY_INVALID = "E_QUERY_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_RULE_INVALID = "E_RULE_INVALID"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_QUERY_EXEC = "E_QUERY_EXEC"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


def issue(
    code: str,
    message: str,
    *,
    query_id: str | None = None,
    command_id: str | None = None,
    stage: str | None = None,
    injection_id: str | None = None,
    detail: Any | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable, deterministically ordered issue record."""

    record: dict[str, Any] = {"code": code, "message": message}
    if query_id is not None:
        record["query_id"] = query_id
    if command_id is not None:
        record["command_id"] = command_id
    if stage is not None:
        record["stage"] = stage
    if injection_id is not None:
        record["injection_id"] = injection_id
    if detail is not None:
        record["detail"] = detail
    return record


def sort_issues(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable failure ordering independent of processing or thread schedules."""

    return sorted(
        items,
        key=lambda item: (
            str(item.get("query_id", "")),
            str(item.get("command_id", "")),
            str(item.get("code", "")),
            str(item.get("stage", "")),
            str(item.get("message", "")),
            str(item.get("injection_id", "")),
        ),
    )
