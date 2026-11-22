"""Stable error records shared by validation, execution and reporting."""

from __future__ import annotations

from typing import Any


def make_failure(
    error_code: str,
    field: str,
    message: str,
    *,
    tensor_id: str | None = None,
    column: int | None = None,
    bit_width: int | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    return {
        "error_code": error_code,
        "field": field,
        "message": message,
        "tensor_id": tensor_id,
        "column": column,
        "bit_width": bit_width,
        "stage": stage,
    }


def failure_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    column = item.get("column")
    width = item.get("bit_width")
    return (
        str(item.get("tensor_id") or ""),
        -1 if column is None else int(column),
        -1 if width is None else int(width),
        str(item.get("field") or ""),
        str(item.get("error_code") or ""),
        str(item.get("stage") or ""),
        str(item.get("message") or ""),
    )


def sort_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort and de-duplicate failures without depending on execution order."""

    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in failures:
        key = (
            item.get("error_code"),
            item.get("field"),
            item.get("tensor_id"),
            item.get("column"),
            item.get("bit_width"),
            item.get("stage"),
            item.get("message"),
        )
        unique.setdefault(key, item)
    return sorted(unique.values(), key=failure_sort_key)
