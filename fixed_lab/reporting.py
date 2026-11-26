"""JSON-safe report conversion."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "as_dict"):
        return to_jsonable(value.as_dict())
    return value


def comparable_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove run-local cache status while retaining stable lineage keys."""

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: strip(value)
                for key, value in node.items()
                if key not in {"cache_status", "status_run"}
            }
        if isinstance(node, list):
            return [strip(item) for item in node]
        return node

    return strip(report)
