from __future__ import annotations

from typing import Any

from .errors import BusinessError


def classification_record(issue: BusinessError) -> dict[str, Any]:
    location = issue.location
    return {
        "code": str(issue.code),
        "injection_id": issue.injection_id,
        "location": {
            "layer_id": location.layer_id,
            "path_id": location.path_id,
            "subpath_index": location.subpath_index,
            "segment_index": location.segment_index,
            "field": location.field,
        },
        "recovery": issue.recovery,
    }


def classification_records(issues: list[BusinessError]) -> list[dict[str, Any]]:
    return [classification_record(issue) for issue in issues]


def classification_key(record: dict[str, Any]) -> tuple:
    location = record.get("location", {})
    return (
        record.get("code"),
        record.get("injection_id"),
        location.get("layer_id"),
        location.get("path_id"),
        location.get("subpath_index"),
        location.get("segment_index"),
        location.get("field"),
        record.get("recovery"),
    )


def classification_keys(records: list[dict[str, Any]]) -> list[tuple]:
    return [classification_key(record) for record in records]
