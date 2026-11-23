"""Session snapshot creation, integrity checks and stable difference reports."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from .canonical import atomic_write_json, content_digest, with_content_digest
from .errors import E_SNAPSHOT, AppError


SNAPSHOT_FORMAT_VERSION = 1


def create_snapshot_document(engine: Any, active_query: str | None = None) -> dict[str, Any]:
    state = engine.snapshot_state(active_query)
    versions = state["versions"]
    document = {
        "snapshot_format": SNAPSHOT_FORMAT_VERSION,
        "created_mono_ns": time.monotonic_ns(),
        "workspace_id": state["workspace_id"],
        "workspace_name": state["workspace_name"],
        "active_query": state["active_query"],
        "tables_version": int(versions["tables"]),
        "sql_version": int(versions["sql"]),
        "edits_version": int(versions["edits"]),
        "scale_version": int(versions["scale"]),
        "parameter_version": int(state["views"][state["active_query"]]["parameter_version"]),
        "epoch": int(state["epoch"]),
        "table_digest": state["table_digest"],
        "sql_digest": state["sql_digest"],
        "scale_digest": state["scale_digest"],
        "logical_plan_summary": state["logical_plan_summary"],
        "ledger_cursor": int(state["ledger_cursor"]),
        "views": state["views"],
    }
    return with_content_digest(document)


def create_snapshot(
    engine: Any,
    active_query: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    document = create_snapshot_document(engine, active_query)
    if path is None:
        digest_text = document["content_digest"][:16]
        target = engine.snapshot_dir / f"snapshot-{int(document['ledger_cursor']):08d}-{digest_text}.json"
    else:
        target = Path(path)
    atomic_write_json(target, document)
    return {
        "ok": True,
        "path": str(target),
        "content_digest": document["content_digest"],
        "ledger_cursor": document["ledger_cursor"],
        "logical_plan_summary": document["logical_plan_summary"],
        "document": document,
    }


def read_snapshot(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise AppError(E_SNAPSHOT, f"cannot read snapshot: {error}", details={"differences": [{"field": "path"}]}) from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise AppError(
            E_SNAPSHOT,
            f"invalid snapshot JSON: {error.msg}",
            details={"differences": [{"field": "json", "snapshot": f"@{error.pos}", "current": "parseable"}]},
        ) from error
    if not isinstance(document, dict):
        raise AppError(
            E_SNAPSHOT,
            "snapshot root must be an object",
            details={"differences": [{"field": "root", "snapshot": type(document).__name__, "current": "object"}]},
        )
    return document


def snapshot_differences(document: Mapping[str, Any], current: Mapping[str, Any]) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    stored_digest = document.get("content_digest")
    computed_digest = content_digest(dict(document))
    if stored_digest != computed_digest:
        differences.append(
            {
                "field": "content_digest",
                "snapshot": stored_digest,
                "current": computed_digest,
            }
        )

    scalar_fields = (
        "snapshot_format",
        "workspace_id",
        "active_query",
        "tables_version",
        "sql_version",
        "edits_version",
        "scale_version",
        "parameter_version",
        "epoch",
        "table_digest",
        "sql_digest",
        "scale_digest",
        "logical_plan_summary",
        "ledger_cursor",
    )
    for field in scalar_fields:
        if field == "snapshot_format":
            if document.get(field) != SNAPSHOT_FORMAT_VERSION:
                differences.append(
                    {"field": field, "snapshot": document.get(field), "current": SNAPSHOT_FORMAT_VERSION}
                )
            continue
        if document.get(field) != current.get(field):
            differences.append(
                {
                    "field": field,
                    "snapshot": document.get(field),
                    "current": current.get(field),
                }
            )

    snapshot_views = document.get("views", {})
    current_views = current.get("views", {})
    if not isinstance(snapshot_views, dict):
        differences.append({"field": "views", "snapshot": type(snapshot_views).__name__, "current": "object"})
        snapshot_views = {}
    for query_name in sorted(set(snapshot_views) | set(current_views), key=lambda item: item.encode("utf-8")):
        snapshot_view = snapshot_views.get(query_name, {})
        current_view = current_views.get(query_name, {})
        for field in (
            "sql",
            "parameter_version",
            "epoch",
            "edit_seq",
            "edits_version",
            "plan_summary",
            "ledger_cursor",
        ):
            if snapshot_view.get(field) != current_view.get(field):
                differences.append(
                    {
                        "field": f"views.{query_name}.{field}",
                        "snapshot": snapshot_view.get(field),
                        "current": current_view.get(field),
                    }
                )
    differences.sort(key=lambda entry: str(entry.get("field", "")).encode("utf-8"))
    return differences
