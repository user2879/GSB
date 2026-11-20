"""Isolated fault injection for demonstrations and tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .config import normalise_config
from .errors import E_CONFIG_INVALID, WorkbenchError, diagnostic
from .util import read_json


def load_injections(path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        return [], []
    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkbenchError(E_CONFIG_INVALID, f"cannot load injection plan: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("items", []), list):
        raise WorkbenchError(E_CONFIG_INVALID, "injection plan must contain an items list")
    items: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("items", [])):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            failures.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    "injection item requires a string id",
                    field=f"injections.items[{index}]",
                    stage="injection",
                )
            )
            continue
        items.append(deepcopy(item))
    return items, failures


def apply_raw_injections(
    drafts: list[dict[str, Any]], items: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {draft["object"].get("row_id"): draft for draft in drafts}
    hits: list[dict[str, Any]] = []
    for item in items:
        kind = item.get("kind")
        stage = item.get("stage", "load")
        if stage == "config":
            continue
        row_id = str(item.get("row_id", ""))
        column = item.get("column")
        draft = by_id.get(row_id)
        if draft is None or not isinstance(column, str):
            hits.append(
                {
                    "injection_id": item["id"],
                    "stage": stage,
                    "kind": kind,
                    "row_id": row_id or None,
                    "column": column if isinstance(column, str) else None,
                    "expected_business_code": item.get("business_code"),
                    "applied": False,
                    "matched": False,
                    "message": "injection target row or column was not found",
                }
            )
            continue
        if kind == "non_finite_value":
            draft["object"][column] = item.get("value", float("nan"))
            if kind == "non_finite_value" and item.get("value") is None:
                draft["object"][column] = float("nan")
        elif kind == "unmapped_group_value":
            draft["object"][column] = item.get("value", "__UNMAPPED_GROUP__")
        else:
            hits.append(
                {
                    "injection_id": item["id"],
                    "stage": stage,
                    "kind": kind,
                    "row_id": row_id,
                    "column": column,
                    "expected_business_code": item.get("business_code"),
                    "applied": False,
                    "matched": False,
                    "message": f"unsupported raw injection kind {kind!r}",
                }
            )
            continue
        hits.append(
            {
                "injection_id": item["id"],
                "stage": stage,
                "kind": kind,
                "row_id": row_id,
                "column": column,
                "expected_business_code": item.get("business_code"),
                "applied": True,
                "matched": False,
            }
        )
    return drafts, hits


def shadow_config_validate(
    config: dict[str, Any], items: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate config injections on a discarded shadow copy.

    The real pipeline keeps the valid base configuration, so one bad injected
    parameter cannot prevent unrelated valid rows from being processed.  Each hit
    is still proven by a real E_CONFIG_INVALID diagnostic from the normal validator.
    """

    hits: list[dict[str, Any]] = []
    shadow = deepcopy(config)
    for item in items:
        if item.get("stage") != "config":
            continue
        target = item.get("target")
        value = item.get("value")
        if not isinstance(target, str):
            hits.append(
                {
                    "injection_id": item["id"],
                    "stage": "config",
                    "kind": item.get("kind"),
                    "field": None,
                    "expected_business_code": item.get("business_code", E_CONFIG_INVALID),
                    "applied": False,
                    "matched": False,
                    "message": "config injection requires a dotted target",
                }
            )
            continue
        cursor: Any = shadow
        parts = target.split(".")
        ok = True
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                ok = False
                break
            cursor = cursor[part]
        if not ok or not isinstance(cursor, dict):
            hits.append(
                {
                    "injection_id": item["id"],
                    "stage": "config",
                    "kind": item.get("kind"),
                    "field": target,
                    "expected_business_code": item.get("business_code", E_CONFIG_INVALID),
                    "applied": False,
                    "matched": False,
                    "message": "config injection target field was not found",
                }
            )
            continue
        cursor[parts[-1]] = deepcopy(value)
        hits.append(
            {
                "injection_id": item["id"],
                "stage": "config",
                "kind": item.get("kind"),
                "field": target,
                "injected_value": value,
                "expected_business_code": item.get("business_code", E_CONFIG_INVALID),
                "applied": True,
                "matched": False,
            }
        )
    if not hits:
        return [], []
    _, errors = normalise_config(shadow)
    config_errors = [error for error in errors if error.get("business_code") == E_CONFIG_INVALID]
    for hit in hits:
        if not hit.get("applied"):
            continue
        field = hit.get("field", "")
        related = [error for error in config_errors if error.get("field") == field]
        hit["matched"] = bool(related)
        if related:
            hit["matched_diagnostic"] = related[0]
    return hits, config_errors


def finalize_row_injection_hits(
    hits: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    for hit in hits:
        if hit.get("stage") == "config" or not hit.get("applied"):
            continue
        expected = hit.get("expected_business_code")
        row_id = hit.get("row_id")
        column = hit.get("column")
        matches = [
            failure
            for failure in failures
            if failure.get("row_id") == row_id
            and failure.get("column") == column
            and (expected is None or failure.get("business_code") == expected)
        ]
        hit["matched"] = bool(matches)
        if matches:
            hit["matched_business_code"] = matches[0].get("business_code")
    return sorted(hits, key=lambda item: item["injection_id"])
