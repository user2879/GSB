"""Fast-prefix and authoritative ledger views with budget diagnostics."""

from __future__ import annotations

import time
from typing import Any, Callable

from .errors import E_LEDGER_DIVERGED, E_RESPONSE_BUDGET_EXCEEDED, Issue
from .util import canonical_json


def _budget_response(
    *,
    mode: str,
    entries: list[dict[str, Any]],
    total: int,
    invalidated_total: int,
    invalidated_completed: int,
    budget_ms: int | None,
    exhausted: bool,
    summary: dict[str, Any],
) -> dict[str, Any]:
    errors = [E_RESPONSE_BUDGET_EXCEEDED] if exhausted else []
    return {
        "mode": mode,
        "status": "partial" if exhausted else "complete",
        "error_codes": errors,
        "entries": entries,
        "progress": {
            "completed": len(entries),
            "total": total,
            "fraction": f"{len(entries)}/{total}",
            "complete": not exhausted,
            "invalidated_total": invalidated_total,
            "invalidated_completed": invalidated_completed,
        },
        "budget_ms": budget_ms,
        "summary": summary,
    }


def render_view(
    ledger: dict[str, Any],
    *,
    mode: str,
    budget_ms: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if mode not in {"fast", "authoritative"}:
        raise ValueError("mode must be fast or authoritative")
    entries = ledger["entries"]
    total = len(entries)
    invalidation = ledger["invalidation"]
    invalidated_total = len(invalidation["invalidated_request_ids"])
    if mode == "authoritative" and budget_ms is None:
        completed = list(entries)
        invalidated_completed = sum(
            1
            for entry in completed
            if entry.get("dependency", {}).get("recompute_required")
        )
        return _budget_response(
            mode=mode,
            entries=completed,
            total=total,
            invalidated_total=invalidated_total,
            invalidated_completed=invalidated_completed,
            budget_ms=budget_ms,
            exhausted=False,
            summary=ledger["summary"],
        )

    started = clock()
    deadline = None if budget_ms is None else started + max(0, budget_ms) / 1000.0
    completed: list[dict[str, Any]] = []
    invalidated_completed = 0
    exhausted = False
    for entry in entries:
        if deadline is not None and clock() >= deadline:
            exhausted = True
            break
        completed.append(entry)
        if entry.get("dependency", {}).get("recompute_required"):
            invalidated_completed += 1
    if deadline is not None and len(completed) < total and clock() >= deadline:
        exhausted = True
    return _budget_response(
        mode=mode,
        entries=completed,
        total=total,
        invalidated_total=invalidated_total,
        invalidated_completed=invalidated_completed,
        budget_ms=budget_ms,
        exhausted=exhausted,
        summary=ledger["summary"],
    )


def _diff_fields(
    left: Any,
    right: Any,
    *,
    prefix: str = "",
) -> list[str]:
    fields: list[str] = []
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                fields.append(child)
            else:
                fields.extend(_diff_fields(left[key], right[key], prefix=child))
        return fields
    if isinstance(left, list):
        if len(left) != len(right):
            fields.append(f"{prefix}.length")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            fields.extend(_diff_fields(left_item, right_item, prefix=f"{prefix}[{index}]"))
        return fields
    if left != right:
        fields.append(prefix or "<root>")
    return fields


def compare_views(
    fast: dict[str, Any],
    authoritative: dict[str, Any],
) -> list[Issue]:
    """Compare the common completed range; returns field-specific issues."""

    issues: list[Issue] = []
    common = min(len(fast.get("entries", [])), len(authoritative.get("entries", [])))
    for index in range(common):
        fast_entry = fast["entries"][index]
        authority_entry = authoritative["entries"][index]
        fields = _diff_fields(fast_entry, authority_entry)
        if fields:
            issues.append(
                Issue(
                    code=E_LEDGER_DIVERGED,
                    message=f"fast and authoritative ledger differ at entry index {index}",
                    item=str(fast_entry.get("entry_id", index)),
                    context={"fields": fields, "entry_index": index},
                )
            )
    fast_progress = fast.get("progress", {})
    authority_progress = authoritative.get("progress", {})
    if min(fast_progress.get("completed", 0), authority_progress.get("completed", 0)) == 0:
        return issues
    shared_summary_fields = ("invalidated_total",)
    for field in shared_summary_fields:
        left = fast_progress.get(field)
        right = authority_progress.get(field)
        if left != right:
            issues.append(
                Issue(
                    code=E_LEDGER_DIVERGED,
                    message=f"fast and authoritative progress differ on {field}",
                    item="progress",
                    context={"fields": [f"progress.{field}"]},
                )
            )
    return sorted(issues, key=Issue.sort_key)


def views_identical(left: dict[str, Any], right: dict[str, Any], *, ignore_mode: bool = True) -> bool:
    left_copy = {key: value for key, value in left.items() if not (ignore_mode and key == "mode")}
    right_copy = {key: value for key, value in right.items() if not (ignore_mode and key == "mode")}
    return canonical_json(left_copy) == canonical_json(right_copy)

