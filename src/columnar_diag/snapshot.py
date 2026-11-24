from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import AppError
from .models import SNAPSHOT_FORMAT_VERSION, BudgetConfig
from .planner import bundle_to_dict
from .util import atomic_write_json, read_json, sha256_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def build_snapshot(workspace: Any) -> dict[str, Any]:
    state = workspace.ledger.export_active_state()
    generation = state["generation"]
    tasks = state["tasks"]
    snapshot: dict[str, Any] = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "created_at": _now(),
        "segment_set": workspace.segment_set_summary(),
        "versions": {
            "segment_format_version": workspace.versions["segment_format_version"],
            "encoding_selection_version": workspace.versions[
                "encoding_selection_version"
            ],
            "budget_params_version": workspace.versions["budget_params_version"],
        },
        "budget": workspace.budget.to_dict(),
        "cursor": {
            "task_index": int(generation["cursor"]),
            "tasks_total": len(workspace.bundle.tasks),
            "committed_batches": int(generation["committed_batches"]),
        },
        "scan_sequence": [task.to_dict() for task in workspace.bundle.tasks],
        "plan_bundle": bundle_to_dict(workspace.bundle),
        "block_digests": {
            block_id: encoded.block_digest
            for block_id, encoded in sorted(workspace.image.blocks.items())
        },
        "result_digests": {
            row["task_id"]: row["result_digest"] for row in tasks
        },
        "statistics": workspace.ledger.statistics(),
        "task_trajectory": {
            "batches": state["batches"],
            "tasks": tasks,
        },
        "ledger_state": state,
    }
    snapshot["content_digest"] = sha256_json(snapshot)
    return snapshot


def save_snapshot(workspace: Any, path: str | Path) -> dict[str, Any]:
    snapshot = build_snapshot(workspace)
    atomic_write_json(path, snapshot, pretty=True)
    return {
        "path": str(path),
        "content_digest": snapshot["content_digest"],
        "cursor": snapshot["cursor"],
        "task_count": len(snapshot["task_trajectory"]["tasks"]),
        "query_result_count": len(snapshot["ledger_state"]["query_results"]),
        "segment_set_digest": snapshot["segment_set"]["digest"],
    }


def load_snapshot(path: str | Path) -> dict[str, Any]:
    try:
        snapshot = read_json(path)
    except (OSError, ValueError) as error:
        raise AppError(
            "E_SNAPSHOT", f"无法读取快照: {error}", field="snapshot"
        ) from error
    if not isinstance(snapshot, dict):
        raise AppError("E_SNAPSHOT", "快照根节点必须为对象", field="snapshot")
    if snapshot.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
        raise AppError(
            "E_SNAPSHOT",
            "不支持的快照格式版本",
            field="snapshot_format_version",
            details={"actual": snapshot.get("snapshot_format_version")},
        )
    expected = snapshot.get("content_digest")
    payload = dict(snapshot)
    payload.pop("content_digest", None)
    actual = sha256_json(payload)
    if not isinstance(expected, str) or expected != actual:
        raise AppError(
            "E_SNAPSHOT",
            "快照内容摘要不匹配，拒绝恢复",
            field="content_digest",
            details={"expected": expected, "actual": actual},
        )
    for key in (
        "segment_set",
        "versions",
        "budget",
        "cursor",
        "scan_sequence",
        "block_digests",
        "result_digests",
        "statistics",
        "ledger_state",
    ):
        if key not in snapshot:
            raise AppError("E_SNAPSHOT", f"快照缺少字段 {key}", field=key)
    return snapshot


def restore_snapshot(workspace: Any, path: str | Path) -> dict[str, Any]:
    snapshot = load_snapshot(path)
    workspace.ledger.import_as_current(
        snapshot["ledger_state"], reason="snapshot-whole-view-replace"
    )
    current_segment_set = workspace.segment_set_summary()
    snapshot_segment_set = snapshot["segment_set"]
    mismatches: dict[str, Any] = {}
    if current_segment_set["digest"] != snapshot_segment_set.get("digest"):
        mismatches["segment_set_digest"] = {
            "current": current_segment_set["digest"],
            "snapshot": snapshot_segment_set.get("digest"),
        }
    current_versions = workspace.versions
    snapshot_versions = snapshot["versions"]
    for key in ("segment_format_version", "encoding_selection_version"):
        if int(current_versions[key]) != int(snapshot_versions.get(key, -1)):
            mismatches[key] = {
                "current": int(current_versions[key]),
                "snapshot": snapshot_versions.get(key),
            }
    snapshot_sequence = [
        (item["task_id"], int(item["block_ordinal"]))
        for item in snapshot["scan_sequence"]
    ]
    current_sequence = [
        (task.task_id, task.block_ordinal) for task in workspace.bundle.tasks
    ]
    if snapshot_sequence != current_sequence:
        first_mismatch = 1
        for left, right in zip(snapshot_sequence, current_sequence):
            if left != right:
                first_mismatch = left[1]
                break
        mismatches["scan_sequence"] = {
            "snapshot_length": len(snapshot_sequence),
            "current_length": len(current_sequence),
            "first_block_ordinal": first_mismatch,
        }
    if mismatches:
        raise AppError(
            "E_SNAPSHOT",
            "快照版本或摘要与当前上下文不一致；当前视图已整体替换为快照内容",
            field="snapshot_context",
            block_ordinal=mismatches.get("scan_sequence", {}).get(
                "first_block_ordinal"
            ),
            details={"state_replaced": True, "mismatches": mismatches},
        )
    budget_restored = False
    if (
        workspace.budget.to_dict() != snapshot["budget"]
        or workspace.versions["budget_params_version"]
        != int(snapshot_versions["budget_params_version"])
    ):
        workspace.budget = BudgetConfig.from_dict(snapshot["budget"])
        workspace.config["budget"] = workspace.budget.to_dict()
        workspace.config["versions"]["budget_params_version"] = int(
            snapshot_versions["budget_params_version"]
        )
        workspace._save_config()
        budget_restored = True
    generation = workspace.ledger.current_generation()
    return {
        "status": "restored",
        "content_digest": snapshot["content_digest"],
        "cursor": snapshot["cursor"],
        "generation_id": generation["generation_id"] if generation else None,
        "task_count": len(snapshot["task_trajectory"]["tasks"]),
        "segment_set_digest": snapshot_segment_set["digest"],
        "versions": snapshot_versions,
        "budget_restored_from_snapshot": budget_restored,
        "budget": workspace.budget.to_dict(),
        "whole_view_replacement": True,
    }
