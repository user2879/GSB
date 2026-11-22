from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .canonical import content_digest, read_json, write_canonical
from .errors import AppError
from .ledger import Ledger
from .models import Spec


SNAPSHOT_FORMAT_VERSION = 1


def make_snapshot(spec: Spec, ledger: Ledger) -> dict[str, Any]:
    trace = ledger.current_trace()
    payload: dict[str, Any] = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "versions": {
            "clusters": spec.versions.clusters,
            "revisions": spec.versions.revisions,
            "budgets": spec.versions.budgets,
            "ledger": ledger.ledger_versions(),
        },
        "spec": spec.to_dict(),
        "plan": ledger.run_plan() if trace["run_id"] is not None else None,
        "run": trace,
    }
    payload["content_digest"] = content_digest(payload)
    return payload


def save_snapshot(path: str | Path, spec: Spec, ledger: Ledger) -> dict[str, Any]:
    payload = make_snapshot(spec, ledger)
    write_canonical(path, payload)
    return payload


def load_snapshot(path: str | Path, current_versions: dict[str, int] | None = None) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except FileNotFoundError:
        raise AppError("E_SNAPSHOT", "快照文件不存在", "snapshot", str(path)) from None
    except (OSError, ValueError) as exc:
        raise AppError("E_SNAPSHOT", f"快照无法解析: {exc}", "snapshot", str(path)) from None
    if not isinstance(payload, dict):
        raise AppError("E_SNAPSHOT", "快照必须为 JSON 对象", "snapshot", "$")
    if payload.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise AppError(
            "E_SNAPSHOT",
            "快照格式版本不受支持",
            "format_version",
            "$.format_version",
            {"found": payload.get("format_version"), "supported": SNAPSHOT_FORMAT_VERSION},
        )
    actual = payload.get("content_digest")
    unsigned = {key: value for key, value in payload.items() if key != "content_digest"}
    expected = content_digest(unsigned)
    if not isinstance(actual, str) or actual != expected:
        raise AppError(
            "E_SNAPSHOT",
            "快照内容摘要不一致，拒绝恢复",
            "content_digest",
            "$.content_digest",
            {"expected": expected, "actual": actual, "replacement_mode": "whole_snapshot_only"},
        )
    if "spec" not in payload or "run" not in payload:
        raise AppError("E_SNAPSHOT", "快照缺少会话数据", "snapshot", "$")

    restored = dict(unsigned)
    restored["content_digest"] = actual
    version_changed = False
    reasons: list[str] = []
    if current_versions is not None:
        for key in ("clusters", "revisions", "budgets"):
            old = current_versions.get(key)
            new = payload.get("versions", {}).get(key)
            if old != new:
                version_changed = True
                reasons.append(f"{key} version {old} -> {new}")
    restored["restore"] = {
        "mode": "whole_replace",
        "incremental_merge": False,
        "version_changed": version_changed,
        "reasons": reasons,
    }
    return restored


def _replay_mismatch(
    trace: dict[str, Any], field: str, expected: Any, actual: Any, checked: int
) -> AppError:
    return AppError(
        "E_REPLAY",
        "恢复后的重放结果与快照不一致",
        field,
        trace.get("shard_id"),
        {
            "cluster_id": trace.get("cluster_id"),
            "task_id": trace.get("task_id"),
            "shard_id": trace.get("shard_id"),
            "field": field,
            "expected": expected,
            "actual": actual,
            "records_checked_before_mismatch": checked,
            "basis": "first completed shard in stable (cluster,task,shard_index) order",
        },
    )


def replay_snapshot(
    snapshot: dict[str, Any],
    dispatch: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    plan = snapshot.get("plan")
    if not isinstance(plan, dict):
        raise AppError("E_REPLAY", "快照没有可重放的分发计划", "plan", "$.plan")
    plan_shards = {shard["shard_id"]: shard for shard in plan.get("shards", [])}
    traces = [
        trace
        for trace in snapshot.get("run", {}).get("traces", [])
        if trace.get("outcome") == "completed"
    ]
    traces.sort(key=lambda item: (item["cluster_id"], item["task_id"], item["shard_index"], item["shard_id"]))
    checked = 0
    replayed: list[str] = []
    for trace in traces:
        shard = plan_shards.get(trace["shard_id"])
        if shard is None:
            raise _replay_mismatch(trace, "plan", "shard present", "missing", checked)
        try:
            response = dispatch(shard)
        except AppError as exc:
            raise AppError(
                "E_REPLAY",
                "恢复重放的分片下发失败",
                "dispatch",
                trace["shard_id"],
                {
                    "cluster_id": trace["cluster_id"],
                    "task_id": trace["task_id"],
                    "shard_id": trace["shard_id"],
                    "field": "dispatch",
                    "cause": exc.to_dict(),
                },
            ) from None
        actual = response.get("aggregate") or {}
        expected = trace.get("aggregate") or {}
        for field in ("scan_rows", "byte_digest", "summary_digest", "groups"):
            if expected.get(field) != actual.get(field):
                raise _replay_mismatch(trace, field, expected.get(field), actual.get(field), checked)
        checked += 1
        replayed.append(trace["shard_id"])
    return {
        "records_replayed": checked,
        "shard_ids": replayed,
        "first_mismatch": None,
        "status": "consistent",
    }
