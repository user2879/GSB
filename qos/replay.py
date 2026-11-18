"""恢复后的一致性重放校验。

从快照记录的规则集与依赖图版本出发，重新推导失效集合并全量编译计划，
逐项比较失效序列、计划摘要与每规则产物摘要；首个差异以 E_REPLAY 报告，
并给出具体依据（期望值/实际值/字段）。
"""
from __future__ import annotations

from typing import Any

from .canonical import canonical_text
from .changeset import ChangeSet, apply_changeset
from .compiler import PlanCompiler
from .errors import ReplayError
from .models import Rule, parse_rules
from .snapshot import Snapshot


def replay_snapshot(snapshot: Snapshot,
                    baseline_rules: list[Rule] | None = None) -> dict[str, Any]:
    """重放快照。

    baseline_rules: 快照变更前的基线规则集。若提供，则按快照 change_summary
    重建变更集并重放失效推导；否则直接对快照规则集做全量编译比对。
    """
    snap_rules = parse_rules(snapshot.rules)

    # 1) 重放失效序列与传播轨迹。
    if baseline_rules is not None and snapshot.change_summary:
        changeset = _rebuild_changeset(snapshot, baseline_rules)
        applied = apply_changeset(baseline_rules, changeset)
        replay_sequence = applied.invalidation_sequence
        replay_trace = [entry.to_dict() for entry in applied.trace]
        target_rules = applied.rules
    else:
        # 无基线时，失效序列只能与快照记录做存在性一致性校验。
        replay_sequence = list(snapshot.invalidation_sequence)
        replay_trace = list(snapshot.invalidation_trace)
        target_rules = snap_rules

    if replay_sequence != snapshot.invalidation_sequence:
        first = _first_list_diff(replay_sequence, snapshot.invalidation_sequence)
        raise ReplayError(
            "重放失效规则序列与快照不一致",
            details={"field": "invalidation_sequence", **first,
                     "basis": "按快照规则集版本与依赖图版本重新推导"},
        )
    if canonical_text(replay_trace) != canonical_text(snapshot.invalidation_trace):
        raise ReplayError(
            "重放失效传播轨迹与快照不一致",
            details={"field": "invalidation_trace",
                     "replay": replay_trace,
                     "snapshot": snapshot.invalidation_trace,
                     "basis": "传播触发源/依赖边记录不一致"},
        )

    # 2) 全量重编译并比较计划摘要与每规则产物摘要。
    from .ruleset import build_graph
    graph = build_graph(target_rules)
    plan = PlanCompiler.full_recompile(target_rules, graph)

    if plan.plan_hash != snapshot.plan_hash:
        first_rule, field = _first_artifact_diff(plan, snapshot)
        raise ReplayError(
            "重放计划摘要与快照不一致",
            details={"field": field or "plan_hash",
                     "first_diff_rule": first_rule,
                     "replay_plan_hash": plan.plan_hash,
                     "snapshot_plan_hash": snapshot.plan_hash,
                     "basis": "对快照规则集全量重新编译后的规范化计划"},
        )

    for rid in sorted(set(plan.artifact_hashes) | set(snapshot.artifact_hashes)):
        replay_hash = plan.artifact_hashes.get(rid)
        snapshot_hash = snapshot.artifact_hashes.get(rid)
        if replay_hash != snapshot_hash:
            raise ReplayError(
                "重放规则产物摘要与快照不一致",
                details={"field": "artifact_hashes", "first_diff_rule": rid,
                         "replay_hash": replay_hash,
                         "snapshot_hash": snapshot_hash,
                         "basis": f"规则 {rid} 的规范化编译产物哈希"},
            )

    return {
        "ok": True,
        "plan_hash": plan.plan_hash,
        "invalidation_sequence": replay_sequence,
        "artifact_rule_count": len(plan.artifact_hashes),
        "compared_fields": ["invalidation_sequence", "invalidation_trace",
                            "plan_hash", "artifact_hashes"],
    }


def _rebuild_changeset(snapshot: Snapshot,
                       baseline_rules: list[Rule]) -> ChangeSet:
    """依据快照 change_summary 重建变更集（新增/修改用快照中的规则体）。"""
    summary = snapshot.change_summary
    snap_by_id = {item["rule_id"]: item for item in snapshot.rules}
    upsert_ids = set(summary.get("added", [])) | set(summary.get("modified", []))
    upserts = [snap_by_id[rid] for rid in sorted(upsert_ids)]
    edges = [{"from_rule": edge.split("->")[1] if "->" in edge else edge,
              "upstream": edge.split("->")[0]}
             for edge in summary.get("edge_removed", [])]
    return ChangeSet(
        upserts=upserts,
        delete_rule_ids=list(summary.get("deleted", [])),
        remove_edges=edges,
    )


def _first_list_diff(actual: list[str], expected: list[str]) -> dict[str, Any]:
    for idx in range(max(len(actual), len(expected))):
        a = actual[idx] if idx < len(actual) else None
        e = expected[idx] if idx < len(expected) else None
        if a != e:
            return {"index": idx, "replay_value": a, "snapshot_value": e}
    return {"index": -1, "replay_value": None, "snapshot_value": None}


def _first_artifact_diff(plan: Any, snapshot: Snapshot) -> tuple[str | None, str | None]:
    replay_entries = {e["rule_id"]: e for e in plan.entries}
    snap_rules = {item["rule_id"]: item for item in snapshot.rules}
    for rid in sorted(set(replay_entries) | set(snap_rules)):
        if plan.artifact_hashes.get(rid) != snapshot.artifact_hashes.get(rid):
            return rid, "artifact_hashes"
    return None, None
