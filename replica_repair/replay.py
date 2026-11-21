"""Ledger replay with first-divergence evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import ReplayError
from .ledger import Ledger
from .manifest_repo import compute_state_digest
from .model import Manifest
from .node_service import NodeClient
from .planner import RepairBudget, plan_repairs
from .truth import CopyState, TruthEngine, TruthReport


DISTRIBUTION_FIELDS = (
    "size_bytes",
    "digest",
    "expected_size_bytes",
    "expected_digest",
    "priority",
    "status",
    "manifest_mismatch",
)
TRUTH_FIELDS = ("size_bytes", "truth_digest", "truth_nodes", "ranking")
TASK_FIELDS = (
    "wave",
    "block_id",
    "source_node",
    "target_node",
    "size_bytes",
    "repair_kb",
    "repair_weight",
    "source_priority",
    "target_priority",
    "cost_units",
)


@dataclass(frozen=True)
class ReplayResult:
    group_id: str
    active_snapshot_id: int
    baseline_snapshot_id: int | None
    distribution_checks: int
    truth_checks: int
    plan_checks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "group_id": self.group_id,
            "active_snapshot_id": self.active_snapshot_id,
            "baseline_snapshot_id": self.baseline_snapshot_id,
            "distribution_checks": self.distribution_checks,
            "truth_checks": self.truth_checks,
            "plan_checks": self.plan_checks,
        }


class ReplayVerifier:
    def __init__(self, ledger: Ledger, root, clients: Mapping[str, NodeClient]) -> None:
        self.ledger = ledger
        self.root = root
        self.clients = clients

    def verify(self) -> ReplayResult:
        active_row = self.ledger.active_snapshot_row()
        if active_row is None:
            raise ReplayError(
                "no current ledger snapshot is available for replay",
                details={
                    "first_mismatch": {
                        "block_id": "*",
                        "node_id": "*",
                        "field": "active_snapshot",
                        "ledger_value": None,
                        "recomputed_value": "required",
                        "basis": "ledger has no active non-stale snapshot after recovery",
                    }
                },
            )
        after_manifest = Manifest.from_dict(json.loads(active_row["manifest_json"]))
        after_data = self.ledger.snapshot_data(active_row["snapshot_id"])
        engine = TruthEngine(after_manifest)
        current_states = engine.collect(self.clients, strict_manifest=False)
        current_truth = engine.determine(current_states)
        distribution_checks = _compare_distributions(
            "after_distribution",
            after_data["distributions"],
            current_states,
            basis="sha256 recomputation from persisted block bytes and current manifest",
        )
        truth_checks = _compare_truth(
            "after_truth",
            after_data["truth"],
            current_truth,
            basis="deterministic majority rerun over recomputed current distribution",
        )
        computed_state_digest = compute_state_digest(self.root, after_manifest)
        if computed_state_digest != active_row["state_digest"]:
            _mismatch(
                stage="state_digest",
                block_id="*",
                node_id="*",
                field="state_digest",
                ledger_value=active_row["state_digest"],
                recomputed_value=computed_state_digest,
                basis="current manifest and persisted block bytes differ from the active ledger snapshot",
            )

        baseline_id = active_row["parent_snapshot_id"]
        plan_checks = 0
        if baseline_id is not None:
            baseline_row = self.ledger.snapshot_row(baseline_id)
            if baseline_row is None:
                _mismatch(
                    stage="baseline",
                    block_id="*",
                    node_id="*",
                    field="snapshot_id",
                    ledger_value=baseline_id,
                    recomputed_value=None,
                    basis="active snapshot references a missing baseline",
                )
            baseline_manifest = Manifest.from_dict(json.loads(baseline_row["manifest_json"]))
            baseline_data = self.ledger.snapshot_data(baseline_id)
            baseline_states = _states_from_ledger(
                baseline_manifest,
                baseline_data["distributions"],
            )
            baseline_engine = TruthEngine(baseline_manifest)
            recomputed_baseline_truth = baseline_engine.determine(baseline_states)
            _compare_truth(
                "baseline_truth",
                baseline_data["truth"],
                recomputed_baseline_truth,
                basis="majority rerun from ledger baseline distributions",
            )
            budget_payload = dict(baseline_data["config"]["budget"])
            budget_payload.pop("quota_scope", None)
            budget = RepairBudget(**budget_payload)
            recomputed_plan = plan_repairs(baseline_manifest, recomputed_baseline_truth, budget)
            plan_checks = _compare_plan(
                baseline_data["tasks"],
                recomputed_plan,
                basis="repair plan rerun from replayed baseline truth and stored budget",
            )
        return ReplayResult(
            group_id=after_manifest.group_id,
            active_snapshot_id=int(active_row["snapshot_id"]),
            baseline_snapshot_id=baseline_id,
            distribution_checks=distribution_checks,
            truth_checks=truth_checks,
            plan_checks=plan_checks,
        )


def _states_from_ledger(manifest: Manifest, rows: Sequence[dict[str, Any]]) -> tuple[CopyState, ...]:
    states: list[CopyState] = []
    for row in rows:
        status = row["status"]
        if status == "mismatch" and not row["manifest_mismatch"]:
            status = "ok"
        states.append(
            CopyState(
                node_id=row["node_id"],
                block_id=row["block_id"],
                size_bytes=row["size_bytes"],
                digest=row["digest"],
                expected_size_bytes=row["expected_size_bytes"],
                expected_digest=row["expected_digest"],
                priority=row["priority"],
                status=status,
                manifest_mismatch=bool(row["manifest_mismatch"]),
            )
        )
    states.sort(key=lambda item: (item.block_id, item.node_id))
    return tuple(states)


def _compare_distributions(
    stage: str,
    ledger_rows: Sequence[dict[str, Any]],
    states: Sequence[CopyState],
    *,
    basis: str,
) -> int:
    ledger_map = {(row["block_id"], row["node_id"]): row for row in ledger_rows}
    replay_map = {(row.block_id, row.node_id): row for row in states}
    keys = sorted(set(ledger_map) | set(replay_map))
    for block_id, node_id in keys:
        if (block_id, node_id) not in ledger_map:
            _mismatch(
                stage=stage,
                block_id=block_id,
                node_id=node_id,
                field="presence",
                ledger_value=None,
                recomputed_value="present",
                basis=basis,
            )
        if (block_id, node_id) not in replay_map:
            _mismatch(
                stage=stage,
                block_id=block_id,
                node_id=node_id,
                field="presence",
                ledger_value="present",
                recomputed_value=None,
                basis=basis,
            )
        ledger_row = ledger_map[(block_id, node_id)]
        state = replay_map[(block_id, node_id)]
        replay_row = {
            "size_bytes": state.size_bytes,
            "digest": state.digest,
            "expected_size_bytes": state.expected_size_bytes,
            "expected_digest": state.expected_digest,
            "priority": state.priority,
            "status": state.status,
            "manifest_mismatch": state.manifest_mismatch,
        }
        for field in DISTRIBUTION_FIELDS:
            if ledger_row[field] != replay_row[field]:
                _mismatch(
                    stage=stage,
                    block_id=block_id,
                    node_id=node_id,
                    field=field,
                    ledger_value=ledger_row[field],
                    recomputed_value=replay_row[field],
                    basis=basis,
                )
    return len(keys)


def _compare_truth(
    stage: str,
    ledger_rows: Sequence[dict[str, Any]],
    replay: TruthReport,
    *,
    basis: str,
) -> int:
    ledger_map = {row["block_id"]: row for row in ledger_rows}
    replay_map = {block.block_id: block.to_dict() for block in replay.blocks}
    keys = sorted(set(ledger_map) | set(replay_map))
    for block_id in keys:
        if block_id not in ledger_map:
            _mismatch(
                stage=stage,
                block_id=block_id,
                node_id="*",
                field="presence",
                ledger_value=None,
                recomputed_value="present",
                basis=basis,
            )
        if block_id not in replay_map:
            _mismatch(
                stage=stage,
                block_id=block_id,
                node_id="*",
                field="presence",
                ledger_value="present",
                recomputed_value=None,
                basis=basis,
            )
        ledger_row = ledger_map[block_id]
        replay_row = replay_map[block_id]
        normalized_ledger = {
            "size_bytes": ledger_row["size_bytes"],
            "truth_digest": ledger_row["truth_digest"],
            "truth_nodes": ledger_row["truth_nodes"],
            "ranking": ledger_row["ranking"],
        }
        for field in TRUTH_FIELDS:
            if normalized_ledger[field] != replay_row[field]:
                node_id = "*"
                if field == "truth_nodes" and replay_row[field]:
                    node_id = str(replay_row[field][0])
                _mismatch(
                    stage=stage,
                    block_id=block_id,
                    node_id=node_id,
                    field=field,
                    ledger_value=normalized_ledger[field],
                    recomputed_value=replay_row[field],
                    basis=basis,
                )
    return len(keys)


def _compare_plan(
    ledger_tasks: Sequence[dict[str, Any]],
    replay_plan,
    *,
    basis: str,
) -> int:
    ledger_map = {task["task_id"]: task for task in ledger_tasks}
    replay_map = {task.task_id: task for task in replay_plan.tasks}
    keys = sorted(set(ledger_map) | set(replay_map))
    for task_id in keys:
        if task_id not in ledger_map:
            replay_task = replay_map[task_id]
            _mismatch(
                stage="after_plan",
                block_id=replay_task.block_id,
                node_id=replay_task.target_node,
                field="task_presence",
                ledger_value=None,
                recomputed_value=task_id,
                basis=basis,
            )
        if task_id not in replay_map:
            ledger_task = ledger_map[task_id]
            _mismatch(
                stage="after_plan",
                block_id=ledger_task["block_id"],
                node_id=ledger_task["target_node"],
                field="task_presence",
                ledger_value=task_id,
                recomputed_value=None,
                basis=basis,
            )
        ledger_task = ledger_map[task_id]
        replay_task = replay_map[task_id].to_dict()
        for field in TASK_FIELDS:
            if ledger_task[field] != replay_task[field]:
                _mismatch(
                    stage="after_plan",
                    block_id=ledger_task["block_id"],
                    node_id=ledger_task["target_node"],
                    field=field,
                    ledger_value=ledger_task[field],
                    recomputed_value=replay_task[field],
                    basis=basis,
                )
    return len(keys)


def _mismatch(
    *,
    stage: str,
    block_id: str,
    node_id: str,
    field: str,
    ledger_value: Any,
    recomputed_value: Any,
    basis: str,
) -> None:
    raise ReplayError(
        f"E_REPLAY at block {block_id}, node {node_id}, field {field}",
        path=f"$.{stage}.{block_id}.{node_id}.{field}",
        details={
            "stage": stage,
            "first_mismatch": {
                "block_id": block_id,
                "node_id": node_id,
                "field": field,
                "ledger_value": ledger_value,
                "recomputed_value": recomputed_value,
                "basis": basis,
            },
        },
    )
