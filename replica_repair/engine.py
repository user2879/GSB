"""End-to-end local pipeline: upgrade, truth, budgeted repair, ledger, replay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import RepairError, StateError
from .ledger import Ledger
from .manifest_repo import ManifestRepository, compute_state_digest
from .manifest_store import ManifestStore
from .model import Manifest, recompute_derivations
from .node_service import NodeClient
from .planner import RepairBudget, RepairExecutionResult, RepairExecutor, RepairPlan, plan_repairs
from .replay import ReplayVerifier
from .truth import TruthEngine, TruthReport
from .util import sha256_file


@dataclass(frozen=True)
class PipelineResult:
    manifest: Manifest
    upgrade: dict[str, Any]
    reconciliation: dict[str, Any]
    recovery: dict[str, Any]
    before_truth: TruthReport
    plan: RepairPlan
    execution: RepairExecutionResult
    after_truth: TruthReport
    replay: dict[str, Any] | None
    ledger_report: dict[str, Any]
    before_snapshot_id: int
    after_snapshot_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "group_id": self.manifest.group_id,
            "manifest_version": self.manifest.manifest_version,
            "upgrade": self.upgrade,
            "reconciliation": self.reconciliation,
            "recovery": self.recovery,
            "before_truth": self.before_truth.to_dict(),
            "plan": self.plan.to_dict(),
            "execution": self.execution.to_dict(),
            "after_truth": self.after_truth.to_dict(),
            "replay": self.replay,
            "ledger": self.ledger_report,
            "before_snapshot_id": self.before_snapshot_id,
            "after_snapshot_id": self.after_snapshot_id,
        }


def run_pipeline(
    *,
    root: Path | str,
    manifest_path: Path | str,
    ledger_path: Path | str,
    clients: Mapping[str, NodeClient],
    budget: RepairBudget,
    verify_replay: bool = True,
    after_progress: Callable[[int], None] | None = None,
) -> PipelineResult:
    root = Path(root)
    manifest_path = Path(manifest_path)
    budget.validate()

    upgrade_result = ManifestStore(root).load_current(manifest_path)
    manifest = upgrade_result.manifest
    upgrade_payload = upgrade_result.to_dict()

    # If a previous run was interrupted after writing bytes but before
    # committing manifest digests, reconcile only copies that already equal
    # the prior run's recorded truth. Ledger recovery still marks those tasks
    # unknown; reconciliation merely restores on-disk consistency first.
    with Ledger(ledger_path) as ledger:
        ledger.bind_source(root, manifest_path)
        reconciliation = reconcile_interrupted_repairs(
            ledger=ledger,
            manifest_path=manifest_path,
            manifest=manifest,
            clients=clients,
            root=root,
        )
        manifest = reconciliation["manifest"]
        if reconciliation.get("manifest_changed"):
            for client in clients.values():
                client.reload_manifest()

        before_state_digest = compute_state_digest(root, manifest)
        recovery = ledger.open_for_state(manifest, before_state_digest)
        truth_engine = TruthEngine(manifest)
        before_truth = truth_engine.run(clients, strict_manifest=True)
        plan = plan_repairs(manifest, before_truth, budget)
        before_snapshot_id = ledger.record_before(
            manifest,
            before_state_digest,
            before_truth,
            plan,
            budget,
        )

        executor = RepairExecutor(
            manifest,
            clients,
            progress_hook=ledger.update_task_progress,
            after_progress=after_progress,
        )
        try:
            execution = executor.execute(plan)
        except BaseException as exc:
            code = exc.code if isinstance(exc, RepairError) else "E_STATE"
            ledger.mark_run_failed(code)
            raise

        if execution.results:
            manifest = apply_repairs_to_manifest(manifest_path, manifest, execution)
            for client in clients.values():
                client.reload_manifest()

        after_state_digest = compute_state_digest(root, manifest)
        after_truth = TruthEngine(manifest).run(clients, strict_manifest=True)
        after_snapshot_id = ledger.record_after(
            manifest,
            after_state_digest,
            after_truth,
            before_snapshot_id,
            budget,
        )
        replay_payload: dict[str, Any] | None = None
        if verify_replay:
            replay_payload = ReplayVerifier(ledger, root, clients).verify().to_dict()
        ledger_report = ledger.current_report()
        reconciliation_payload = dict(reconciliation)
        reconciliation_payload.pop("manifest", None)
        return PipelineResult(
            manifest=manifest,
            upgrade=upgrade_payload,
            reconciliation=reconciliation_payload,
            recovery=recovery.to_dict(),
            before_truth=before_truth,
            plan=plan,
            execution=execution,
            after_truth=after_truth,
            replay=replay_payload,
            ledger_report=ledger_report,
            before_snapshot_id=before_snapshot_id,
            after_snapshot_id=after_snapshot_id,
        )


def reconcile_interrupted_repairs(
    *,
    ledger: Ledger,
    manifest_path: Path,
    manifest: Manifest,
    clients: Mapping[str, NodeClient],
    root: Path,
) -> dict[str, Any]:
    active = ledger.active_snapshot_row()
    progress_status = ledger.current_report()["progress"]["status"]
    if active is None or progress_status not in {"running", "recovering", "failed"}:
        return {
            "performed": False,
            "manifest_changed": False,
            "manifest": manifest,
            "reconciled_tasks": [],
            "reason": "no interrupted ledger run",
        }
    snapshot = ledger.snapshot_data(active["snapshot_id"])
    if snapshot["phase"] != "before":
        return {
            "performed": False,
            "manifest_changed": False,
            "manifest": manifest,
            "reconciled_tasks": [],
            "reason": "active interrupted snapshot is not a baseline",
        }
    truth_by_block = {row["block_id"]: row for row in snapshot["truth"]}
    node_map = manifest.node_map()
    reconciled: list[dict[str, Any]] = []
    changed = False
    for task in snapshot["tasks"]:
        truth = truth_by_block.get(task["block_id"])
        if truth is None or task["target_node"] not in node_map:
            continue
        target_node = node_map[task["target_node"]]
        block_path = root / "nodes" / task["target_node"] / "blocks" / f"{task['block_id']}.bin"
        if not block_path.is_file():
            continue
        if block_path.stat().st_size != truth["size_bytes"]:
            continue
        if sha256_file(block_path) != truth["truth_digest"]:
            continue
        target_blocks = target_node.block_map()
        block = target_blocks.get(task["block_id"])
        if block is None:
            continue
        if block.digest == truth["truth_digest"] and block.repair_source == task["source_node"]:
            continue
        new_block = replace(
            block,
            size_bytes=truth["size_bytes"],
            digest=truth["truth_digest"],
            repair_source=task["source_node"],
        )
        new_blocks = tuple(
            new_block if item.block_id == new_block.block_id else item
            for item in target_node.blocks
        )
        node_map[task["target_node"]] = replace(target_node, blocks=new_blocks)
        reconciled.append(
            {
                "task_id": task["task_id"],
                "block_id": task["block_id"],
                "target_node": task["target_node"],
                "source_node": task["source_node"],
            }
        )
        changed = True
    if not changed:
        return {
            "performed": True,
            "manifest_changed": False,
            "manifest": manifest,
            "reconciled_tasks": [],
            "reason": "no persisted repair bytes match recorded truth",
        }
    reconciled_nodes = recompute_derivations(
        tuple(node_map[node_id] for node_id in sorted(node_map))
    )
    candidate = replace(manifest, nodes=reconciled_nodes)
    save_result = ManifestRepository(manifest_path).save_current(
        candidate,
        recompute_relations=True,
    )
    return {
        "performed": True,
        "manifest_changed": True,
        "manifest": save_result.manifest,
        "reconciled_tasks": reconciled,
        "manifest_version_incremented": save_result.version_incremented,
        "changes": save_result.changes.to_dict(),
        "reason": "restored manifest digests for bytes matching the interrupted run's recorded truth",
    }


def apply_repairs_to_manifest(
    manifest_path: Path,
    manifest: Manifest,
    execution: RepairExecutionResult,
) -> Manifest:
    nodes = {node.node_id: node for node in manifest.nodes}
    for result in execution.results:
        node = nodes[result.target_node]
        blocks = node.block_map()
        if result.block_id not in blocks:
            raise StateError(
                "repaired block is absent from target manifest",
                details={"node_id": result.target_node, "block_id": result.block_id},
            )
        blocks[result.block_id] = replace(
            blocks[result.block_id],
            size_bytes=result.bytes_written,
            digest=result.digest,
            repair_source=result.source_node,
        )
        nodes[result.target_node] = replace(
            node,
            blocks=tuple(blocks[block_id] for block_id in sorted(blocks)),
        )
    candidate = replace(
        manifest,
        nodes=recompute_derivations(tuple(nodes[node_id] for node_id in sorted(nodes))),
    )
    save_result = ManifestRepository(manifest_path).save_current(
        candidate,
        recompute_relations=True,
    )
    return save_result.manifest
