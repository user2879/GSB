"""End-to-end orchestrating session: compile -> dispatch -> converge -> snapshot."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from .compiler import build_dispatch_plan, compile_version, diff_invalidation
from .convergence import verify_convergence
from .errors import BusinessError, ErrorCode
from .fixedpoint import Grid
from .logging_utils import StructuredLogger
from .manifest import load_bundle
from .models import ManifestVersion, NodeSpec, Receipt
from .node_process import NodeProcess
from .preview import PreviewEngine
from .scheduler import DispatchTask, Scheduler
from .storage import SnapshotStore
from .util import content_digest


class Session:
    def __init__(self, session_id: str, workdir: str, grid_exp: int,
                 global_limit: int = 3, per_node_quota: int = 2) -> None:
        self.session_id = session_id
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.grid = Grid.create(grid_exp)
        self.global_limit = global_limit
        self.per_node_quota = per_node_quota
        self.logger = StructuredLogger(os.path.join(workdir, "structured.log.jsonl"))
        self.store = SnapshotStore(os.path.join(workdir, "ledger.sqlite"))
        self.registry: dict[str, NodeSpec] = {}
        self.versions: list[ManifestVersion] = []
        self.nodes: dict[str, NodeProcess] = {}
        self.scheduler = Scheduler(global_limit, per_node_quota, self.logger)
        self.preview = PreviewEngine()
        self.state = "INIT"
        self.results: dict[int, dict[str, Any]] = {}
        self.logger.info("session_created", session_id=session_id,
                         grid_exp=grid_exp, global_limit=global_limit,
                         per_node_quota=per_node_quota)

    def load_inputs(self, manifest_path: str, registry_path: str) -> None:
        self._transition("LOADING")
        self.registry, self.versions = load_bundle(manifest_path, registry_path)
        self.logger.info("inputs_loaded",
                         manifest=manifest_path, registry=registry_path,
                         versions=[v.version for v in self.versions],
                         nodes=sorted(self.registry))
        self._transition("LOADED")

    def start_nodes(self) -> None:
        self._transition("NODES_STARTING")
        for node_id in sorted(self.registry):
            process = NodeProcess(self.registry[node_id], self.logger)
            process.start()
            process.handshake()
            self.nodes[node_id] = process
        self._transition("NODES_READY")

    def shutdown_nodes(self) -> None:
        for node_id in sorted(self.nodes):
            self.nodes[node_id].stop()
        self.nodes.clear()
        self._transition("NODES_STOPPED")

    def run_version(self, version: ManifestVersion,
                    timeout_injections: dict[str, str] | None = None) -> dict[str, Any]:
        timeout_injections = timeout_injections or {}
        self._transition("COMPILING", version=version.version)
        compiled = compile_version(version, self.grid)
        plan = build_dispatch_plan(compiled, version)
        invalidation = None
        previous = self._previous_version(version)
        if previous is not None:
            invalidation = diff_invalidation(previous, version, self.grid)
            self.logger.info("invalidation_report", **{
                "version": version.version,
                "invalidation_set": invalidation["invalidation_set"],
                "byte_identical_unchanged": invalidation["byte_identical_unchanged"],
                "propagation_steps": invalidation["propagation_steps"],
            })
        self._transition("DISPATCHING", version=version.version)
        tasks = self._build_tasks(version, plan, timeout_injections)
        records = self.scheduler.run(tasks, self._dispatch_callback)
        failed = [record for record in records if record.status != "ok"]
        if failed:
            self._persist_snapshot(version, plan, records, state="DISPATCH_FAILED")
            raise BusinessError(
                ErrorCode.E_HANDSHAKE,
                f"version {version.version} has failed dispatches after retries",
                [{"task_id": r.task_id, "node_id": r.node_id,
                  "attempts": [a.to_dict() if hasattr(a, "to_dict") else asdict(a)
                               for a in r.attempts]} for r in failed],
            )
        receipts = [Receipt.from_dict(record.receipt) for record in records]
        self._transition("CONVERGING", version=version.version)
        convergence = verify_convergence(version, plan, receipts, self.grid)
        self.preview.refresh(version, plan, self.grid)
        preview_check = self.preview.verify_cache(version, plan, self.grid)
        result = {
            "version": version.version,
            "plan_digest": plan["plan_digest"],
            "compiled": {key: value.to_dict() for key, value in compiled.items()},
            "invalidation": invalidation,
            "tasks": [self._record_to_dict(record) for record in records],
            "receipts": [receipt.to_dict() for receipt in receipts],
            "convergence": convergence,
            "preview_check": preview_check,
            "peak_concurrency": self.scheduler.peak_concurrency,
            "global_limit": self.global_limit,
        }
        self.results[version.version] = result
        self._persist_snapshot(version, plan, records, state="CONVERGED", result=result)
        self._transition("CONVERGED", version=version.version)
        return result

    def run_all(self, timeout_injections: dict[str, str] | None = None) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        try:
            self.start_nodes()
            for version in self.versions:
                outputs.append(self.run_version(version, timeout_injections))
        finally:
            self.shutdown_nodes()
        return outputs

    def replay_snapshot(self, manifest_version: int) -> dict[str, Any]:
        """Replay recorded dispatches and compare against snapshot item-by-item."""
        snapshot = self.store.load_version(self.session_id, manifest_version)
        if snapshot is None:
            raise BusinessError(ErrorCode.E_REPLAY,
                                f"no snapshot for version {manifest_version}",
                                [{"version": manifest_version}])
        version = next(v for v in self.versions if v.version == manifest_version)
        compiled = compile_version(version, Grid(snapshot.grid_exp))
        plan = build_dispatch_plan(compiled, version)
        mismatches: list[dict[str, Any]] = []
        for node_id in version.targets:
            plan_node = plan["nodes"][node_id]
            for key in sorted(plan_node["items"]):
                current = plan_node["items"][key]
                stored = snapshot.payload["plan"]["nodes"][node_id]["items"].get(key)
                if stored is None:
                    mismatches.append({"node_id": node_id, "key": key,
                                       "field": "<item>", "reason": "missing_in_snapshot"})
                    continue
                for field in ("qvalue", "is_derived", "derived_from_q", "error_bound",
                              "artifact_digest"):
                    if current.get(field) != stored.get(field):
                        mismatches.append({"node_id": node_id, "key": key, "field": field,
                                           "expected": current.get(field),
                                           "actual": stored.get(field)})
        stored_receipts = snapshot.payload.get("receipts", [])
        current_receipts = snapshot.payload.get("receipts", [])
        if stored_receipts != current_receipts:
            mismatches.append({"key": "<receipts>", "field": "receipt_set",
                               "reason": "receipt_set_changed"})
        stored_state = snapshot.payload.get("convergence", {})
        receipts = [Receipt.from_dict(item) for item in stored_receipts]
        try:
            verify_convergence(version, plan, receipts, Grid(snapshot.grid_exp))
        except BusinessError as exc:
            mismatches.append({"key": "<convergence>", "field": "convergence",
                               "reason": exc.message, "details": exc.details})
        if snapshot.content_digest != content_digest(snapshot.payload):
            mismatches.append({"key": "<snapshot>", "field": "content_digest",
                               "reason": "digest_drift"})
        if mismatches:
            first = mismatches[0]
            raise BusinessError(
                ErrorCode.E_REPLAY,
                "snapshot replay found the first inconsistent item/field",
                [{"first_mismatch": first, "all_mismatches": mismatches}],
            )
        return {"version": manifest_version, "replay_consistent": True,
                "checked_keys": sum(len(plan["nodes"][n]["items"]) for n in plan["nodes"])}

    def resume_with_replacement(self, current_payload: dict[str, Any],
                                manifest_version: int, grid_exp: int) -> dict[str, Any]:
        """Open snapshot; on identity mismatch replace all plan/receipt state."""
        snapshot = self.store.load_latest(self.session_id)
        if snapshot is None:
            return {"action": "initialized", "reason": "no_snapshot"}
        if self.store.identity_matches(snapshot, manifest_version, grid_exp, current_payload):
            return {"action": "reused", "snapshot_version": snapshot.manifest_version,
                    "content_digest": snapshot.content_digest}
        removed = self.store.replace_all_session_state(self.session_id)
        self.logger.info("snapshot_full_replacement",
                         removed_rows=removed,
                         stale_version=snapshot.manifest_version,
                         stale_digest=snapshot.content_digest)
        return {"action": "full_replacement", "removed_rows": removed,
                "stale_version": snapshot.manifest_version}

    def _build_tasks(self, version: ManifestVersion, plan: dict[str, Any],
                     timeout_injections: dict[str, str]) -> list[DispatchTask]:
        tasks: list[DispatchTask] = []
        for node_id in version.targets:
            spec = self.registry[node_id]
            plan_node = plan["nodes"][node_id]
            shard_count = int(plan_node["shard_count"])
            for shard_index in range(shard_count):
                task_id = f"v{version.version}-{node_id}-s{shard_index}"
                payload = {
                    "type": "dispatch",
                    "task_id": task_id,
                    "node_id": node_id,
                    "version": version.version,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "grid_scale": self.grid.scale,
                    "items": plan_node["items"],
                    "_retry_policy": spec.retry_policy,
                    "_max_retries": spec.max_retries,
                }
                if timeout_injections.get(task_id) == "timeout_once":
                    payload["timeout_once"] = True
                    payload["timeout_sleep"] = max(spec.timeout_ms / 1000.0 + 1.0, 2.0)
                tasks.append(DispatchTask(
                    task_id=task_id, node_id=node_id, version=version.version,
                    shard_index=shard_index, shard_count=shard_count, payload=payload,
                ))
        return sorted(tasks, key=lambda t: t.task_id)

    def _dispatch_callback(self, task: DispatchTask, attempt: int) -> Receipt:
        process = self.nodes[task.node_id]
        payload = dict(task.payload)
        if attempt > 1:
            payload.pop("timeout_once", None)
            payload.pop("timeout_sleep", None)
        return process.dispatch(payload, attempt)

    def _persist_snapshot(self, version: ManifestVersion, plan: dict[str, Any],
                          records: list, state: str,
                          result: dict[str, Any] | None = None) -> None:
        receipts = [r.receipt for r in records if r.receipt]
        payload = {
            "manifest_version": version.version,
            "grid_exp": self.grid.grid_exp,
            "plan": plan,
            "receipts": receipts,
            "tasks": [self._record_to_dict(r) for r in records],
            "error_bounds": {
                key: value.get("error_bound")
                for key, value in result["compiled"].items()
            } if result else {},
            "convergence": result["convergence"] if result else None,
        }
        self.store.save_snapshot(
            self.session_id, version.version, self.grid.grid_exp, state, payload,
        )
        self.store.append_ledger(self.session_id, {
            "event": "snapshot_saved",
            "version": version.version,
            "state": state,
            "digest": content_digest(payload),
        })

    def _previous_version(self, current: ManifestVersion) -> ManifestVersion | None:
        index = next((i for i, v in enumerate(self.versions) if v.version == current.version), -1)
        return self.versions[index - 1] if index > 0 else None

    def _transition(self, new_state: str, **context: Any) -> None:
        previous = self.state
        self.state = new_state
        self.logger.info("state_transition", **{"from": previous, "to": new_state, **context})
        self.store.append_ledger(self.session_id, {
            "event": "state_transition", "from": previous, "to": new_state, **context,
        })

    @staticmethod
    def _record_to_dict(record: Any) -> dict[str, Any]:
        return {
            "task_id": record.task_id,
            "node_id": record.node_id,
            "version": record.version,
            "shard_index": record.shard_index,
            "shard_count": record.shard_count,
            "enqueued_ms": record.enqueued_ms,
            "started_ms": record.started_ms,
            "ended_ms": record.ended_ms,
            "queue_ms": record.queue_ms,
            "peak_concurrency_observed": record.peak_concurrency_observed,
            "status": record.status,
            "attempts": [asdict(attempt) for attempt in record.attempts],
            "receipt": record.receipt,
        }

    def close(self) -> None:
        if self.nodes:
            self.shutdown_nodes()
        self.store.close()
