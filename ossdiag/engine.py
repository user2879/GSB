from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import ThroughputBudget
from .content import LEAF_SIZE
from .digest import leaf_digest, make_aggregator, object_digest
from .errors import DiagnosticError, DigestError, LeaseError, SnapshotError, SpecError
from .events import EventLogger, monotonic_ms
from .fixedpoint import FixedPoint, ObjectNumericStats, compute_object_stats, rank_objects_by_completion
from .lease import LeaseManager
from .ledger import Ledger, ReconcileResult
from .models import SessionSpec, load_spec
from .planner import PathSelection, choose_path, selection_from_stored
from .scheduler import FaultPlan, ShardScheduler, ShardTask


@dataclass
class RunOptions:
    inject_lease_overflow: bool = False
    inject_digest_mismatch: bool = False
    inject_snapshot_conflict: bool = False
    target_object_key: str | None = None
    target_shard_index: int | None = None
    crash_after_batches: int | None = None


@dataclass
class RunSummary:
    ok: bool
    objects: list[dict[str, Any]]
    ranking: list[dict[str, Any]]
    ranking_affected_pairs: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    reconcile: dict[str, Any]
    throughput: dict[str, int]
    leases: dict[str, Any]
    path_verification: list[dict[str, Any]]
    ledger: dict[str, Any]
    tasks: list[dict[str, Any]]
    note: str = "强制终止时的退出报告不可用；恢复结论仅来自已提交账本批次。"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "objects": self.objects,
            "ranking": self.ranking,
            "ranking_affected_pairs": self.ranking_affected_pairs,
            "errors": self.errors,
            "reconcile": self.reconcile,
            "throughput": self.throughput,
            "leases": self.leases,
            "path_verification": self.path_verification,
            "ledger": self.ledger,
            "tasks": self.tasks,
            "note": self.note,
        }


class Workbench:
    """Single-process orchestration for one deterministic diagnostic session."""

    def __init__(
        self,
        session: SessionSpec,
        *,
        ledger_path: str | Path,
        data_dir: str | Path,
        log_path: str | Path,
        backup_dir: str | Path | None = None,
    ) -> None:
        self.session = session
        self.ledger_path = Path(ledger_path)
        self.data_dir = Path(data_dir)
        self.log_path = Path(log_path)
        self.backup_dir = Path(backup_dir) if backup_dir else self.ledger_path.parent / "ledger_backups"
        self.logger = EventLogger(self.log_path)
        self.ledger: Ledger | None = None
        self.leases: LeaseManager | None = None
        self.selections: dict[str, PathSelection] = {}
        self.reconcile_result: ReconcileResult | None = None
        self.last_tasks: list[ShardTask] = []
        self.last_errors: list[dict[str, Any]] = []
        self.last_verification: dict[str, dict[str, Any]] = {}
        self.last_path_verification: list[dict[str, Any]] = []
        self.scale_exponent = session.scale_exponent
        self._initialized = False
        self._run_lock = threading.Lock()

    @classmethod
    def from_spec(cls, spec_path: str | Path, **kwargs: Any) -> "Workbench":
        return cls(load_spec(spec_path), **kwargs)

    def initialize(self, *, force_snapshot_conflict: bool = False) -> ReconcileResult:
        if self._initialized:
            if not force_snapshot_conflict:
                assert self.reconcile_result is not None
                return self.reconcile_result
            self._discard_open_ledger()
        self.ledger = Ledger(self.ledger_path, self.data_dir, self.backup_dir)
        result = self.ledger.reconcile(self.session, force_snapshot_conflict=force_snapshot_conflict)
        self.reconcile_result = result
        if result.replaced:
            target_key, target_shard = self._snapshot_target(result)
            error = SnapshotError(
                "账本格式版本/内容摘要/磁盘分片不一致，已整体替换并保留只读备份",
                object_key=target_key,
                shard_index=target_shard,
                detail={
                    "reason": result.reason,
                    "snapshot_id": result.snapshot_id,
                    "backup_path": result.backup_path,
                    "replaced_shards": [
                        {"object_key": key, "shard_index": index} for key, index in result.replaced_shards
                    ],
                    "issues": list(result.issues),
                },
            )
            self.logger.business_error(error)
            self.last_errors.append({"monotonic_ms": monotonic_ms(), **error.to_dict()})

        self.selections = {}
        stored_exponents: set[int] = set()
        for spec in self.session.objects:
            stored = self.ledger.object_row(spec.object_key)
            if stored is not None and stored.get("scale_exponent") is not None:
                stored_exponents.add(int(stored["scale_exponent"]))
            if stored is not None and stored.get("selected_path"):
                self.selections[spec.object_key] = selection_from_stored(
                    spec,
                    str(stored["selected_path"]),
                    stored.get("path_reason"),
                    stored.get("path_scores_json"),
                    bool(stored.get("tie_boundary")),
                    self.session.tie_threshold_work,
                )
            else:
                self.selections[spec.object_key] = choose_path(spec, self.session.tie_threshold_work)
        if len(stored_exponents) == 1:
            self.scale_exponent = next(iter(stored_exponents))
        else:
            self.scale_exponent = self.session.scale_exponent
        self.ledger.prepare_session(self.session, self.selections)
        self.leases = LeaseManager(
            global_quota=self.session.global_lease_quota,
            sink=self.ledger,
            event_callback=self._lease_event,
            initial_version=self.ledger.max_lease_version(),
        )
        self._initialized = True
        self.logger.emit(
            "session_opened",
            ledger_path=str(self.ledger_path.resolve()),
            ledger_digest=self.ledger.meta().get("ledger_digest"),
            format_version=self.ledger.meta().get("format_version"),
        )
        return result

    def _discard_open_ledger(self) -> None:
        if self.leases is not None:
            try:
                self.leases.reclaim_all("reopened_for_snapshot_injection")
            finally:
                self.leases = None
        if self.ledger is not None:
            try:
                self.ledger.close()
            finally:
                self.ledger = None
        self._initialized = False

    def _snapshot_target(self, result: ReconcileResult) -> tuple[str | None, int | None]:
        if result.replaced_shards:
            return result.replaced_shards[0]
        if self.session.objects:
            return self.session.objects[0].object_key, 0
        return None, None

    def _lease_event(self, payload: dict[str, Any]) -> None:
        event = str(payload.pop("event"))
        self.logger.emit(event, **payload)

    def close(self) -> None:
        if self.ledger is None and self.leases is None:
            return
        if self.leases is not None:
            try:
                self.leases.reclaim_all("reclaimed_on_close")
            finally:
                self.leases = None
        if self.ledger is not None:
            try:
                self.ledger.close()
            finally:
                self.ledger = None
        self._initialized = False

    def __enter__(self) -> "Workbench":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _committed_bytes(self) -> int:
        assert self.ledger is not None
        total = 0
        for row in self.ledger.all_shard_rows():
            total += max(0, int(row["progress_cursor"]) - int(row["byte_start"]))
        return total

    def _inject_lease_overflow(self) -> DiagnosticError | None:
        assert self.leases is not None
        held: list[Any] = []
        failure: LeaseError | None = None
        try:
            candidates = sorted(self.session.objects, key=lambda item: item.object_key)
            for spec in candidates:
                object_held = 0
                for shard_index in range(0, min(spec.lease_policy.max_concurrent_leases, spec.shard_count)):
                    try:
                        held.append(self.leases.grant(spec.object_key, shard_index, spec.lease_policy))
                        object_held += 1
                    except LeaseError:
                        break
                probe_index = object_held
                while probe_index < spec.shard_count:
                    try:
                        held.append(
                            self.leases.grant(spec.object_key, probe_index, spec.lease_policy)
                        )
                        probe_index += 1
                    except LeaseError as exc:
                        failure = LeaseError(
                            "注入的租约越界申请被拒绝",
                            object_key=spec.object_key,
                            shard_index=probe_index,
                            detail={"injected": True, **exc.detail},
                        )
                        break
                if failure is not None:
                    break
        finally:
            for lease in reversed(held):
                self.leases.release(lease, "injection_probe_released")
        if failure is not None:
            self.logger.business_error(failure)
            return failure
        failure = LeaseError(
            "无法构造租约越界探针，样本分片/配额不足",
            object_key=self.session.objects[0].object_key if self.session.objects else None,
            shard_index=0,
            detail={"injected": True, "reason": "insufficient_probe_capacity"},
        )
        self.logger.business_error(failure)
        return failure

    def _post_completion_digest_probe(
        self, target_object_key: str | None, target_shard_index: int | None
    ) -> DigestError | None:
        assert self.ledger is not None
        candidates: list[tuple[Any, dict[str, Any]]] = []
        for spec in sorted(self.session.objects, key=lambda item: item.object_key):
            for shard in self.ledger.shard_rows(spec.object_key):
                if shard["status"] == "complete" and shard["content_digest"]:
                    candidates.append((spec, shard))
        if not candidates:
            return None
        selected = candidates[0]
        if target_object_key is not None:
            filtered = [item for item in candidates if item[0].object_key == target_object_key]
            if filtered:
                selected = filtered[0]
                if target_shard_index is not None:
                    selected = next(
                        (item for item in filtered if int(item[1]["shard_index"]) == target_shard_index),
                        filtered[0],
                    )
        spec, shard = selected
        shard_index = int(shard["shard_index"])
        path = self.ledger.shard_path(spec.object_key, shard_index)
        length = min(LEAF_SIZE, int(shard["byte_end"]) - int(shard["byte_start"]))
        data = path.read_bytes()[:length]
        expected = leaf_digest(spec.object_key, shard_index, 0, length, data).hex()
        observed = "00" * 32
        error = DigestError(
            "完成态摘要探针检测到不符（参数注入）",
            object_key=spec.object_key,
            shard_index=shard_index,
            detail={
                "injected": True,
                "mode": "post_completion_probe",
                "leaf_index": 0,
                "expected": expected,
                "actual": observed,
            },
        )
        self.logger.business_error(error)
        return error

    def run(self, options: RunOptions | None = None) -> RunSummary:
        options = options or RunOptions()
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("a diagnostic run is already active")
        try:
            self.last_errors = []
            self.initialize(force_snapshot_conflict=options.inject_snapshot_conflict)
            assert self.ledger is not None
            assert self.leases is not None
            errors: list[dict[str, Any]] = list(self.last_errors)

            if options.inject_lease_overflow:
                injected = self._inject_lease_overflow()
                if injected is not None:
                    errors.append({"monotonic_ms": monotonic_ms(), **injected.to_dict()})

            initial_used = self._committed_bytes()
            budget = ThroughputBudget(self.session.throughput_budget_bytes, initial_used_bytes=initial_used)
            faults = FaultPlan(
                digest_mismatch=options.inject_digest_mismatch,
                target_object_key=options.target_object_key,
                target_shard_index=options.target_shard_index,
                crash_after_batches=options.crash_after_batches,
            )
            scheduler = ShardScheduler(
                session=self.session,
                selections=self.selections,
                ledger=self.ledger,
                leases=self.leases,
                budget=budget,
                logger=self.logger,
                faults=faults,
            )
            self.last_tasks = scheduler.run()
            for task in self.last_tasks:
                if task.error is not None:
                    errors.append({"monotonic_ms": monotonic_ms(), **task.error.to_dict()})

            verification: dict[str, dict[str, Any]] = {}
            for spec in self.session.objects:
                shard_rows = self.ledger.shard_rows(spec.object_key)
                if shard_rows and all(row["status"] == "complete" for row in shard_rows):
                    checked = self.ledger.verify_completed_object(spec.object_key)
                    if checked.get("ok") is False and checked.get("reason") == "object_digest_mismatch" and checked.get(
                        "ledger_object_digest"
                    ) is None:
                        self.ledger.mark_object_digest(spec.object_key, str(checked["object_digest"]))
                        checked = self.ledger.verify_completed_object(spec.object_key)
                    verification[spec.object_key] = checked
                    if not checked.get("ok", False):
                        digest_error = DigestError(
                            "对象最终摘要校验失败",
                            object_key=spec.object_key,
                            shard_index=checked.get("shard_index"),
                            detail=checked,
                        )
                        self.logger.business_error(digest_error)
                        errors.append({"monotonic_ms": monotonic_ms(), **digest_error.to_dict()})
                else:
                    verification[spec.object_key] = {
                        "ok": False,
                        "reason": "incomplete_shards",
                        "completed": sum(1 for row in shard_rows if row["status"] == "complete"),
                        "total": len(shard_rows),
                    }
            self.last_verification = verification
            self.last_path_verification = self._verify_paths()
            if options.inject_digest_mismatch and not any(item.get("code") == "E_DIGEST" for item in errors):
                probe_error = self._post_completion_digest_probe(
                    options.target_object_key, options.target_shard_index
                )
                if probe_error is not None:
                    errors.append({"monotonic_ms": monotonic_ms(), **probe_error.to_dict()})
            dashboard = self.build_dashboard()
            objects = dashboard["objects"]
            all_complete = all(item["status"] == "complete" for item in objects)
            ok = all_complete and not errors and all(item.get("digest_status") == "PASS" for item in objects)
            summary = RunSummary(
                ok=ok,
                objects=objects,
                ranking=dashboard["ranking"],
                ranking_affected_pairs=dashboard["ranking_affected_pairs"],
                errors=errors,
                reconcile=self.reconcile_result.as_dict() if self.reconcile_result else {},
                throughput=budget.snapshot(),
                leases=self.leases.snapshot(),
                path_verification=self.last_path_verification,
                ledger=dashboard["ledger"],
                tasks=[task.as_dict() for task in self.last_tasks],
            )
            self.last_errors = errors
            self.logger.emit(
                "run_finished",
                ok=ok,
                error_count=len(errors),
                constructed_bytes=budget.used_bytes,
                max_active_leases=self.leases.snapshot()["peak_global"],
            )
            return summary
        finally:
            self._run_lock.release()

    def _verify_paths(self) -> list[dict[str, Any]]:
        assert self.ledger is not None
        results: list[dict[str, Any]] = []
        for spec in sorted(self.session.objects, key=lambda item: item.object_key):
            object_row = self.ledger.object_row(spec.object_key)
            shards = self.ledger.shard_rows(spec.object_key)
            if object_row is None or not shards or any(row["status"] != "complete" for row in shards):
                continue
            per_path: dict[str, Any] = {}
            digests: list[str] = []
            for path in ("direct", "sliding", "hierarchical"):
                descriptors: list[tuple[int, int, int, bytes]] = []
                shard_roots: dict[str, str] = {}
                for shard in shards:
                    leaves = [bytes.fromhex(value) for value in self.ledger.leaf_hex_values(spec.object_key, int(shard["shard_index"]))]
                    aggregator = make_aggregator(path, leaves)
                    root = aggregator.finalize()
                    shard_roots[str(shard["shard_index"])] = root.hex()
                    descriptors.append(
                        (int(shard["shard_index"]), int(shard["byte_start"]), int(shard["byte_end"]), root)
                    )
                digest = object_digest(spec.object_key, spec.total_bytes, descriptors).hex()
                digests.append(digest)
                per_path[path] = {"shard_roots": shard_roots, "object_digest": digest}
            stored = object_row.get("object_digest")
            consistent = len(set(digests)) == 1 and stored == digests[0]
            results.append(
                {
                    "object_key": spec.object_key,
                    "selected_path": object_row["selected_path"],
                    "consistent": consistent,
                    "stored_object_digest": stored,
                    "paths": per_path,
                }
            )
        return results

    def _object_stats(self) -> list[ObjectNumericStats]:
        assert self.ledger is not None
        fixed = FixedPoint(self.scale_exponent)
        stats: list[ObjectNumericStats] = []
        for spec in sorted(self.session.objects, key=lambda item: item.object_key):
            grants, renews = self.ledger.lease_stats(spec.object_key)
            stats.append(
                compute_object_stats(
                    object_key=spec.object_key,
                    total_bytes=spec.total_bytes,
                    shard_rows=self.ledger.shard_rows(spec.object_key),
                    fixed=fixed,
                    lease_grants=grants,
                    renew_count=renews,
                )
            )
        return stats

    def build_dashboard(self) -> dict[str, Any]:
        if not self._initialized:
            self.initialize()
        assert self.ledger is not None
        assert self.leases is not None
        fixed = FixedPoint(self.scale_exponent)
        specs = sorted(self.session.objects, key=lambda item: item.object_key)
        objects: list[dict[str, Any]] = []
        all_stats = self._object_stats()
        stats_by_key = {item.object_key: item for item in all_stats}
        lease_snapshot = self.leases.snapshot(expire=False)
        for spec in specs:
            row = self.ledger.object_row(spec.object_key)
            shards = self.ledger.shard_rows(spec.object_key)
            stats = stats_by_key[spec.object_key]
            check = self.last_verification.get(spec.object_key)
            all_complete = bool(shards) and all(shard["status"] == "complete" for shard in shards)
            unknown_count = sum(1 for shard in shards if shard["status"] == "unknown" or int(shard["unknown"]))
            if all_complete and check is not None and check.get("ok"):
                digest_status = "PASS"
            elif unknown_count:
                digest_status = "UNKNOWN"
            elif row and row.get("status") == "failed":
                digest_status = "FAIL"
            elif all_complete and row and row.get("object_digest"):
                digest_status = "PASS"
            elif any(shard["status"] in ("uploading", "pending", "unknown") for shard in shards):
                digest_status = "IN_PROGRESS"
            else:
                digest_status = "FAIL"
            active_object = int(lease_snapshot["per_object_active"].get(spec.object_key, 0))
            object_limit = spec.effective_concurrency
            shard_view = []
            for shard in shards:
                shard_view.append(
                    {
                        "object_key": spec.object_key,
                        "shard_index": int(shard["shard_index"]),
                        "byte_start": int(shard["byte_start"]),
                        "byte_end": int(shard["byte_end"]),
                        "byte_length": int(shard["byte_end"]) - int(shard["byte_start"]),
                        "status": str(shard["status"]),
                        "unknown": bool(shard["unknown"]),
                        "progress_cursor": int(shard["progress_cursor"]),
                        "progress_fixed": fixed.ratio(
                            max(0, int(shard["progress_cursor"]) - int(shard["byte_start"])),
                            max(1, int(shard["byte_end"]) - int(shard["byte_start"])),
                        ),
                        "content_digest": shard["content_digest"],
                        "partial_digest": shard["partial_digest"],
                        "lease_version": int(shard["lease_version"]),
                        "renew_count": int(shard["renew_count"]),
                        "completed_mono_ms": shard["completed_mono_ms"],
                        "attempt": int(shard["attempt"]),
                    }
                )
            path_estimate = self.selections[spec.object_key]
            objects.append(
                {
                    "object_key": spec.object_key,
                    "total_bytes": spec.total_bytes,
                    "shard_size": spec.shard_size,
                    "batch_size": spec.batch_size,
                    "status": str(row["status"]) if row else "pending",
                    "selected_path": str(row["selected_path"]) if row else path_estimate.path,
                    "path_reason": str(row["path_reason"]) if row and row.get("path_reason") else path_estimate.reason,
                    "boundary_case": bool(row["tie_boundary"]) if row else path_estimate.boundary_case,
                    "path_scores": json.loads(row["path_scores_json"]) if row and row.get("path_scores_json") else [],
                    "tie_break_rule": path_estimate.tie_break_rule,
                    "digest_status": digest_status,
                    "object_digest": row.get("object_digest") if row else None,
                    "digest_verification": check,
                    "unknown_shards": unknown_count,
                    "lease_watermark": {
                        "global_active": int(lease_snapshot["active_global"]),
                        "global_limit": self.session.global_lease_quota,
                        "global_remaining": int(lease_snapshot["remaining_global"]),
                        "object_active": active_object,
                        "object_limit": object_limit,
                        "object_remaining": max(0, object_limit - active_object),
                        "peak_global": int(lease_snapshot["peak_global"]),
                        "peak_object": int(lease_snapshot["peak_per_object"].get(spec.object_key, 0)),
                    },
                    "fixed_stats": stats.as_dict(fixed),
                    "shards": shard_view,
                }
            )
        ranking, affected_pairs = rank_objects_by_completion(all_stats, fixed)
        return {
            "generated_mono_ms": monotonic_ms(),
            "scale_policy": fixed.policy(),
            "throughput": {
                "limit_bytes": self.session.throughput_budget_bytes,
                "committed_bytes": self._committed_bytes(),
                "remaining_bytes": max(0, self.session.throughput_budget_bytes - self._committed_bytes()),
            },
            "ledger": {
                "path": str(self.ledger_path.resolve()),
                "format_version": int(self.ledger.meta().get("format_version", "0")),
                "ledger_digest": self.ledger.meta().get("ledger_digest"),
                "recomputed_digest": self.ledger.ledger_digest(),
                "snapshots": self.ledger.list_snapshots(),
                "replacement_shards": {
                    str(snapshot["snapshot_id"]): self.ledger.replacement_shards(int(snapshot["snapshot_id"]))
                    for snapshot in self.ledger.list_snapshots()
                },
            },
            "objects": objects,
            "ranking": ranking,
            "ranking_affected_pairs": affected_pairs,
        }
