"""End-to-end local session: compile, dispatch, persist, evolve and recover."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import E_DIVERGE, E_HANDSHAKE, E_MANIFEST, E_REPLAY, PipelineError
from .evolution import EvolutionDelta, build_evolution, diff_shard_views
from .manifest import Manifest, load_manifest
from .planner import Plan, ShardPlan, compile_plan
from .scheduler import ScheduleReport, ScheduleStats, ShardTrace, schedule_shards
from .simulator import SimulatorPool
from .store import SNAPSHOT_FORMAT_VERSION, Store
from .util import atomic_write_json, canonical_json, digest_json, monotonic_ms


@dataclass
class RecoveryInfo:
    snapshot_present: bool = False
    snapshot_generation: str | None = None
    snapshot_replaced: bool = False
    replacement_reasons: tuple[str, ...] = ()
    unknown_shards: tuple[str, ...] = ()
    replayed_shards: tuple[str, ...] = ()
    carried_shards: tuple[str, ...] = ()
    ledger_cursor: int = 0
    ledger_format_version: int = 0
    ledger_content_digest: str = ""

    def canonical(self) -> dict[str, Any]:
        return {
            "snapshot_present": self.snapshot_present,
            "snapshot_generation": self.snapshot_generation,
            "snapshot_replaced": self.snapshot_replaced,
            "replacement_reasons": list(self.replacement_reasons),
            "unknown_shards": list(self.unknown_shards),
            "replayed_shards": list(self.replayed_shards),
            "carried_shards": list(self.carried_shards),
            "ledger_cursor": self.ledger_cursor,
            "ledger_format_version": self.ledger_format_version,
            "ledger_content_digest": self.ledger_content_digest,
        }


@dataclass
class RunReport:
    generation: str
    plan_digest: str
    latest_revision: int
    snapshot_content_digest: str
    shards: tuple[dict[str, Any], ...]
    stats: dict[str, Any]
    invalidations: tuple[dict[str, Any], ...]
    diff: dict[str, Any]
    recovery: RecoveryInfo
    recompute_ids: tuple[str, ...]
    removed_ids: tuple[str, ...]
    evolution: dict[str, Any] | None = None

    @property
    def failed_shards(self) -> tuple[dict[str, Any], ...]:
        return tuple(item for item in self.shards if item["status"] == "failed")

    def canonical(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "plan_digest": self.plan_digest,
            "latest_revision": self.latest_revision,
            "snapshot_content_digest": self.snapshot_content_digest,
            "shards": list(self.shards),
            "stats": self.stats,
            "invalidations": list(self.invalidations),
            "diff": self.diff,
            "recovery": self.recovery.canonical(),
            "recompute_ids": list(self.recompute_ids),
            "removed_ids": list(self.removed_ids),
            "evolution": self.evolution,
            "failed_shard_ids": [
                item["stable_id"] for item in self.failed_shards
            ],
        }


def _nested_difference(
    left: Any, right: Any, prefix: str = ""
) -> tuple[str, Any, Any] | None:
    if type(left) is not type(right):
        return (prefix or "<value>", left, right)
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            field = f"{prefix}.{key}" if prefix else key
            if key not in left:
                return (field, "<missing>", right[key])
            if key not in right:
                return (field, left[key], "<missing>")
            difference = _nested_difference(left[key], right[key], field)
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        for index in range(min(len(left), len(right))):
            difference = _nested_difference(
                left[index], right[index], f"{prefix}[{index}]"
            )
            if difference is not None:
                return difference
        if len(left) != len(right):
            return (
                f"{prefix}[{min(len(left), len(right))}]",
                left,
                right,
            )
        return None
    if left != right:
        return (prefix or "<value>", left, right)
    return None


class PipelineSession:
    def __init__(
        self,
        manifest: Manifest | str | Path | Mapping[str, Any],
        db_path: str | Path,
        *,
        snapshot_path: str | Path | None = None,
        handshake_timeout: float = 5.0,
        dispatch_timeout: float = 30.0,
        scheduler_timeout: float = 120.0,
    ) -> None:
        self.manifest = manifest if isinstance(manifest, Manifest) else load_manifest(manifest)
        self.store = Store(db_path)
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.handshake_timeout = handshake_timeout
        self.dispatch_timeout = dispatch_timeout
        self.scheduler_timeout = scheduler_timeout
        self._ledger_by_shard: dict[str, dict[str, Any]] = {}
        self._ledger_lock = threading.RLock()
        self._closed = False

    @property
    def plan(self) -> Plan:
        return compile_plan(self.manifest)

    def close(self) -> None:
        if not self._closed:
            self.store.close()
            self._closed = True

    def __enter__(self) -> "PipelineSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _snapshot_file_document(self, save_result: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(save_result["payload"])
        return {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "generation": save_result["generation"],
            "content_digest": digest_json(payload),
            "payload": payload,
        }

    def _write_snapshot_file(self, save_result: Mapping[str, Any]) -> None:
        if self.snapshot_path is None:
            return
        atomic_write_json(self.snapshot_path, self._snapshot_file_document(save_result))

    def _snapshot_issues(
        self, snapshot: Mapping[str, Any] | None, expected: Manifest
    ) -> tuple[str, ...]:
        if snapshot is None:
            return ()
        reasons: list[str] = []
        payload = snapshot.get("payload")
        if not isinstance(payload, dict):
            reasons.append("snapshot.payload:not-an-object")
        else:
            payload_digest = digest_json(payload)
            if payload_digest != snapshot.get("content_digest"):
                reasons.append(
                    "snapshot.content_digest:"
                    f"{snapshot.get('content_digest')}->{payload_digest}"
                )
            if payload.get("format_version") != SNAPSHOT_FORMAT_VERSION:
                reasons.append(
                    "snapshot.format_version:"
                    f"{payload.get('format_version')}!={SNAPSHOT_FORMAT_VERSION}"
                )
            expected_versions = {
                "cluster_version": expected.cluster_version,
                "revision_chain_version": expected.revision_chain_version,
                "budget_version": expected.budget_version,
            }
            for field, expected_value in expected_versions.items():
                actual = payload.get(field)
                if actual != expected_value:
                    reasons.append(f"snapshot.{field}:{actual}->{expected_value}")
                row_value = snapshot.get(field)
                if row_value != actual:
                    reasons.append(f"snapshot.row_{field}:{row_value}->{actual}")
            if payload.get("manifest_digest") != expected.manifest_digest:
                reasons.append("snapshot.manifest_digest:changed")
            if int(snapshot.get("ledger_cursor", -1)) != self.store.ledger_cursor:
                reasons.append(
                    "snapshot.ledger_cursor:"
                    f"{snapshot.get('ledger_cursor')}->{self.store.ledger_cursor}"
                )
        if snapshot.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            reasons.append("snapshot.row_format_version:unsupported")

        if self.snapshot_path is not None and self.snapshot_path.exists():
            try:
                document = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                reasons.append(f"snapshot.file:unreadable:{exc}")
            else:
                file_payload = document.get("payload") if isinstance(document, dict) else None
                if not isinstance(file_payload, dict):
                    reasons.append("snapshot.file:payload-missing")
                else:
                    file_digest = digest_json(file_payload)
                    if document.get("content_digest") != file_digest:
                        reasons.append("snapshot.file:content_digest-mismatch")
                    if document.get("generation") != snapshot.get("generation"):
                        reasons.append("snapshot.file:generation-mismatch")
                    if file_digest != snapshot.get("content_digest"):
                        reasons.append("snapshot.file:db-payload-mismatch")
        return tuple(dict.fromkeys(reasons))

    def _current_revision_map(self, snapshot: Mapping[str, Any] | None) -> dict[str, int]:
        if snapshot is None:
            return {}
        payload = snapshot.get("payload")
        if not isinstance(payload, dict):
            return {}
        if payload.get("manifest_digest") != self.manifest.manifest_digest:
            return {}
        revisions: dict[str, int] = {}
        for shard in payload.get("shards", []):
            if not isinstance(shard, dict):
                continue
            stable_id = shard.get("stable_id")
            revision = shard.get("revision_version")
            if isinstance(stable_id, str) and isinstance(revision, int) and not isinstance(revision, bool):
                if revision in self.manifest.revision_by_version:
                    revisions[stable_id] = revision
        return revisions

    def _plan_from_snapshot(self, snapshot: Mapping[str, Any] | None) -> Plan:
        return compile_plan(self.manifest, self._current_revision_map(snapshot))

    def _carry_entries(
        self,
        plan: Plan,
        *,
        unknown_ids: set[str],
    ) -> dict[str, dict[str, Any]]:
        latest = self.store.latest_committed_entries()
        carried: dict[str, dict[str, Any]] = {}
        for shard in plan.shards:
            if shard.stable_id in unknown_ids:
                continue
            entry = latest.get(shard.stable_id)
            if entry is None:
                continue
            if entry["outcome"] != "success":
                continue
            if (
                entry["compile_digest"] != shard.compile_digest
                or entry["revision_version"] != shard.revision_version
            ):
                continue
            summary = entry.get("aggregate_summary")
            if not isinstance(summary, dict) or not isinstance(entry.get("aggregate_digest"), str):
                raise PipelineError(
                    E_REPLAY,
                    "a committed successful ledger entry has no aggregate summary",
                    field="ledger.aggregate_summary",
                    shard_id=shard.stable_id,
                )
            if entry["aggregate_digest"] != digest_json(summary):
                raise PipelineError(
                    E_REPLAY,
                    "ledger aggregate digest does not match its stored summary",
                    field="ledger.aggregate_digest",
                    shard_id=shard.stable_id,
                )
            carried[shard.stable_id] = entry
        return carried

    def _base_shard_payload(
        self,
        shard: ShardPlan,
        *,
        status: str,
        entry: Mapping[str, Any] | None = None,
        trace: ShardTrace | None = None,
        carried: bool = False,
    ) -> dict[str, Any]:
        if trace is not None:
            payload = trace.canonical()
        else:
            payload = {
                "stable_id": shard.stable_id,
                "cluster_id": shard.cluster_id,
                "task_id": shard.task_id,
                "shard_index": shard.shard_index,
                "revision_version": shard.revision_version,
                "resource_units": shard.resource_units,
                "status": status,
                "queued_at_ms": 0,
                "wait_start_ms": 0,
                "wait_end_ms": 0,
                "wait_ms": 0,
                "execution_start_ms": 0,
                "execution_end_ms": 0,
                "duration_ms": 0,
                "service_elapsed_ms": 0,
                "aggregate_summary": None,
                "aggregate_digest": None,
                "error_code": None,
                "error_message": None,
                "thread_name": None,
                "ledger_entry_id": None,
            }
        payload["compile_digest"] = shard.compile_digest
        payload["carried"] = carried
        if entry is not None:
            if trace is None:
                payload["status"] = entry.get("outcome", status)
                payload["aggregate_summary"] = entry.get("aggregate_summary")
                payload["aggregate_digest"] = entry.get("aggregate_digest")
                payload["revision_version"] = int(entry["revision_version"])
                payload["resource_units"] = int(entry["resource_units"])
            payload["ledger_entry_id"] = int(entry["entry_id"])
            if payload.get("ledger_entry_id") is None:
                payload["ledger_entry_id"] = int(entry["entry_id"])
        if payload.get("status") == "success" and not payload.get("aggregate_digest"):
            raise PipelineError(
                E_REPLAY,
                "successful shard is missing its aggregate digest",
                field="aggregate_digest",
                shard_id=shard.stable_id,
            )
        return payload

    def _stats_payload(
        self,
        shards: list[Mapping[str, Any]],
        schedule_report: ScheduleReport | None,
        *,
        phase: str,
    ) -> dict[str, Any]:
        cluster_quotas = {
            cluster.id: cluster.concurrency for cluster in self.manifest.clusters
        }
        task_limits = {task.id: task.concurrency_limit for task in self.manifest.tasks}
        total = sum(int(item.get("resource_units", 0)) for item in shards)
        carried_total = sum(
            int(item.get("resource_units", 0))
            for item in shards
            if item.get("carried")
        )
        result: dict[str, Any] = {
            "phase": phase,
            "global_budget": self.manifest.global_resource_budget,
            "cluster_quotas": {
                key: cluster_quotas[key] for key in sorted(cluster_quotas)
            },
            "task_limits": {key: task_limits[key] for key in sorted(task_limits)},
            "current_resource_units": total,
            "carried_resource_units": carried_total,
            "ledger_total_resource_units": sum(
                int(entry["resource_units"]) for entry in self.store.ledger_entries()
            ),
            "run": None if schedule_report is None else schedule_report.stats.canonical(),
        }
        return result

    def _make_payload(
        self,
        *,
        generation: str,
        plan: Plan,
        shards: list[Mapping[str, Any]],
        stats: Mapping[str, Any],
        invalidations: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
        diff: Mapping[str, Any],
        recovery: RecoveryInfo,
        phase: str,
    ) -> dict[str, Any]:
        return {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "generation": generation,
            "phase": phase,
            "manifest_digest": self.manifest.manifest_digest,
            "latest_revision": self.manifest.latest_revision.version,
            "cluster_version": self.manifest.cluster_version,
            "revision_chain_version": self.manifest.revision_chain_version,
            "budget_version": self.manifest.budget_version,
            "ledger_cursor": self.store.ledger_cursor,
            "plan_digest": plan.plan_digest,
            "manifest": self.manifest.canonical(),
            "plan": plan.canonical(),
            "shards": [
                dict(item) for item in sorted(shards, key=lambda item: str(item["stable_id"]))
            ],
            "aggregate_summaries": {
                str(item["stable_id"]): item["aggregate_summary"]
                for item in sorted(shards, key=lambda item: str(item["stable_id"]))
                if item.get("aggregate_summary") is not None
            },
            "budget": dict(stats),
            "invalidations": [dict(item) for item in invalidations],
            "diff": dict(diff),
            "recovery": recovery.canonical(),
        }

    def _save_payload(
        self,
        *,
        generation: str,
        payload: Mapping[str, Any],
        shards: list[Mapping[str, Any]],
        invalidations: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
        stats: Mapping[str, Any],
        reason: str | None,
    ) -> dict[str, Any]:
        save_result = self.store.save_snapshot(
            generation=generation,
            cluster_version=self.manifest.cluster_version,
            revision_chain_version=self.manifest.revision_chain_version,
            budget_version=self.manifest.budget_version,
            ledger_cursor=self.store.ledger_cursor,
            payload=payload,
            shards=shards,
            invalidations=invalidations,
            stats=stats,
            invalidated_reason=reason,
        )
        self._write_snapshot_file(save_result)
        return save_result

    def _verify_replay(
        self,
        pooled: SimulatorPool,
        plan: Plan,
        carried: Mapping[str, Mapping[str, Any]],
    ) -> tuple[str, ...]:
        replayed: list[str] = []
        for stable_id in sorted(carried):
            shard = plan.by_id.get(stable_id)
            entry = carried[stable_id]
            if shard is None:
                continue
            if entry["revision_version"] != shard.revision_version:
                raise PipelineError(
                    E_REPLAY,
                    "ledger revision does not match the replay plan",
                    shard_id=stable_id,
                    field="revision_version",
                    details={
                        "ledger": entry["revision_version"],
                        "plan": shard.revision_version,
                    },
                )
            expected = entry.get("aggregate_summary")
            if not isinstance(expected, dict):
                raise PipelineError(
                    E_REPLAY,
                    "ledger entry has no replayable aggregate summary",
                    shard_id=stable_id,
                    field="aggregate_summary",
                )
            if entry.get("aggregate_digest") != digest_json(expected):
                raise PipelineError(
                    E_REPLAY,
                    "ledger aggregate digest is inconsistent before replay",
                    shard_id=stable_id,
                    field="aggregate_digest",
                    details={
                        "stored": entry.get("aggregate_digest"),
                        "computed": digest_json(expected),
                    },
                )
            reply = pooled.dispatch(shard, timeout=self.dispatch_timeout)
            actual = reply["aggregate_summary"]
            difference = _nested_difference(expected, actual)
            if difference is not None:
                field, before, after = difference
                raise PipelineError(
                    E_REPLAY,
                    "replayed aggregate summary differs from the committed ledger",
                    shard_id=stable_id,
                    field=f"aggregate_summary.{field}",
                    details={"ledger": before, "replayed": after},
                    basis=[
                        f"ledger_revision={entry['revision_version']}",
                        f"plan_revision={shard.revision_version}",
                        f"compile_digest={shard.compile_digest}",
                    ],
                )
            replayed.append(stable_id)
        return tuple(replayed)

    def _dispatch_with_ledger(
        self,
        pooled: SimulatorPool,
        *,
        generation: str,
        crash_after_completed: int | None,
    ) -> Any:
        completed = 0
        trigger_lock = threading.Lock()
        crashed = False

        def dispatch(shard: ShardPlan) -> dict[str, Any]:
            nonlocal completed, crashed
            with trigger_lock:
                if (
                    crash_after_completed is not None
                    and completed >= crash_after_completed
                    and not crashed
                ):
                    crashed = True
                    self.store.mark_inflight(generation, shard.stable_id)
                    pooled.stop()
                    os._exit(137)
            self.store.mark_inflight(generation, shard.stable_id)
            try:
                reply = pooled.dispatch(shard, timeout=self.dispatch_timeout)
            except PipelineError as exc:
                try:
                    failure = self.store.commit_shard(
                        generation=generation,
                        shard_id=shard.stable_id,
                        outcome="failed",
                        resource_units=shard.resource_units,
                        aggregate_summary=None,
                        aggregate_digest=None,
                        revision_version=shard.revision_version,
                        compile_digest=shard.compile_digest,
                    )
                    with self._ledger_lock:
                        self._ledger_by_shard[shard.stable_id] = failure
                finally:
                    with trigger_lock:
                        completed += 1
                raise
            except Exception as exc:  # noqa: BLE001 - convert to a persisted E_REPLY
                wrapped = PipelineError(
                    E_REPLY,
                    "unexpected dispatch failure",
                    field="dispatch",
                    shard_id=shard.stable_id,
                    details={"reason": f"{type(exc).__name__}: {exc}"},
                )
                try:
                    failure = self.store.commit_shard(
                        generation=generation,
                        shard_id=shard.stable_id,
                        outcome="failed",
                        resource_units=shard.resource_units,
                        aggregate_summary=None,
                        aggregate_digest=None,
                        revision_version=shard.revision_version,
                        compile_digest=shard.compile_digest,
                    )
                    with self._ledger_lock:
                        self._ledger_by_shard[shard.stable_id] = failure
                finally:
                    with trigger_lock:
                        completed += 1
                raise wrapped from exc

            committed = self.store.commit_shard(
                generation=generation,
                shard_id=shard.stable_id,
                outcome="success",
                resource_units=shard.resource_units,
                aggregate_summary=reply["aggregate_summary"],
                aggregate_digest=reply["aggregate_digest"],
                revision_version=shard.revision_version,
                compile_digest=shard.compile_digest,
            )
            with self._ledger_lock:
                self._ledger_by_shard[shard.stable_id] = committed
            with trigger_lock:
                completed += 1
            result = dict(reply)
            result["ledger_entry_id"] = committed["entry_id"]
            return result

        return dispatch

    def _project_view(self, shards: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            str(item["stable_id"]): {
                "revision_version": int(item["revision_version"]),
                "compile_digest": str(item["compile_digest"]),
                "aggregate_digest": item.get("aggregate_digest"),
                "status": item.get("status"),
            }
            for item in shards
        }

    def _validate_consistency(self, plan: Plan, shards: list[Mapping[str, Any]]) -> None:
        plan_ids = set(plan.by_id)
        view_ids = {str(item["stable_id"]) for item in shards}
        if plan_ids != view_ids:
            missing = sorted(plan_ids - view_ids)
            extra = sorted(view_ids - plan_ids)
            raise PipelineError(
                E_DIVERGE,
                "current view and compiled plan are not self-consistent",
                field="shards",
                details={"missing": missing, "extra": extra},
            )
        for item in shards:
            shard = plan.by_id[str(item["stable_id"])]
            if int(item["revision_version"]) not in self.manifest.revision_by_version:
                raise PipelineError(
                    E_DIVERGE,
                    "view refers to a revision absent from the manifest",
                    shard_id=shard.stable_id,
                    field="revision_version",
                )
            if item["compile_digest"] != shard.compile_digest:
                raise PipelineError(
                    E_DIVERGE,
                    "view compile digest differs from the compiled plan",
                    shard_id=shard.stable_id,
                    field="compile_digest",
                    details={
                        "view": item["compile_digest"],
                        "plan": shard.compile_digest,
                    },
                )
            if item.get("status") == "success":
                summary = item.get("aggregate_summary")
                digest = item.get("aggregate_digest")
                if not isinstance(summary, dict) or digest != digest_json(summary):
                    raise PipelineError(
                        E_REPLAY,
                        "current view aggregate summary is incomplete or inconsistent",
                        shard_id=shard.stable_id,
                        field="aggregate_summary",
                    )

    def _execute(
        self,
        *,
        plan: Plan,
        delta: EvolutionDelta | None,
        recovery_issues: tuple[str, ...],
        verify_replay: bool,
        crash_after_completed: int | None,
        old_view: dict[str, dict[str, Any]],
        prior_snapshot: Mapping[str, Any] | None = None,
    ) -> RunReport:
        unknown_records = self.store.mark_recovery_unknown()
        unknown_ids = {str(item["shard_id"]) for item in unknown_records}
        for item in self.store.current_view():
            if item.get("status") in {"running", "unknown"}:
                unknown_ids.add(str(item["stable_id"]))

        carried = self._carry_entries(plan, unknown_ids=unknown_ids)
        current_invalidations_before = self.store.current_invalidations()
        snapshot_before = (
            prior_snapshot if prior_snapshot is not None else self.store.latest_snapshot()
        )
        recovery = RecoveryInfo(
            snapshot_present=snapshot_before is not None,
            snapshot_generation=(
                None if snapshot_before is None else str(snapshot_before["generation"])
            ),
            snapshot_replaced=bool(recovery_issues)
            or (delta is not None and snapshot_before is not None),
            replacement_reasons=tuple(recovery_issues),
            unknown_shards=tuple(sorted(unknown_ids)),
            ledger_cursor=self.store.ledger_cursor,
            ledger_format_version=self.store.ledger_format_version,
            ledger_content_digest=self.store.ledger_content_digest,
        )
        if delta is not None:
            recovery.replacement_reasons = tuple(
                dict.fromkeys(
                    tuple(recovery.replacement_reasons)
                    + (
                        "revision_evolution:"
                        f"{self.manifest.latest_revision.version}",
                    )
                )
            )

        generation = self.store.next_generation_id()
        initial_shards: list[dict[str, Any]] = []
        for shard in plan.shards:
            entry = carried.get(shard.stable_id)
            status = "success" if entry is not None else (
                "unknown" if shard.stable_id in unknown_ids else "pending"
            )
            initial_shards.append(
                self._base_shard_payload(
                    shard,
                    status=status,
                    entry=entry,
                    carried=entry is not None,
                )
            )
        if delta is not None:
            invalidations = tuple(
                {
                    "revision_version": item.revision_version,
                    "shard_id": item.stable_id,
                    "stable_id": item.stable_id,
                    "kind": item.kind,
                    "basis": list(item.basis),
                }
                for item in delta.invalidations
            )
        else:
            invalidations = current_invalidations_before
        initial_diff = diff_shard_views(
            old_view, self._project_view(initial_shards)
        )
        initial_stats = self._stats_payload(initial_shards, None, phase="prepared")
        initial_payload = self._make_payload(
            generation=generation,
            plan=plan,
            shards=initial_shards,
            stats=initial_stats,
            invalidations=invalidations,
            diff=initial_diff,
            recovery=recovery,
            phase="prepared",
        )
        self._save_payload(
            generation=generation,
            payload=initial_payload,
            shards=initial_shards,
            invalidations=invalidations,
            stats=initial_stats,
            reason=(
                "snapshot mismatch: " + "; ".join(recovery_issues)
                if recovery_issues
                else None
            ),
        )

        pending_shards = [
            shard for shard in plan.shards if shard.stable_id not in carried
        ]
        clusters_needed = sorted(
            {shard.cluster_id for shard in pending_shards}
            | ({shard.cluster_id for shard in plan.shards} if verify_replay else set())
        )
        clusters = [self.manifest.cluster_by_id[item] for item in clusters_needed]
        schedule_report: ScheduleReport | None = None
        replayed: tuple[str, ...] = ()
        if clusters:
            pooled = SimulatorPool(clusters, handshake_timeout=self.handshake_timeout)
            pooled.start()
            try:
                if verify_replay:
                    replayed = self._verify_replay(pooled, plan, carried)
                    recovery.replayed_shards = replayed
                if pending_shards:
                    subset = Plan(
                        manifest_digest=plan.manifest_digest,
                        shards=tuple(pending_shards),
                        plan_digest=plan.plan_digest,
                    )
                    dispatcher = self._dispatch_with_ledger(
                        pooled,
                        generation=generation,
                        crash_after_completed=crash_after_completed,
                    )
                    schedule_report = schedule_shards(
                        subset,
                        dispatcher,
                        global_budget=self.manifest.global_resource_budget,
                        cluster_quotas={
                            cluster.id: cluster.concurrency
                            for cluster in self.manifest.clusters
                        },
                        timeout_seconds=self.scheduler_timeout,
                    )
            finally:
                pooled.stop()

        latest_entries = self.store.latest_committed_entries()
        trace_by_id = (
            {trace.stable_id: trace for trace in schedule_report.traces}
            if schedule_report is not None
            else {}
        )
        final_shards: list[dict[str, Any]] = []
        for shard in plan.shards:
            trace = trace_by_id.get(shard.stable_id)
            if trace is not None:
                entry = latest_entries.get(shard.stable_id)
                final_shards.append(
                    self._base_shard_payload(
                        shard,
                        status=trace.status,
                        entry=entry,
                        trace=trace,
                        carried=False,
                    )
                )
                continue
            entry = carried.get(shard.stable_id)
            final_shards.append(
                self._base_shard_payload(
                    shard,
                    status="success",
                    entry=entry,
                    carried=True,
                )
            )
        self._validate_consistency(plan, final_shards)
        recovery.carried_shards = tuple(sorted(carried))
        recovery.ledger_cursor = self.store.ledger_cursor
        recovery.ledger_content_digest = self.store.ledger_content_digest

        final_diff = diff_shard_views(old_view, self._project_view(final_shards))
        final_stats = self._stats_payload(final_shards, schedule_report, phase="complete")
        final_payload = self._make_payload(
            generation=generation,
            plan=plan,
            shards=final_shards,
            stats=final_stats,
            invalidations=invalidations,
            diff=final_diff,
            recovery=recovery,
            phase="complete",
        )
        save_result = self._save_payload(
            generation=generation,
            payload=final_payload,
            shards=final_shards,
            invalidations=invalidations,
            stats=final_stats,
            reason=None,
        )
        recompute_ids = (
            delta.recompute_ids
            if delta is not None
            else tuple(sorted(item.stable_id for item in pending_shards))
        )
        removed_ids = () if delta is None else delta.removed_ids
        return RunReport(
            generation=generation,
            plan_digest=plan.plan_digest,
            latest_revision=self.manifest.latest_revision.version,
            snapshot_content_digest=str(save_result["content_digest"]),
            shards=tuple(
                sorted(final_shards, key=lambda item: str(item["stable_id"]))
            ),
            stats=final_stats,
            invalidations=tuple(invalidations),
            diff=final_diff,
            recovery=recovery,
            recompute_ids=recompute_ids,
            removed_ids=removed_ids,
            evolution=None if delta is None else delta.canonical(),
        )

    def run_current(
        self,
        *,
        verify_replay: bool = False,
        crash_after_completed: int | None = None,
    ) -> RunReport:
        snapshot = self.store.latest_snapshot()
        issues = self._snapshot_issues(snapshot, self.manifest)
        old_view = self._project_view(list(self.store.current_view()))
        if issues:
            self.store.invalidate_current_view()
        plan = self._plan_from_snapshot(snapshot)
        return self._execute(
            plan=plan,
            delta=None,
            recovery_issues=issues,
            verify_replay=verify_replay,
            crash_after_completed=crash_after_completed,
            old_view=old_view,
            prior_snapshot=snapshot,
        )

    def recover(self, *, crash_after_completed: int | None = None) -> RunReport:
        return self.run_current(
            verify_replay=True,
            crash_after_completed=crash_after_completed,
        )

    def evolve(
        self,
        new_manifest: Manifest | str | Path | Mapping[str, Any],
        *,
        verify_replay: bool = False,
        crash_after_completed: int | None = None,
    ) -> RunReport:
        evolved = new_manifest if isinstance(new_manifest, Manifest) else load_manifest(new_manifest)
        old_manifest = self.manifest
        snapshot = self.store.latest_snapshot()
        issues = self._snapshot_issues(snapshot, old_manifest)
        old_view = self._project_view(list(self.store.current_view()))
        if issues:
            self.store.invalidate_current_view()
        old_plan = self._plan_from_snapshot(snapshot)
        delta = build_evolution(old_manifest, evolved, old_plan=old_plan)
        self.manifest = evolved
        report = self._execute(
            plan=delta.new_plan,
            delta=delta,
            recovery_issues=issues,
            verify_replay=verify_replay,
            crash_after_completed=crash_after_completed,
            old_view=old_view,
            prior_snapshot=snapshot,
        )
        report.evolution = delta.canonical()
        return report

    def replay(self) -> dict[str, Any]:
        """Recompute every current successful shard and compare with the ledger."""

        snapshot = self.store.latest_snapshot()
        issues = self._snapshot_issues(snapshot, self.manifest)
        if issues:
            self.store.invalidate_current_view()
        plan = self._plan_from_snapshot(snapshot)
        unknown_ids = {str(item["shard_id"]) for item in self.store.mark_recovery_unknown()}
        carried = self._carry_entries(plan, unknown_ids=unknown_ids)
        clusters = sorted({shard.cluster_id for shard in plan.shards})
        pooled = SimulatorPool(
            [self.manifest.cluster_by_id[item] for item in clusters],
            handshake_timeout=self.handshake_timeout,
        )
        pooled.start()
        try:
            replayed = self._verify_replay(pooled, plan, carried)
        finally:
            pooled.stop()
        return {
            "ok": True,
            "replayed_shards": list(replayed),
            "ledger_cursor": self.store.ledger_cursor,
            "ledger_content_digest": self.store.ledger_content_digest,
        }
