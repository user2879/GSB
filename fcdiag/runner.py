"""End-to-end deterministic runner producing reviewable reuse/recompute ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .errors import E_ARTIFACT_BUDGET_PENDING, E_STAGE_BUDGET_EXCEEDED
from .invalidation import CacheIndex
from .models import Artifact, LedgerEntry, ReceiptRecord
from .pipeline import FixedPipeline, artifact_id, cache_key
from .receipts import downgrade_v2_to_v1, downgrade_v3_to_v2, upgrade_v1_to_v2, upgrade_v2_to_v3
from .scheduler import Scheduler, Task, stable_sort_key
from .snapshot import SnapshotStore
from .state import WorkState


@dataclass
class RunResult:
    state: WorkState
    artifacts: Dict[str, Artifact]
    cache_keys: Dict[str, str]
    ledger: Tuple[LedgerEntry, ...]
    events: Tuple[dict, ...]
    failures: Tuple[dict, ...]
    conversion_diffs: Tuple[dict, ...]
    receipts: Tuple[ReceiptRecord, ...]
    quant_logs: Dict[str, Tuple[dict, ...]]
    pending: Tuple[Task, ...]
    budget_used: Dict[str, int]
    resume_decision: Optional[dict]


def expected_artifact_param_version(artifact: Artifact) -> int:
    stage = artifact.stage
    return sum(
        version
        for key, version in artifact.param_versions.items()
        if key.startswith(f"{stage}.")
    )


def normalize_receipts_to_version(
    receipts: Tuple[ReceiptRecord, ...],
    target_version: int,
    current_version: int,
    global_param_version: int,
) -> Tuple[Tuple[ReceiptRecord, ...], Tuple[dict, ...]]:
    if current_version == target_version:
        return receipts, ()
    records = receipts
    diffs: List[dict] = []
    version = current_version
    while version < target_version:
        if version == 1:
            result = upgrade_v1_to_v2(records, global_param_version)
        else:
            result = upgrade_v2_to_v3(records)
        records = result.records
        diffs.extend(result.diffs)
        version += 1
    while version > target_version:
        if version == 2:
            result = downgrade_v2_to_v1(records)
        else:
            result = downgrade_v3_to_v2(records)
        records = result.records
        diffs.extend(result.diffs)
        version -= 1
    diffs.sort(key=lambda item: (item["artifact_id"], item["field"]))
    return records, tuple(diffs)


class Runner:
    def __init__(
        self,
        state: WorkState,
        receipts: Tuple[ReceiptRecord, ...],
        baseline_artifacts: Optional[Dict[str, Artifact]] = None,
        snapshot_store: Optional[SnapshotStore] = None,
        max_workers: int = 1,
    ):
        self.state = state
        self.baseline_artifacts = dict(baseline_artifacts or {})
        self.snapshot_store = snapshot_store
        self.max_workers = max_workers
        active_receipts, diffs = normalize_receipts_to_version(
            receipts,
            state.receipt_version,
            receipts[0].source_version if receipts else state.receipt_version,
            state.global_param_version(),
        )
        self.active_receipts = active_receipts
        self.conversion_diffs = diffs
        self.cache_index = CacheIndex(active_receipts, state.config.policy.degraded_reuse)
        self.computer = FixedPipeline(state.pipeline, state.config.fixed)
        self.sample_order = {sample.sample_id: idx for idx, sample in enumerate(state.samples)}
        self.artifacts: Dict[str, Artifact] = {}
        self.cache_keys: Dict[str, str] = {}
        self.ledger: List[LedgerEntry] = []
        self.events: List[dict] = []
        self.failures: List[dict] = []
        self.quant_logs: Dict[str, Tuple[dict, ...]] = {}
        self.pending: Tuple[Task, ...] = ()
        self.resume_decision: Optional[dict] = None
        self.budget_used: Dict[str, int] = {}
        self.candidates: Dict[str, Artifact] = {}
        self.sample_fingerprint_changed = False

    def run(self, max_batches: Optional[int] = None) -> RunResult:
        if self.snapshot_store is not None:
            decision = self.snapshot_store.resume_check(
                self.state.sequence,
                self.state.config_hash(),
                self.state.sample_fingerprint(),
                self.state.receipt_fingerprint(),
            )
            self.resume_decision = {
                "resume": decision.resume,
                "reason": decision.reason,
                "code": decision.code,
            }
        raw_artifacts = self.computer.raw_artifacts(self.state.samples)
        for artifact in raw_artifacts:
            self.artifacts[artifact.artifact_id] = artifact
            self.candidates[artifact.artifact_id] = artifact
            self.cache_keys[artifact.artifact_id] = cache_key(artifact.artifact_id, artifact.digest)
            self._raw_ledger_entry(artifact)
        tasks = self._build_tasks(raw_artifacts)
        scheduler = Scheduler(
            self.state.config.budget.batch_points,
            self.state.config.budget.stage_quotas,
            self.max_workers,
        )
        carry_pending: Tuple[Task, ...] = ()
        accepted_ids = set()
        batch_no = 0
        while True:
            batch_no += 1
            active_tasks = tuple(
                task for task in tasks if task.artifact_id not in accepted_ids
            )
            batch = scheduler.run_batch(
                active_tasks,
                carry_pending=carry_pending,
                execute=self._execute_task,
            )
            self._record_batch(batch)
            carry_pending = batch.pending
            self.budget_used = dict(batch.stage_points_used)
            for outcome in batch.outcomes:
                if outcome.status in ("DONE", "REUSED"):
                    accepted_ids.add(outcome.task.artifact_id)
            if not carry_pending or (max_batches is not None and batch_no >= max_batches):
                self.pending = carry_pending
                break
        self._commit_snapshots()
        return RunResult(
            state=self.state,
            artifacts=dict(self.artifacts),
            cache_keys=dict(self.cache_keys),
            ledger=tuple(self.ledger),
            events=tuple(self.events),
            failures=tuple(self.failures),
            conversion_diffs=self.conversion_diffs,
            receipts=self.active_receipts,
            quant_logs=dict(self.quant_logs),
            pending=self.pending,
            budget_used=self.budget_used,
            resume_decision=self.resume_decision,
        )

    def _raw_ledger_entry(self, artifact: Artifact) -> None:
        sample_id = artifact.scope.split(":", 1)[1]
        receipt = self.cache_index.receipt_for(artifact.artifact_id)
        degradation_basis = None
        if receipt is not None and receipt.source_version == 3:
            digest_match = receipt.digest == artifact.digest
            decision = "REUSE" if digest_match else "RECOMPUTE"
            reason = "raw v3 digest match" if digest_match else "raw sample changed"
        else:
            digest_match = None
            decision = "RECOMPUTE"
            reason = "raw slice materialized; degraded receipts cannot prove raw reuse"
            degradation_basis = (
                f"source=v{receipt.source_version}; raw digest absent; materialization required"
                if receipt is not None and receipt.source_version < 3
                else None
            )
        self.ledger.append(
            LedgerEntry(
                order=len(self.ledger),
                sort_key=f"0000:1:{self.sample_order[sample_id]:08d}:raw:{artifact.scope}",
                artifact_id=artifact.artifact_id,
                stage="raw",
                scope=artifact.scope,
                decision=decision,
                reason=reason,
                receipt_version=receipt.source_version if receipt else None,
                digest_match=digest_match,
                degraded=receipt is not None and receipt.source_version < 3,
                degradation_basis=degradation_basis,
                upstream=(),
                params=artifact.params,
                points=0,
                points_after=0,
                stage_points_after=0,
            )
        )

    def _build_tasks(self, raw_artifacts: Tuple[Artifact, ...]) -> Tuple[Task, ...]:
        tasks: List[Task] = []
        reusable = {artifact.artifact_id: True for artifact in raw_artifacts}
        receipt = self.cache_index.receipt_for(raw_artifacts[0].artifact_id)
        if receipt is not None and receipt.source_version < 3:
            self.sample_fingerprint_changed = any(
                record.digest != artifact.digest
                for artifact in raw_artifacts
                for record in [self.cache_index.receipt_for(artifact.artifact_id)]
                if record is not None
            )
        for stage_index, stage in enumerate(self.state.pipeline.stages, start=1):
            stage_result = self.computer.compute_stage(
                stage, self.state.samples, self.candidates, self.state.param_versions
            )
            self.quant_logs[stage.name] = stage_result.quant_log
            candidates = {artifact.artifact_id: artifact for artifact in stage_result.artifacts}
            self.candidates.update(candidates)
            for artifact in stage_result.artifacts:
                scope = artifact.scope
                identifier = artifact.artifact_id
                upstream_reused = all(reusable.get(upstream, False) for upstream in artifact.upstream)
                decision = self.cache_index.decide(
                    artifact,
                    expected_artifact_param_version(artifact),
                    sample_fingerprint_changed=self.sample_fingerprint_changed,
                    upstream_reused=upstream_reused,
                )
                task = Task(
                    stage_index=stage_index,
                    stage_name=stage.name,
                    scope=scope,
                    artifact_id=identifier,
                    points=1 if scope == "global" else 2,
                    sort_key=stable_sort_key(stage_index, stage.name, scope, self.sample_order),
                    params=artifact.params,
                    upstream=artifact.upstream,
                    expected_param_version=expected_artifact_param_version(artifact),
                    candidate=artifact,
                    decision=decision,
                    plan=self._make_plan(stage, identifier, candidates),
                )
                reusable[identifier] = decision.reuse and upstream_reused
                tasks.append(task)
        return tuple(tasks)

    def _make_plan(self, stage, identifier: str, candidates: Dict[str, Artifact]):
        def plan() -> Artifact:
            return candidates[identifier]
        return plan

    def _execute_task(self, task: Task) -> Optional[Artifact]:
        if task.decision.reuse:
            return None
        return task.plan()

    def _record_batch(self, batch) -> None:
        for outcome in batch.outcomes:
            task = outcome.task
            decision = task.decision
            if outcome.status not in ("DONE", "REUSED"):
                continue
            if outcome.status == "DONE":
                artifact = outcome.artifact
                self.artifacts[artifact.artifact_id] = artifact
                self.cache_keys[artifact.artifact_id] = cache_key(artifact.artifact_id, artifact.digest)
                ledger_decision = "RECOMPUTE"
            elif outcome.status == "REUSED":
                baseline = self.baseline_artifacts.get(task.artifact_id)
                artifact = baseline if baseline is not None and baseline.digest == task.candidate.digest else task.candidate
                self.artifacts[task.artifact_id] = artifact
                self.cache_keys[task.artifact_id] = cache_key(artifact.artifact_id, artifact.digest)
                ledger_decision = "REUSE"
            else:
                ledger_decision = outcome.status
            self.ledger.append(
                LedgerEntry(
                    order=len(self.ledger),
                    sort_key=task.sort_key,
                    artifact_id=task.artifact_id,
                    stage=task.stage_name,
                    scope=task.scope,
                    decision=ledger_decision,
                    reason=decision.reason if outcome.status in ("DONE", "REUSED") else outcome.reason,
                    receipt_version=decision.receipt_version,
                    digest_match=decision.digest_match,
                    degraded=decision.degraded,
                    degradation_basis=decision.degradation_basis,
                    upstream=task.upstream,
                    params=task.params,
                    points=task.points,
                    points_after=outcome.points_after,
                    stage_points_after=outcome.stage_points_after,
                )
            )
        for event in batch.events:
            self.events.append(dict(event))
            self.failures.append(dict(event))
        self.events.sort(key=lambda item: (item["sort_key"], item["code"]))
        self.failures.sort(key=lambda item: (item["sort_key"], item["code"]))

    def _commit_snapshots(self) -> None:
        if self.snapshot_store is None:
            return
        grouped: Dict[str, List[Artifact]] = {}
        for artifact in self.artifacts.values():
            if artifact.stage != "raw":
                grouped.setdefault(artifact.stage, []).append(artifact)
        manifest_base = {
            "config_hash": self.state.config_hash(),
            "sample_fingerprint": self.state.sample_fingerprint(),
            "receipt_fingerprint": self.state.receipt_fingerprint(),
            "completed_stages": [],
        }
        for stage in sorted(grouped):
            self.snapshot_store.commit_stage(
                self.state.sequence,
                stage,
                tuple(sorted(grouped[stage], key=lambda item: item.artifact_id)),
                manifest_base,
                self.budget_used,
            )
