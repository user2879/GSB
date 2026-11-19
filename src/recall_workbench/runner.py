"""Concurrent, resumable execution of recall, fusion, and evaluation stages."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from .diagnostics import boundary_sensitive_candidates, compare_precisions
from .evaluation import evaluate_user, summarize_evaluation
from .fusion import run_fusion_for_user
from .models import AppConfig, Dataset
from .recall import RecallContext, run_recall_for_user
from .snapshot import SnapshotStore


@dataclass
class RunOutcome:
    exit_code: int
    status: str
    stopped: bool
    completed_stage: str | None
    stats: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    def __init__(
        self,
        dataset: Dataset,
        config: AppConfig,
        snapshot: SnapshotStore,
        *,
        serial: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.snapshot = snapshot
        self.serial = serial
        self.stop_event = stop_event or threading.Event()
        self.user_ids = dataset.user_ids
        self.context = RecallContext.build(dataset.items, dataset.histories, config)
        self.stats: dict[str, Any] = {
            "mode": "serial" if serial else "thread_pool",
            "configured_max_workers": config.max_workers,
            "snapshot_commit_interval": config.snapshot_commit_interval,
            "users": len(self.user_ids),
            "stages": {},
        }

    def _stage_stats(self, stage: str, *, computed: int, resumed: int, commit_count: int) -> None:
        self.stats["stages"][stage] = {
            "computed_this_run": computed,
            "resumed_users": resumed,
            "commit_count": commit_count,
            "max_workers_used": 1 if self.serial else self.config.max_workers,
        }

    def _run_batch(self, batch: list[str], function: Callable[[str], dict[str, Any]]) -> list[dict[str, Any]]:
        if self.serial:
            return [function(user_id) for user_id in batch]
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="rw-user") as executor:
            futures = {executor.submit(function, user_id): user_id for user_id in batch}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [results[user_id] for user_id in batch]

    def _process_stage(
        self,
        stage: str,
        function: Callable[[str], dict[str, Any]],
        finalize: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        existing = self.snapshot.load_stage(stage)
        if existing is not None and existing.get("complete"):
            payload = existing["payload"]
            self._stage_stats(stage, computed=0, resumed=len(existing.get("completed_users", [])), commit_count=0)
            return payload, False

        records: list[dict[str, Any]] = []
        committed: set[str] = set()
        processed: set[str] = set()
        if existing is not None:
            payload = existing.get("payload", {})
            records = list(payload.get("records", []))
            committed = set(existing.get("completed_users", []))
            processed = set(committed)
        computed = 0
        commit_count = 0
        pending_records: list[dict[str, Any]] = []
        resumed = len(committed)
        batch_size = max(1, self.config.max_workers)

        def commit(complete: bool, *, flush_pending: bool = False) -> dict[str, Any]:
            nonlocal commit_count
            if flush_pending and pending_records:
                records.extend(pending_records)
                committed.update(record["user_id"] for record in pending_records)
                pending_records.clear()
            records.sort(key=lambda record: record["user_id"])
            payload = {"records": records}
            if complete and finalize is not None:
                payload = finalize(records)
            self.snapshot.commit_stage(stage, payload, sorted(committed), complete)
            commit_count += 1
            return payload

        def flush_interval_commits() -> None:
            while len(pending_records) >= self.config.snapshot_commit_interval:
                batch = pending_records[: self.config.snapshot_commit_interval]
                del pending_records[: self.config.snapshot_commit_interval]
                records.extend(batch)
                committed.update(record["user_id"] for record in batch)
                commit(False)

        for offset in range(0, len(self.user_ids), batch_size):
            if self.stop_event.is_set():
                commit(False, flush_pending=True)
                self._stage_stats(stage, computed=computed, resumed=resumed, commit_count=commit_count)
                return None, True
            batch = [user_id for user_id in self.user_ids[offset : offset + batch_size] if user_id not in processed]
            if not batch:
                continue
            batch_results = self._run_batch(batch, function)
            pending_records.extend(batch_results)
            processed.update(batch)
            computed += len(batch_results)
            flush_interval_commits()
            if self.stop_event.is_set():
                commit(False, flush_pending=True)
                self._stage_stats(stage, computed=computed, resumed=resumed, commit_count=commit_count)
                return None, True

        payload = commit(True, flush_pending=True)
        self._stage_stats(stage, computed=computed, resumed=resumed, commit_count=commit_count)
        return payload, False

    def run(self) -> RunOutcome:
        failures = sorted((failure.to_dict() for failure in self.dataset.failures), key=lambda failure: (
            failure["source"],
            -1 if failure["line"] is None else failure["line"],
            failure["field"],
            failure["message"],
            failure["code"],
        ))
        self.snapshot.commit_failures(failures)
        self.snapshot.set_status("running")

        recall_payload, stopped = self._process_stage("recall", lambda user_id: run_recall_for_user(user_id, self.context, self.config))
        if stopped:
            return self._interrupted("recall")
        assert recall_payload is not None
        recall_by_user = {record["user_id"]: record for record in recall_payload["records"]}

        if self.stop_event.is_set():
            return self._interrupted("recall", after_stage=True)

        def compute_fusion(user_id: str) -> dict[str, Any]:
            recall_record = recall_by_user[user_id]
            primary = run_fusion_for_user(recall_record, self.config, self.config.primary_bits)
            reference = run_fusion_for_user(recall_record, self.config, self.config.reference_bits)
            comparison = compare_precisions(primary, reference, self.config)
            return {
                "user_id": user_id,
                "task_key": f"{user_id}|fusion={self.config.primary_bits},{self.config.reference_bits}",
                "primary": primary,
                "reference": reference,
                "comparison": comparison,
                "boundary_sensitive": boundary_sensitive_candidates(primary, reference, comparison, self.config),
            }

        fusion_payload, stopped = self._process_stage("fusion", compute_fusion)
        if stopped:
            return self._interrupted("fusion")
        assert fusion_payload is not None
        fusion_by_user = {record["user_id"]: record for record in fusion_payload["records"]}

        if self.stop_event.is_set():
            return self._interrupted("fusion", after_stage=True)

        def compute_evaluation(user_id: str) -> dict[str, Any]:
            fusion_record = fusion_by_user[user_id]
            return {
                "user_id": user_id,
                "task_key": f"{user_id}|evaluation",
                "primary": evaluate_user(fusion_record["primary"], self.dataset, self.config),
                "reference": evaluate_user(fusion_record["reference"], self.dataset, self.config),
            }

        def finalize_evaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "records": records,
                "summary": {
                    "primary": summarize_evaluation(
                        [record["primary"] for record in records], self.config, self.config.primary_bits
                    ),
                    "reference": summarize_evaluation(
                        [record["reference"] for record in records], self.config, self.config.reference_bits
                    ),
                },
            }

        evaluation_payload, stopped = self._process_stage("evaluation", compute_evaluation, finalize_evaluation)
        if stopped:
            return self._interrupted("evaluation")
        self.snapshot.set_status("completed")
        return RunOutcome(
            exit_code=0,
            status="completed",
            stopped=False,
            completed_stage="evaluation",
            stats=self.stats,
        )

    def _interrupted(self, stage: str, *, after_stage: bool = False) -> RunOutcome:
        self.snapshot.set_status("interrupted")
        self.stats["interrupted_stage"] = stage
        self.stats["interrupted_after_stage_commit"] = after_stage
        return RunOutcome(
            exit_code=0,
            status="interrupted",
            stopped=True,
            completed_stage=stage if after_stage else None,
            stats=self.stats,
        )
