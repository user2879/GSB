"""Deterministic budgeted and quota-constrained parallel MinHash scheduling."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .cache import SignatureCache
from .codes import (
    E_ENCODING_INVALID,
    E_GLOBAL_BUDGET_EXHAUSTED,
    E_MINHASH_FAILED,
    E_SOURCE_QUOTA_EXCEEDED,
)
from .config import DedupConfig, SchedulerConfig
from .hashing import compute_signature_from_shingles, shingle_hashes, signature_cache_key
from .models import Document, Failure, SignatureRecord


@dataclass(frozen=True)
class ScheduledTask:
    document: Document
    stage: str = "minhash"

    @property
    def stable_key(self) -> str:
        return f"{self.stage}\x1f{self.document.source}\x1f{self.document.doc_id}"


@dataclass
class PreparedTask:
    task: ScheduledTask
    cache_record: SignatureRecord | None = None
    shingles: frozenset[int] | None = None
    point_cost: int = 0
    error: Exception | None = None


@dataclass
class SchedulerResult:
    signatures: dict[str, SignatureRecord]
    consumed_points: int
    source_points: dict[str, int]
    deductions: list[dict[str, Any]]
    pending: list[dict[str, Any]]
    failures: list[Failure]
    batch_count: int
    status: str
    stopped_early: bool
    pending_reasons: dict[str, str] = field(default_factory=dict)
    reclaimed_points: int = 0

    def state_dict(self) -> dict[str, Any]:
        pending = self.pending
        cursor = {
            "batch_count": self.batch_count,
            "next_pending_stable_key": pending[0]["stable_key"] if pending else None,
            "next_pending_source": pending[0]["source"] if pending else None,
            "next_pending_doc_id": pending[0]["doc_id"] if pending else None,
            "remaining_by_source": dict(
                sorted(
                    {
                        source: sum(1 for item in pending if item["source"] == source)
                        for source in {item["source"] for item in pending}
                    }.items()
                )
            ),
        }
        return {
            "signatures": {
                key: record.to_dict() for key, record in sorted(self.signatures.items())
            },
            "consumed_points": self.consumed_points,
            "source_points": dict(sorted(self.source_points.items())),
            "deductions": self.deductions,
            "pending": self.pending,
            "pending_reasons": dict(sorted(self.pending_reasons.items())),
            "reclaimed_points": self.reclaimed_points,
            "failures": [failure.to_dict() for failure in self.failures],
            "batch_count": self.batch_count,
            "cursor": cursor,
            "status": self.status,
            "stopped_early": self.stopped_early,
        }


def _pending_exception_code(exc: Exception) -> str:
    if isinstance(exc, (ValueError, UnicodeError)):
        return E_ENCODING_INVALID
    return E_MINHASH_FAILED


class BudgetScheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        dedup_config: DedupConfig,
        signature_cache: SignatureCache,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.scheduler_config = scheduler_config
        self.dedup_config = dedup_config
        self.signature_cache = signature_cache
        self.progress_callback = progress_callback

    def execute(
        self,
        documents: Iterable[Document],
        resume_state: dict[str, Any] | None = None,
        stop_after_batches: int | None = None,
    ) -> SchedulerResult:
        docs = sorted(documents, key=lambda document: (document.source, document.doc_id))
        doc_by_key = {document.stable_key: document for document in docs}
        resume_state = resume_state or {}

        signatures: dict[str, SignatureRecord] = {}
        consumed_points = int(resume_state.get("consumed_points", 0) or 0)
        source_points = {
            str(source): int(points)
            for source, points in (resume_state.get("source_points", {}) or {}).items()
        }
        deductions = [dict(item) for item in (resume_state.get("deductions", []) or [])]
        batch_count = int(resume_state.get("batch_count", 0) or 0)
        reclaimed_points = int(resume_state.get("reclaimed_points", 0) or 0)
        failures: list[Failure] = []
        completed_keys: set[str] = set()
        failed_keys: set[str] = set()

        deduction_totals: dict[str, int] = defaultdict(int)
        for item in deductions:
            deduction_totals[str(item.get("stable_key", ""))] += int(item.get("points", 0) or 0)

        def normalize_deductions() -> None:
            deductions.sort(
                key=lambda item: (
                    str(item.get("source", "")),
                    str(item.get("doc_id", "")),
                    str(item.get("stable_key", "")),
                )
            )
            running = self.scheduler_config.global_shingle_budget
            for index, item in enumerate(deductions):
                running -= int(item.get("points", 0) or 0)
                item["sequence"] = index
                item["remaining_global"] = max(0, running)

        def reclaim_signature(key: str, source: str) -> None:
            nonlocal consumed_points, reclaimed_points
            charged = int(deduction_totals.pop(key, 0) or 0)
            if charged <= 0:
                return
            deductions[:] = [item for item in deductions if item.get("stable_key") != key]
            consumed_points = max(0, consumed_points - charged)
            reclaimed_points += charged
            source_points[source] = max(0, source_points.get(source, 0) - charged)
            if source_points[source] == 0:
                source_points.pop(source, None)

        for key, raw_record in (resume_state.get("signatures", {}) or {}).items():
            document = doc_by_key.get(key)
            if document is None:
                continue
            try:
                record = SignatureRecord.from_dict(raw_record)
            except (KeyError, TypeError, ValueError):
                continue
            expected_cache_key = signature_cache_key(document.text, self.dedup_config)
            if (
                record.fingerprint == document.fingerprint
                and record.shingle_length == self.dedup_config.shingle_length
                and record.signature_dim == self.dedup_config.signature_dim
                and record.cache_key == expected_cache_key
            ):
                signatures[key] = SignatureRecord(
                    source=record.source,
                    doc_id=record.doc_id,
                    fingerprint=record.fingerprint,
                    shingle_length=record.shingle_length,
                    signature_dim=record.signature_dim,
                    point_cost=record.point_cost,
                    signature=record.signature,
                    cache_key=record.cache_key,
                    cache_hit=record.cache_hit,
                    reused_from_snapshot=True,
                )
                completed_keys.add(key)
            else:
                reclaim_signature(key, record.source)

        normalize_deductions()

        for raw_failure in resume_state.get("failures", []) or []:
            try:
                failure = Failure.from_dict(raw_failure)
            except (TypeError, ValueError):
                continue
            if failure.code in {E_SOURCE_QUOTA_EXCEEDED, E_GLOBAL_BUDGET_EXHAUSTED}:
                continue
            failures.append(failure)
            if failure.stage == "minhash":
                failed_keys.add(f"{failure.source}\x1f{failure.doc_id}")

        remaining_tasks: dict[str, ScheduledTask] = {}
        for document in docs:
            if document.stable_key in completed_keys or document.stable_key in failed_keys:
                continue
            task = ScheduledTask(document)
            remaining_tasks[task.document.stable_key] = task
        sources = sorted({task.document.source for task in remaining_tasks.values()})

        halted_sources: set[str] = set()
        pending_reasons: dict[str, str] = {}
        global_exhausted = False
        stopped_early = False
        failure_keys = {(failure.stable_key, failure.code, failure.message) for failure in failures}

        def add_failure(failure: Failure) -> None:
            marker = (failure.stable_key, failure.code, failure.message)
            if marker not in failure_keys:
                failure_keys.add(marker)
                failures.append(failure)

        def all_unfinished() -> list[ScheduledTask]:
            return sorted(
                (
                    task
                    for key, task in remaining_tasks.items()
                    if key not in completed_keys and key not in failed_keys
                ),
                key=lambda task: task.stable_key,
            )

        def record_pending(reason_default: str) -> list[dict[str, Any]]:
            unfinished = all_unfinished()
            result: list[dict[str, Any]] = []
            for task in unfinished:
                key = task.document.stable_key
                reason = pending_reasons.get(key)
                if reason is None:
                    reason = "budget_quota" if task.document.source in halted_sources else reason_default
                result.append(
                    {
                        "source": task.document.source,
                        "doc_id": task.document.doc_id,
                        "stable_key": key,
                        "task_key": task.stable_key,
                        "stage": "minhash",
                        "reason": reason,
                    }
                )
            return result

        def snapshot_state(status: str, stopped: bool) -> dict[str, Any]:
            pending = record_pending("checkpoint")
            result = SchedulerResult(
                signatures=signatures,
                consumed_points=consumed_points,
                source_points=source_points,
                deductions=deductions,
                pending=pending,
                failures=sorted(failures, key=Failure.sort_key),
                batch_count=batch_count,
                status=status,
                stopped_early=stopped,
                pending_reasons=pending_reasons,
                reclaimed_points=reclaimed_points,
            )
            return result.state_dict()

        global_remaining = max(0, self.scheduler_config.global_shingle_budget - consumed_points)
        batch_size = self.scheduler_config.batch_size

        while True:
            selected: list[ScheduledTask] = []
            selected_keys: set[str] = set()
            while len(selected) < batch_size:
                progressed = False
                for task in all_unfinished():
                    if task.document.stable_key in selected_keys:
                        continue
                    if task.document.source in halted_sources:
                        continue
                    selected.append(task)
                    selected_keys.add(task.document.stable_key)
                    progressed = True
                    if len(selected) >= batch_size:
                        break
                if not progressed:
                    break
            if not selected:
                break
            selected.sort(key=lambda task: task.stable_key)

            prepared: dict[str, PreparedTask] = {}
            misses: list[ScheduledTask] = []
            for task in selected:
                key = signature_cache_key(task.document.text, self.dedup_config)
                cache_record = self.signature_cache.lookup(
                    key,
                    source=task.document.source,
                    doc_id=task.document.doc_id,
                    fingerprint=task.document.fingerprint,
                    shingle_length=self.dedup_config.shingle_length,
                    signature_dim=self.dedup_config.signature_dim,
                )
                if cache_record is not None:
                    prepared[task.stable_key] = PreparedTask(task=task, cache_record=cache_record)
                else:
                    misses.append(task)

            if misses:
                with ThreadPoolExecutor(max_workers=self.scheduler_config.max_workers) as executor:
                    futures = {
                        task.stable_key: executor.submit(
                            shingle_hashes,
                            task.document.text,
                            self.dedup_config.shingle_length,
                        )
                        for task in misses
                    }
                    for task in misses:
                        try:
                            hashes = futures[task.stable_key].result()
                            prepared[task.stable_key] = PreparedTask(
                                task=task,
                                shingles=hashes,
                                point_cost=len(hashes),
                            )
                        except Exception as exc:
                            prepared[task.stable_key] = PreparedTask(task=task, error=exc)

            compute: list[PreparedTask] = []
            for task in selected:
                prep = prepared[task.stable_key]
                source = task.document.source
                if source in halted_sources or global_exhausted:
                    continue
                if prep.error is not None:
                    code = _pending_exception_code(prep.error)
                    add_failure(
                        Failure(
                            source=source,
                            doc_id=task.document.doc_id,
                            stage="minhash",
                            code=code,
                            message=f"shingle preparation failed: {prep.error}",
                        )
                    )
                    failed_keys.add(task.document.stable_key)
                    remaining_tasks.pop(task.document.stable_key, None)
                    continue
                if prep.cache_record is not None:
                    signatures[task.document.stable_key] = prep.cache_record
                    completed_keys.add(task.document.stable_key)
                    remaining_tasks.pop(task.document.stable_key, None)
                    continue

                point_cost = prep.point_cost
                quota = self.scheduler_config.source_quotas.get(source)
                if quota is not None and source_points.get(source, 0) + point_cost > quota:
                    halted_sources.add(source)
                    add_failure(
                        Failure(
                            source=source,
                            doc_id=task.document.doc_id,
                            stage="minhash",
                            code=E_SOURCE_QUOTA_EXCEEDED,
                            message=(
                                f"source quota {quota} would be exceeded by "
                                f"{source_points.get(source, 0)} + {point_cost} points"
                            ),
                        )
                    )
                    pending_reasons.setdefault(task.document.stable_key, "budget_quota")
                    continue
                if point_cost > global_remaining:
                    global_exhausted = True
                    pending_reasons.setdefault(task.document.stable_key, "budget_global")
                    for pending_source in sources:
                        if pending_source in halted_sources:
                            continue
                        pending_tasks = [
                            pending_task
                            for pending_task in all_unfinished()
                            if pending_task.document.source == pending_source
                        ]
                        if not pending_tasks:
                            continue
                        first = pending_tasks[0]
                        for pending_task in pending_tasks:
                            pending_reasons.setdefault(pending_task.document.stable_key, "budget_global")
                        add_failure(
                            Failure(
                                source=pending_source,
                                doc_id=first.document.doc_id,
                                stage="minhash",
                                code=E_GLOBAL_BUDGET_EXHAUSTED,
                                message=(
                                    f"global budget exhausted: remaining {global_remaining}, "
                                    f"next document requires {point_cost}"
                                ),
                            )
                        )
                    break

                consumed_points += point_cost
                global_remaining -= point_cost
                source_points[source] = source_points.get(source, 0) + point_cost
                deductions.append(
                    {
                        "sequence": len(deductions),
                        "source": source,
                        "doc_id": task.document.doc_id,
                        "stable_key": task.document.stable_key,
                        "points": point_cost,
                        "remaining_global": global_remaining,
                    }
                )
                compute.append(prep)

            normalize_deductions()

            if compute:
                with ThreadPoolExecutor(max_workers=self.scheduler_config.max_workers) as executor:
                    futures = {
                        prep.task.stable_key: executor.submit(
                            compute_signature_from_shingles,
                            prep.shingles,
                            self.dedup_config.signature_dim,
                        )
                        for prep in compute
                    }
                    for prep in compute:
                        task = prep.task
                        try:
                            signature = futures[task.stable_key].result()
                            cache_key = signature_cache_key(task.document.text, self.dedup_config)
                            record = SignatureRecord(
                                source=task.document.source,
                                doc_id=task.document.doc_id,
                                fingerprint=task.document.fingerprint,
                                shingle_length=self.dedup_config.shingle_length,
                                signature_dim=self.dedup_config.signature_dim,
                                point_cost=prep.point_cost,
                                signature=signature,
                                cache_key=cache_key,
                            )
                            signatures[task.document.stable_key] = record
                            completed_keys.add(task.document.stable_key)
                            remaining_tasks.pop(task.document.stable_key, None)
                            self.signature_cache.put(record)
                        except Exception as exc:
                            add_failure(
                                Failure(
                                    source=task.document.source,
                                    doc_id=task.document.doc_id,
                                    stage="minhash",
                                    code=E_MINHASH_FAILED,
                                    message=f"signature computation failed: {exc}",
                                )
                            )
                            failed_keys.add(task.document.stable_key)
                            remaining_tasks.pop(task.document.stable_key, None)

            batch_count += 1
            if global_exhausted:
                for task in all_unfinished():
                    pending_reasons.setdefault(
                        task.document.stable_key,
                        "budget_quota" if task.document.source in halted_sources else "budget_global",
                    )
            if (
                stop_after_batches is not None
                and batch_count >= stop_after_batches
                and bool(all_unfinished())
            ):
                stopped_early = True

            should_commit = (
                stopped_early
                or global_exhausted
                or not all_unfinished()
                or batch_count % self.scheduler_config.snapshot_interval_batches == 0
            )
            if should_commit and self.progress_callback is not None:
                if stopped_early:
                    status = "interrupted"
                elif global_exhausted or any(
                    task.document.source in halted_sources for task in all_unfinished()
                ):
                    status = "waiting_budget"
                elif all_unfinished():
                    status = "running"
                else:
                    status = "complete"
                self.progress_callback(snapshot_state(status, stopped_early))
            if stopped_early or global_exhausted:
                break

        pending = record_pending("checkpoint")
        normalize_deductions()
        if stopped_early:
            status = "interrupted"
        elif pending:
            status = "waiting_budget"
        else:
            status = "complete"
        return SchedulerResult(
            signatures=signatures,
            consumed_points=consumed_points,
            source_points=source_points,
            deductions=deductions,
            pending=pending,
            failures=sorted(failures, key=Failure.sort_key),
            batch_count=batch_count,
            status=status,
            stopped_early=stopped_early,
            pending_reasons=pending_reasons,
            reclaimed_points=reclaimed_points,
        )
