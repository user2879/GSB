from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import ThroughputBudget
from .content import LEAF_SIZE, generate_segment
from .digest import MerkleBuilder, batch_digest, digest_chunk, leaf_digest
from .errors import BudgetError, DiagnosticError, DigestError, LeaseError
from .events import EventLogger, monotonic_ms
from .lease import Lease, LeaseManager
from .ledger import Ledger
from .models import ObjectSpec, SessionSpec
from .planner import PathSelection


@dataclass
class FaultPlan:
    digest_mismatch: bool = False
    target_object_key: str | None = None
    target_shard_index: int | None = None
    crash_after_batches: int | None = None


@dataclass
class ShardTask:
    spec: ObjectSpec
    shard_index: int
    queued_at_ms: int
    wait_start_ms: int
    wait_end_ms: int | None = None
    budget_wait_start_ms: int | None = None
    budget_wait_end_ms: int | None = None
    lease: Lease | None = None
    status: str = "queued"
    error: DiagnosticError | None = None
    batches_committed: int = 0
    bytes_constructed: int = 0

    def as_dict(self) -> dict[str, Any]:
        result = {
            "object_key": self.spec.object_key,
            "shard_index": self.shard_index,
            "status": self.status,
            "queued_at_monotonic_ms": self.queued_at_ms,
            "wait_start_monotonic_ms": self.wait_start_ms,
            "wait_end_monotonic_ms": self.wait_end_ms,
            "budget_wait_start_monotonic_ms": self.budget_wait_start_ms,
            "budget_wait_end_monotonic_ms": self.budget_wait_end_ms,
            "batches_committed": self.batches_committed,
            "bytes_constructed": self.bytes_constructed,
        }
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result


class _FaultController:
    def __init__(self, plan: FaultPlan, tasks: list[ShardTask]) -> None:
        self.plan = plan
        self._lock = threading.Lock()
        self._digest_fired = False
        self._batch_count = 0
        target_key = plan.target_object_key
        target_shard = plan.target_shard_index
        if plan.digest_mismatch and (target_key is None or target_shard is None):
            if tasks:
                target_key = tasks[0].spec.object_key
                target_shard = tasks[0].shard_index
        self.target_key = target_key
        self.target_shard = target_shard

    def should_corrupt_digest(self, object_key: str, shard_index: int) -> bool:
        if not self.plan.digest_mismatch:
            return False
        if object_key != self.target_key or shard_index != self.target_shard:
            return False
        with self._lock:
            if self._digest_fired:
                return False
            self._digest_fired = True
            return True

    def count_batch_and_should_crash(self) -> bool:
        limit = self.plan.crash_after_batches
        if limit is None:
            return False
        with self._lock:
            self._batch_count += 1
            return self._batch_count >= limit


class ShardScheduler:
    def __init__(
        self,
        *,
        session: SessionSpec,
        selections: dict[str, PathSelection],
        ledger: Ledger,
        leases: LeaseManager,
        budget: ThroughputBudget,
        logger: EventLogger,
        faults: FaultPlan | None = None,
    ) -> None:
        self.session = session
        self.selections = selections
        self.ledger = ledger
        self.leases = leases
        self.budget = budget
        self.logger = logger
        self.faults = faults or FaultPlan()
        self._tasks: list[ShardTask] = []
        self._fault_controller: _FaultController | None = None

    def _build_tasks(self) -> list[ShardTask]:
        now = monotonic_ms()
        tasks: list[ShardTask] = []
        for spec in sorted(self.session.objects, key=lambda item: item.object_key):
            complete_by_index = {
                int(row["shard_index"]): row["status"] == "complete"
                for row in self.ledger.shard_rows(spec.object_key)
            }
            for shard_index in range(spec.shard_count):
                if complete_by_index.get(shard_index, False):
                    continue
                tasks.append(
                    ShardTask(
                        spec=spec,
                        shard_index=shard_index,
                        queued_at_ms=now,
                        wait_start_ms=now,
                        status="queued",
                    )
                )
        return tasks

    def run(self) -> list[ShardTask]:
        self._tasks = self._build_tasks()
        self._fault_controller = _FaultController(self.faults, self._tasks)
        pending = {id(task): task for task in self._tasks}
        futures: dict[Future[ShardTask], ShardTask] = {}
        worker_limit = min(self.session.worker_threads, self.session.global_lease_quota)
        worker_limit = max(1, worker_limit)

        with ThreadPoolExecutor(max_workers=worker_limit, thread_name_prefix="ossdiag-shard") as executor:
            while pending or futures:
                self.leases.expire_due()
                scheduled = False
                for task_id, task in sorted(
                    list(pending.items()), key=lambda item: (item[1].spec.object_key, item[1].shard_index)
                ):
                    if len(futures) >= worker_limit:
                        break
                    spec = task.spec
                    if self.leases.active_count >= self.session.global_lease_quota:
                        break
                    if self.leases.active_object_count(spec.object_key) >= spec.effective_concurrency:
                        continue
                    try:
                        lease = self.leases.grant(spec.object_key, task.shard_index, spec.lease_policy)
                    except LeaseError:
                        continue
                    task.lease = lease
                    task.wait_end_ms = monotonic_ms()
                    task.status = "running"
                    pending.pop(task_id, None)
                    future = executor.submit(self._process_shard, task)
                    futures[future] = task
                    scheduled = True

                if not futures:
                    if pending:
                        time.sleep(0.005)
                    continue
                done, _ = wait(futures, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    try:
                        completed = future.result()
                    except BaseException as exc:
                        completed = task
                        if isinstance(exc, DiagnosticError):
                            completed.error = exc
                        else:
                            completed.error = DiagnosticError(
                                "E_INTERNAL",
                                f"worker failed unexpectedly: {exc}",
                                object_key=task.spec.object_key,
                                shard_index=task.shard_index,
                            )
                        completed.status = "failed"
                    if completed.error is not None:
                        self.logger.business_error(completed.error)
                        try:
                            self.ledger.mark_object_failed(completed.spec.object_key, completed.error.code)
                        except Exception:
                            pass
                    self._tasks[self._tasks.index(task)] = completed
        return self._tasks

    def _ensure_lease(self, task: ShardTask, lease: Lease, last_renew: int) -> tuple[Lease, int]:
        spec = task.spec
        now = self.leases.clock_ms()
        if now + max(1, spec.lease_policy.renew_interval_ms // 4) >= lease.expires_ms:
            try:
                lease = self.leases.renew(lease, spec.lease_policy.lease_ms)
                last_renew = now
                return lease, last_renew
            except LeaseError:
                pass
        elif now - last_renew >= spec.lease_policy.renew_interval_ms:
            try:
                lease = self.leases.renew(lease, spec.lease_policy.lease_ms)
                last_renew = now
                return lease, last_renew
            except LeaseError:
                pass
        while True:
            try:
                self.leases.assert_valid(lease)
                return lease, last_renew
            except LeaseError:
                self.leases.expire_due()
                try:
                    replacement = self.leases.grant(spec.object_key, task.shard_index, spec.lease_policy)
                    task.lease = replacement
                    return replacement, replacement.granted_ms
                except LeaseError:
                    time.sleep(0.005)

    def _verify_full_shard(
        self,
        *,
        handle: Any,
        object_key: str,
        shard_index: int,
        shard_length: int,
        expected_root_hex: str,
    ) -> None:
        builder = MerkleBuilder()
        handle.seek(0)
        local = 0
        while local < shard_length:
            length = min(LEAF_SIZE, shard_length - local)
            data = handle.read(length)
            if len(data) != length:
                raise DigestError(
                    "落盘分片读取不足",
                    object_key=object_key,
                    shard_index=shard_index,
                    detail={"offset": local, "expected": length, "actual": len(data)},
                )
            builder.add(leaf_digest(object_key, shard_index, local // LEAF_SIZE, length, data))
            local += length
        actual = builder.root().hex()
        if actual != expected_root_hex:
            raise DigestError(
                "落盘分片摘要与构造摘要不一致",
                object_key=object_key,
                shard_index=shard_index,
                detail={"expected": expected_root_hex, "actual": actual},
            )

    def _process_shard(self, task: ShardTask) -> ShardTask:
        spec = task.spec
        assert self._fault_controller is not None
        shard = self.ledger.shard_rows(spec.object_key)[task.shard_index]
        start = int(shard["byte_start"])
        end = int(shard["byte_end"])
        shard_length = end - start
        cursor = int(shard["progress_cursor"])
        relative = cursor - start
        path = self.ledger.shard_path(spec.object_key, task.shard_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_leaf_hex = self.ledger.leaf_hex_values(spec.object_key, task.shard_index)
        selection = self.selections[spec.object_key]
        from .digest import make_aggregator

        aggregator = make_aggregator(selection.path, [bytes.fromhex(value) for value in existing_leaf_hex])
        lease = task.lease
        assert lease is not None
        lease, last_renew = self._ensure_lease(task, lease, lease.granted_ms)
        mode = "r+b" if path.exists() else "w+b"

        try:
            handle = path.open(mode)
        except FileNotFoundError:
            handle = path.open("w+b")
        try:
            handle.seek(relative)
            handle.truncate(relative)
            while relative < shard_length:
                lease, last_renew = self._ensure_lease(task, lease, last_renew)

                batch_length = min(spec.batch_size, shard_length - relative)
                batch_start_absolute = start + relative
                batch_end_absolute = batch_start_absolute + batch_length
                run_id = uuid.uuid4().hex
                task.budget_wait_start_ms = monotonic_ms()
                try:
                    budget_wait = self.budget.reserve(batch_length)
                except BudgetError:
                    task.budget_wait_end_ms = monotonic_ms()
                    raise
                task.budget_wait_end_ms = int(budget_wait["wait_end_monotonic_ms"])
                self.ledger.begin_batch(
                    run_id=run_id,
                    object_key=spec.object_key,
                    shard_index=task.shard_index,
                    byte_start=batch_start_absolute,
                    byte_end=batch_end_absolute,
                    lease_version=lease.version,
                    renew_count=lease.renew_count,
                )
                try:
                    data = generate_segment(
                        spec.object_key,
                        task.shard_index,
                        spec.seed_for(task.shard_index),
                        relative,
                        batch_length,
                    )
                    digest_start_ns = time.monotonic_ns()
                    chunk_digests = digest_chunk(
                        spec.object_key,
                        task.shard_index,
                        relative,
                        shard_length,
                        data,
                    )
                    digest_duration_ns = time.monotonic_ns() - digest_start_ns

                    handle.seek(relative)
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())

                    handle.seek(relative)
                    observed = handle.read(batch_length)
                    observed_digest = batch_digest(
                        spec.object_key, task.shard_index, relative, observed
                    ).hex()
                    if self._fault_controller.should_corrupt_digest(spec.object_key, task.shard_index):
                        observed_digest = "00" * 32
                    if observed_digest != chunk_digests.batch_hex:
                        raise DigestError(
                            "分片批次摘要不符",
                            object_key=spec.object_key,
                            shard_index=task.shard_index,
                            detail={
                                "byte_start": batch_start_absolute,
                                "byte_end": batch_end_absolute,
                                "expected": chunk_digests.batch_hex,
                                "actual": observed_digest,
                            },
                        )

                    if self._fault_controller.count_batch_and_should_crash():
                        os._exit(137)

                    try:
                        self.leases.assert_valid(lease)
                    except LeaseError:
                        self.ledger.abort_inflight(run_id, reason="lease_expired_before_commit")
                        handle.seek(relative)
                        handle.truncate(relative)
                        handle.flush()
                        os.fsync(handle.fileno())
                        lease, last_renew = self._ensure_lease(task, lease, last_renew)
                        continue

                    for leaf_index, digest_hex in chunk_digests.leaves:
                        aggregator.add_leaf(bytes.fromhex(digest_hex))
                    new_relative = relative + batch_length
                    final_root = None
                    if new_relative == shard_length:
                        final_root = aggregator.finalize().hex()
                        self._verify_full_shard(
                            handle=handle,
                            object_key=spec.object_key,
                            shard_index=task.shard_index,
                            shard_length=shard_length,
                            expected_root_hex=final_root,
                        )
                        handle.seek(new_relative)
                    try:
                        self.leases.assert_valid(lease)
                    except LeaseError:
                        self.ledger.abort_inflight(run_id, reason="lease_expired_after_verification")
                        handle.seek(relative)
                        handle.truncate(relative)
                        handle.flush()
                        os.fsync(handle.fileno())
                        aggregator = make_aggregator(
                            selection.path,
                            [
                                bytes.fromhex(value)
                                for value in self.ledger.leaf_hex_values(spec.object_key, task.shard_index)
                            ],
                        )
                        lease, last_renew = self._ensure_lease(task, lease, last_renew)
                        continue
                    self.ledger.commit_batch(
                        run_id=run_id,
                        object_key=spec.object_key,
                        shard_index=task.shard_index,
                        byte_start=batch_start_absolute,
                        byte_end=batch_end_absolute,
                        content_digest_hex=chunk_digests.batch_hex,
                        lease_version=lease.version,
                        renew_count=lease.renew_count,
                        progress_cursor=batch_end_absolute,
                        digest_duration_ns=digest_duration_ns,
                        leaves=chunk_digests.leaves,
                        final_content_digest_hex=final_root,
                    )
                    task.batches_committed += 1
                    task.bytes_constructed += batch_length
                    relative = new_relative
                    self.logger.emit(
                        "batch_committed",
                        object_key=spec.object_key,
                        shard_index=task.shard_index,
                        byte_start=batch_start_absolute,
                        byte_end=batch_end_absolute,
                        batch_digest=chunk_digests.batch_hex,
                        shard_complete=final_root is not None,
                        path=selection.path,
                    )
                except BaseException as exc:
                    try:
                        self.ledger.abort_inflight(run_id, reason=type(exc).__name__)
                    except Exception:
                        pass
                    try:
                        handle.seek(relative)
                        handle.truncate(relative)
                    except Exception:
                        pass
                    raise
        finally:
            try:
                handle.close()
            finally:
                self.leases.release(lease)

        task.status = "complete"
        return task
