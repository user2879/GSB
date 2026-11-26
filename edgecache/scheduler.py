"""Concurrent dispatch under global, lease and quota constraints."""

from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_bytes, content_digest
from .client import NodeClient
from .errors import EdgeError, KilledError, LeaseError, PortError
from .jsonlog import StructuredLogger
from .leases import Lease, LeaseManager
from .ledger import Ledger
from .processes import NodeProcessManager
from .quota import QuotaPool, Reservation
from .sampling import compute_sampling, deterministic_sampling_payload
from .spec import EdgeSpec, FragmentEvent, NodeSpec, RevisionSpec


AMBIGUOUS_CODES = {"E_TIMEOUT", "E_RECEIPT", "E_RATE", "E_SAMPLING", "E_PROTOCOL"}


@dataclass
class DispatchJob:
    fragment: FragmentEvent
    node: NodeSpec
    revision: RevisionSpec


@dataclass
class DispatchResult:
    batch_id: str
    node_id: str
    fragment_id: str
    ok: bool
    committed: bool = False
    ambiguous: bool = False
    error: EdgeError | None = None
    sample_digest: str | None = None
    receipt: dict[str, Any] | None = None
    lease: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "node_id": self.node_id,
            "fragment_id": self.fragment_id,
            "ok": self.ok,
            "committed": self.committed,
            "ambiguous": self.ambiguous,
            "error": None if self.error is None else self.error.to_dict(),
            "sample_digest": self.sample_digest,
            "lease": self.lease,
        }


def initial_node_quotas(spec: EdgeSpec) -> dict[str, int]:
    latest = spec.latest_revision_for_node()
    return {node_id: revision.quota for node_id, revision in latest.items()}


class DispatchScheduler:
    def __init__(
        self,
        spec: EdgeSpec,
        ledger: Ledger,
        run_id: str,
        process_manager: NodeProcessManager,
        logger: StructuredLogger,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.run_id = run_id
        self.process_manager = process_manager
        self.logger = logger
        self._condition = threading.Condition()
        self._running_total = 0
        self._running_node: dict[str, int] = {}
        self._lock = threading.RLock()
        self._lease_manager = LeaseManager(
            lease_limit=spec.global_spec.lease_limit_per_node,
            ttl_ms=spec.global_spec.lease_ttl_ms,
        )
        self._quota_pool = QuotaPool(
            initial_node_quotas(spec),
            spec.global_spec.available_quota,
            spec.global_spec.block_weight,
        )

    def _log_error(self, error: EdgeError, job: DispatchJob | None, ambiguous: bool) -> None:
        node_id = job.node.node_id if job is not None else error.details.get("node_id")
        batch_id = job.fragment.batch_id if job is not None else error.details.get("batch_id")
        self.logger.emit(
            error.code,
            level="ERROR",
            node_id=node_id,
            batch_id=batch_id,
            ambiguous=ambiguous,
            **error.details,
        )
        try:
            self.ledger.add_log(
                run_id=self.run_id,
                level="ERROR",
                code=error.code,
                node_id=str(node_id) if node_id is not None else None,
                batch_id=str(batch_id) if batch_id is not None else None,
                details={"ambiguous": ambiguous, **error.details},
            )
        except Exception:
            pass

    def _mark_intent(self, job: DispatchJob) -> None:
        state = self.ledger.register_batch_intent(
            self.run_id,
            batch_id=job.fragment.batch_id,
            fragment_id=job.fragment.fragment_id,
            node_id=job.node.node_id,
            revision_id=job.revision.revision_id,
            sequence=job.fragment.sequence,
            evict_blocks=job.fragment.evict_blocks,
            block_weight=self.spec.global_spec.block_weight,
            quota_version=job.revision.revision_id,
        )
        if state == "committed":
            raise EdgeError(
                "E_LEDGER",
                "batch was already committed",
                details={"batch_id": job.fragment.batch_id},
                exit_code=12,
            )

    def _verify_receipt(self, job: DispatchJob, receipt: dict[str, Any]) -> dict[str, Any]:
        expected_outcome = compute_sampling(job.node, job.fragment, job.revision)
        expected = deterministic_sampling_payload(expected_outcome.payload)
        received = deterministic_sampling_payload(receipt)
        differences: list[dict[str, Any]] = []
        for key, expected_value in expected.items():
            if key not in received or received[key] != expected_value:
                differences.append(
                    {
                        "field": key,
                        "expected": expected_value,
                        "actual": received.get(key),
                    }
                )
        if differences:
            raise EdgeError(
                "E_SAMPLING",
                "node sampling conclusion differs from independent controller recomputation",
                details={"differences": differences[:8], "fragment_id": job.fragment.fragment_id},
                exit_code=9,
            )
        if receipt.get("passed_blocks_are_only_checked") is not True:
            raise EdgeError("E_SAMPLING", "node counted unchecked blocks as passed", exit_code=9)
        return expected

    def _execute(
        self,
        job: DispatchJob,
        lease: Lease,
        reservation: Reservation,
        intent_state: str,
    ) -> DispatchResult:
        client = NodeClient(job.node, self.spec.global_spec.socket_timeout_ms)
        if not self.process_manager.is_alive(job.node.node_id):
            error = self.process_manager.death_error(job.node)
            self.ledger.fail_batch(self.run_id, job.fragment.batch_id, error)
            self._quota_pool.cancel(reservation)
            self._lease_manager.release(lease, "failed")
            self._log_error(error, job, False)
            return DispatchResult(
                batch_id=job.fragment.batch_id,
                node_id=job.node.node_id,
                fragment_id=job.fragment.fragment_id,
                ok=False,
                error=error,
                lease=lease.to_dict(),
            )
        ambiguous = False
        try:
            client.register_lease(
                lease.lease_id,
                lease.version,
                lease.expires_monotonic_ms,
            )
            renewed = client.renew_lease(
                lease.lease_id,
                lease.version,
                self.spec.global_spec.lease_ttl_ms,
            )
            self._lease_manager.update_after_renewal(
                lease,
                lease.version,
                int(renewed["expires_ms"]),
            )
            receipt = client.validate_batch(
                fragment_id=job.fragment.fragment_id,
                quota_version=job.revision.revision_id,
                fragment_rate_ppm=job.revision.fragment_rate_ppm,
                block_rate_ppm=job.revision.block_rate_ppm,
                lease_id=lease.lease_id,
                lease_version=lease.version,
            )
            record_payload = self._verify_receipt(job, receipt)
            expected_delta = job.fragment.evict_blocks * self.spec.global_spec.block_weight
            self.ledger.commit_batch(
                self.run_id,
                job.fragment.batch_id,
                sampling_payload=record_payload,
                elapsed_monotonic_ms=int(receipt["duration_monotonic_ms"]),
                expected_delta=expected_delta,
            )
            self._quota_pool.finalize(reservation)
            self._lease_manager.release(lease, "completed")
            return DispatchResult(
                batch_id=job.fragment.batch_id,
                node_id=job.node.node_id,
                fragment_id=job.fragment.fragment_id,
                ok=True,
                committed=True,
                sample_digest=content_digest(record_payload),
                receipt=receipt,
                lease=lease.to_dict(),
            )
        except Exception as exc:
            if isinstance(exc, EdgeError):
                error = exc
            else:
                error = EdgeError(
                    "E_INTERNAL",
                    f"dispatch failed: {type(exc).__name__}",
                    details={"error": str(exc), "batch_id": job.fragment.batch_id},
                    exit_code=20,
                )
            ambiguous = error.code in AMBIGUOUS_CODES
            if isinstance(error, PortError) and not self.process_manager.is_alive(job.node.node_id):
                error = self.process_manager.death_error(job.node)
                ambiguous = True
            if ambiguous:
                self.ledger.mark_unknown(self.run_id, job.fragment.batch_id, error.message, error.code)
                self._quota_pool.cancel(reservation)
            else:
                self.ledger.fail_batch(self.run_id, job.fragment.batch_id, error)
                self._quota_pool.cancel(reservation)
            self._lease_manager.release(lease, "expired" if lease.is_expired() else "failed")
            self._log_error(error, job, ambiguous)
            return DispatchResult(
                batch_id=job.fragment.batch_id,
                node_id=job.node.node_id,
                fragment_id=job.fragment.fragment_id,
                ok=False,
                ambiguous=ambiguous,
                error=error,
                lease=lease.to_dict(),
            )

    def preflight_node_failures(self, jobs: list[DispatchJob]) -> list[DispatchResult]:
        results: list[DispatchResult] = []
        for job in jobs:
            error = self.process_manager.startup_error_for(job.node.node_id)
            if error is None and not self.process_manager.is_alive(job.node.node_id):
                error = self.process_manager.death_error(job.node)
            if error is None:
                continue
            self._mark_intent(job)
            self.ledger.fail_batch(self.run_id, job.fragment.batch_id, error)
            self._log_error(error, job, False)
            results.append(
                DispatchResult(
                    batch_id=job.fragment.batch_id,
                    node_id=job.node.node_id,
                    fragment_id=job.fragment.fragment_id,
                    ok=False,
                    error=error,
                )
            )
        return results

    def run(self) -> tuple[list[DispatchResult], dict[str, Any]]:
        jobs: list[DispatchJob] = []
        node_by_id = self.spec.node_by_id()
        revisions = self.spec.revision_by_id()
        for fragment in self.spec.ordered_fragments():
            jobs.append(
                DispatchJob(
                    fragment=fragment,
                    node=node_by_id[fragment.node_id],
                    revision=revisions[fragment.quota_revision],
                )
            )
        preflight = self.preflight_node_failures(jobs)
        bad_batch_ids = {result.batch_id for result in preflight}
        runnable = [job for job in jobs if job.fragment.batch_id not in bad_batch_ids]
        results: list[DispatchResult] = list(preflight)
        queues: dict[str, list[DispatchJob]] = {}
        for job in runnable:
            queues.setdefault(job.node.node_id, []).append(job)
        node_ids = sorted(queues)
        cursors = {node_id: 0 for node_id in node_ids}
        total = len(runnable)
        scheduled = 0
        completed = 0
        futures: dict[Future[DispatchResult], DispatchJob] = {}
        global_limit = self.spec.global_spec.concurrency_limit

        def done_callback(future: Future[DispatchResult], node_id: str) -> None:
            with self._condition:
                self._running_total -= 1
                self._running_node[node_id] = self._running_node.get(node_id, 1) - 1
                self._condition.notify_all()

        executor = ThreadPoolExecutor(max_workers=global_limit, thread_name_prefix="edgecache-dispatch")
        try:
            while scheduled < total or futures:
                progress = True
                while progress and scheduled < total and self._running_total < global_limit:
                    progress = False
                    eligible: list[str] = []
                    for node_id in node_ids:
                        limit = min(
                            queues[node_id][0].node.lease_limit,
                            self.spec.global_spec.lease_limit_per_node,
                        )
                        if (
                            cursors[node_id] < len(queues[node_id])
                            and self._running_node.get(node_id, 0) < limit
                        ):
                            eligible.append(node_id)
                    if not eligible:
                        break
                    for node_id in eligible:
                        if self._running_total >= global_limit or scheduled >= total:
                            break
                        if cursors[node_id] >= len(queues[node_id]):
                            continue
                        job = queues[node_id][cursors[node_id]]
                        reservation: Reservation | None = None
                        lease: Lease | None = None
                        try:
                            reservation = self._quota_pool.reserve(
                                job.node.node_id,
                                job.fragment.evict_blocks,
                                timeout_s=0.05,
                            )
                            self._mark_intent(job)
                            lease = self._lease_manager.acquire(
                                job.node.node_id,
                                job.node.lease_limit,
                                ttl_ms=self.spec.global_spec.lease_ttl_ms,
                                timeout_s=0.05,
                            )
                            intent_state = "pending"
                        except EdgeError as error:
                            if reservation is not None:
                                self._quota_pool.cancel(reservation)
                            if lease is not None:
                                self._lease_manager.release(lease, "failed")
                            self._log_error(error, job, False)
                            self.ledger.fail_batch(self.run_id, job.fragment.batch_id, error)
                            results.append(
                                DispatchResult(
                                    batch_id=job.fragment.batch_id,
                                    node_id=job.node.node_id,
                                    fragment_id=job.fragment.fragment_id,
                                    ok=False,
                                    error=error,
                                )
                            )
                            cursors[node_id] += 1
                            scheduled += 1
                            progress = True
                            continue
                        cursors[node_id] += 1
                        scheduled += 1
                        self._running_total += 1
                        self._running_node[node_id] = self._running_node.get(node_id, 0) + 1
                        future = executor.submit(self._execute, job, lease, reservation, intent_state)
                        futures[future] = job
                        future.add_done_callback(lambda fut, nid=node_id: done_callback(fut, nid))
                        progress = True
                if futures:
                    done, _ = wait(list(futures), timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        job = futures.pop(future)
                        result = future.result()
                        results.append(result)
                        completed += 1
                elif scheduled < total:
                    with self._condition:
                        self._condition.wait(timeout=0.05)
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
        summary = {
            "scheduled": scheduled,
            "completed": completed,
            "succeeded": sum(1 for result in results if result.committed),
            "failed": sum(1 for result in results if not result.ok),
            "ambiguous_unknown": sum(1 for result in results if result.ambiguous),
            "quota_pool": self._quota_pool.snapshot(),
            "leases": self._lease_manager.snapshot(),
        }
        return sorted(results, key=lambda result: (result.node_id, result.batch_id)), summary
