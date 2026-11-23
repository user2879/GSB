"""Bounded parallel scheduling of correlation batches on real worker threads."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading

from .clock import MonotonicClock
from .correlation import exact_correlate, approximate_correlate
from .lease import BatchRange, LeaseManager
from .model import Alert, Topology


@dataclass(frozen=True)
class Batch:
    batch_id: str
    service_id: str
    index_lo: int
    index_hi: int
    alert_ids: tuple[str, ...]


@dataclass
class BatchReport:
    batch: Batch
    holder: str
    wait_start_ms: int
    wait_end_ms: int
    run_start_ms: int
    run_end_ms: int
    wait_ms: int
    lease_id: str
    lease_version: int
    pair_count: int

    def deterministic_dict(self) -> dict:
        return {
            "batch_id": self.batch.batch_id,
            "service_id": self.batch.service_id,
            "index_lo": self.batch.index_lo,
            "index_hi": self.batch.index_hi,
            "alert_ids": list(self.batch.alert_ids),
            "holder": self.holder,
            "wait_start_ms": self.wait_start_ms,
            "wait_end_ms": self.wait_end_ms,
            "wait_ms": self.wait_ms,
            "lease_id": self.lease_id,
            "lease_version": self.lease_version,
            "pair_count": self.pair_count,
        }


@dataclass
class SchedulingResult:
    reports: list[BatchReport] = field(default_factory=list)
    merged_pairs: list[dict] = field(default_factory=list)


def build_batches(alerts: list[Alert], window_ms: int) -> list[Batch]:
    """Split each service's ordered alerts on correlation window boundaries."""
    by_service: dict[str, list[Alert]] = {}
    for alert in alerts:
        by_service.setdefault(alert.service_id, []).append(alert)
    batches: list[Batch] = []
    seq = 0
    for service_id in sorted(by_service):
        ordered = sorted(by_service[service_id], key=lambda a: (
            a.timestamp_ms, a.service_id, a.fingerprint, a.alert_id))
        start = 0
        for end in range(1, len(ordered) + 1):
            is_last = end == len(ordered)
            crosses_window = (not is_last and
                              ordered[end].timestamp_ms - ordered[start].timestamp_ms
                              > window_ms)
            if is_last or crosses_window:
                members = ordered[start:end]
                seq += 1
                batches.append(Batch(
                    batch_id=f"batch-{seq:06d}",
                    service_id=service_id,
                    index_lo=start,
                    index_hi=end,
                    alert_ids=tuple(alert.alert_id for alert in members),
                ))
                start = end
    return batches


class CorrelationScheduler:
    def __init__(self, lease_manager: LeaseManager, topology: Topology,
                 window_ms: int, global_capacity: int,
                 service_capacity: int, clock=None, work_hook=None):
        self.lease_manager = lease_manager
        self.topology = topology
        self.window_ms = window_ms
        self.clock = clock or MonotonicClock()
        # Optional real-work callback invoked per batch on the worker thread.
        # It receives (batch, lease) and may renew the lease; it must block
        # using real time so queueing and occupancy remain genuine.
        self.work_hook = work_hook
        self._global_slots = threading.BoundedSemaphore(global_capacity)
        self._service_slots = {
            service: threading.BoundedSemaphore(service_capacity)
            for service in topology.nodes
        }
        self._id_lock = threading.Lock()
        self._submitted = 0

    def _next_holder(self) -> str:
        with self._id_lock:
            self._submitted += 1
            return f"worker-task-{self._submitted:06d}"

    def _run_batch(self, batch: Batch, alert_lookup: dict[str, Alert],
                   use_approximate: bool) -> BatchReport:
        holder = self._next_holder()
        wait_start = self.clock.now_ms()
        # Real blocking queueing: both semaphores and the lease cap gate entry.
        self._global_slots.acquire()
        try:
            self._service_slots[batch.service_id].acquire()
            try:
                lease = None
                while lease is None:
                    try:
                        lease = self.lease_manager.grant(
                            BatchRange(batch.service_id, batch.index_lo,
                                       batch.index_hi),
                            holder)
                    except Exception:
                        # Lease cap temporarily full: park briefly and retry;
                        # the waiting clock keeps advancing in real time.
                        threading.Event().wait(0.002)
                wait_end = self.clock.now_ms()
                run_start = wait_end
                try:
                    if self.work_hook is not None:
                        self.work_hook(batch, lease)
                    members = [alert_lookup[alert_id]
                               for alert_id in batch.alert_ids]
                    if use_approximate:
                        pairs = approximate_correlate(
                            members, self.topology, self.window_ms)
                    else:
                        pairs = exact_correlate(
                            members, self.topology, self.window_ms)
                finally:
                    run_end = self.clock.now_ms()
                    self.lease_manager.release(lease.lease_id)
                return BatchReport(
                    batch=batch, holder=holder,
                    wait_start_ms=wait_start, wait_end_ms=wait_end,
                    run_start_ms=run_start, run_end_ms=run_end,
                    wait_ms=wait_end - wait_start,
                    lease_id=lease.lease_id, lease_version=lease.version,
                    pair_count=len(pairs))
            finally:
                self._service_slots[batch.service_id].release()
        finally:
            self._global_slots.release()

    def execute(self, batches: list[Batch], alerts: list[Alert],
                use_approximate: bool) -> SchedulingResult:
        lookup = {alert.alert_id: alert for alert in alerts}
        reports: list[BatchReport] = []
        with ThreadPoolExecutor(
                max_workers=self.lease_manager.global_capacity) as pool:
            futures = [pool.submit(self._run_batch, batch, lookup,
                                   use_approximate)
                       for batch in batches]
            for future in futures:
                reports.append(future.result())
        reports.sort(key=lambda report: report.batch.batch_id)
        return SchedulingResult(reports=reports)
