"""Concurrency-constrained parallel receipt scheduling.

Two real limits are enforced simultaneously:

* a global semaphore caps the number of running batches,
* one semaphore per worker caps that worker's in-flight batches.

Batches are enqueued in deterministic order; when a quota is exhausted the
batch waits in the thread pool and the monotonic-clock wait window
(``wait_started_ms`` / ``wait_ended_ms``) is recorded. The actual NumPy
forward result is computed locally and then confirmed through a real
loopback round trip to the worker subprocess.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .errors import BatchDiagError
from .model import build_parameters, run_batch
from .worker import send_receipt


def now_ms():
    return time.monotonic_ns() // 1_000_000


@dataclass
class ScheduledRecord:
    batch_id: str
    worker_id: str
    batch_seq: int
    status: str
    enqueued_ms: int
    wait_started_ms: int
    wait_ended_ms: int
    started_ms: int
    finished_ms: int
    wait_ms: int
    socket_elapsed_ms: int
    output_digest: str
    param_digest: str
    error_code: str = None
    error: str = None


@dataclass
class ScheduleTrace:
    records: list = field(default_factory=list)
    max_global_in_flight: int = 0
    max_per_worker_in_flight: dict = field(default_factory=dict)
    global_limit: int = 0
    worker_limits: dict = field(default_factory=dict)


class _Counters:
    def __init__(self):
        self.lock = threading.Lock()
        self.global_in_flight = 0
        self.max_global = 0
        self.per_worker = {}
        self.max_per_worker = {}

    def enter(self, worker_id):
        with self.lock:
            self.global_in_flight += 1
            self.max_global = max(self.max_global, self.global_in_flight)
            current = self.per_worker.get(worker_id, 0) + 1
            self.per_worker[worker_id] = current
            self.max_per_worker[worker_id] = max(
                self.max_per_worker.get(worker_id, 0), current)

    def leave(self, worker_id):
        with self.lock:
            self.global_in_flight -= 1
            self.per_worker[worker_id] -= 1


def _process_batch(spec, plan, batch, global_sem, worker_sems, counters,
                   timeout_ms):
    wait_started = now_ms()
    worker_sem = worker_sems[batch.worker_id]
    while True:
        worker_acquired = worker_sem.acquire(timeout=0.05)
        if worker_acquired:
            try:
                if global_sem.acquire(timeout=0.05):
                    break
            except Exception:
                worker_sem.release()
                raise
            else:
                worker_sem.release()
    wait_ended = now_ms()
    counters.enter(batch.worker_id)
    started = now_ms()
    record = ScheduledRecord(
        batch_id=batch.batch_id, worker_id=batch.worker_id,
        batch_seq=batch.batch_seq, status="ok",
        enqueued_ms=wait_started, wait_started_ms=wait_started,
        wait_ended_ms=wait_ended, started_ms=started,
        finished_ms=0, wait_ms=wait_ended - wait_started,
        socket_elapsed_ms=0, output_digest="", param_digest="",
    )
    try:
        worker = spec.worker(batch.worker_id)
        params = build_parameters(worker)
        result = run_batch(worker, batch, params)
        ack, socket_elapsed = send_receipt(
            worker.host, worker.port,
            {"worker_id": worker.worker_id, "batch_id": batch.batch_id,
             "output_digest": result.output_digest,
             "param_digest": result.param_digest},
            timeout_ms,
        )
        record.socket_elapsed_ms = socket_elapsed
        record.output_digest = result.output_digest
        record.param_digest = result.param_digest
    except BatchDiagError as exc:
        record.status = "error"
        record.error_code = exc.code
        record.error = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        record.status = "error"
        record.error_code = "E_FIELD"
        record.error = repr(exc)
    finally:
        record.finished_ms = now_ms()
        counters.leave(batch.worker_id)
        global_sem.release()
        worker_sem.release()
    return record, locals().get("result")


def schedule_all(spec, plan, max_workers=None):
    """Run every planned batch through the constrained thread pool."""
    global_limit = min(spec.merge_rules.global_concurrency,
                       len(plan.batches)) or 1
    worker_limits = {w.worker_id: w.quota for w in spec.workers}
    global_sem = threading.Semaphore(global_limit)
    worker_sems = {w.worker_id: threading.Semaphore(w.quota)
                   for w in spec.workers}
    counters = _Counters()
    ordered = sorted(plan.batches, key=lambda b: (b.worker_id, b.batch_seq))
    pool_size = max_workers or min(len(ordered), 16)
    results = {}
    trace = ScheduleTrace(global_limit=global_limit,
                          worker_limits=worker_limits)
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        futures = [
            pool.submit(_process_batch, spec, plan, batch, global_sem,
                        worker_sems, counters,
                        spec.merge_rules.socket_timeout_ms)
            for batch in ordered
        ]
        for future in futures:
            record, result = future.result()
            trace.records.append(record)
            if result is not None and record.status == "ok":
                results[record.batch_id] = result
    trace.records.sort(key=lambda r: (r.worker_id, r.batch_seq))
    trace.max_global_in_flight = counters.max_global
    trace.max_per_worker_in_flight = dict(sorted(
        counters.max_per_worker.items()))
    return results, trace
