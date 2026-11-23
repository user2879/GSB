"""Concurrent request scheduling under dual quota constraints.

A real worker pool (ThreadPoolExecutor) executes requests. At any moment
in-flight requests total <= global_limit, and in-flight requests per
service <= that service's quota. Requests that cannot acquire quota wait;
wait start/end are stamped with the monotonic clock. Each request yields
a trace with queue, decide and forward intervals.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .clock import monotonic_ms


@dataclass
class RequestTrace:
    seq: int
    wait_start_ms: int = 0
    wait_end_ms: int = 0
    decide_start_ms: int = 0
    decide_end_ms: int = 0
    fwd_start_ms: int = -1   # -1 means "not forwarded"
    fwd_end_ms: int = -1
    decision: object = None  # engine.DecisionRecord
    receipt: object = None   # forward.ForwardResult or None

    @property
    def wait_ms(self):
        return self.wait_end_ms - self.wait_start_ms


class _InflightGauge:
    """Tracks current and peak in-flight counts (for verification)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def acquire(self):
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def release(self):
        with self._lock:
            self.current -= 1


class Scheduler:
    def __init__(self, global_limit, service_quotas):
        """service_quotas: dict service_id -> positive int quota."""
        if global_limit < 1:
            raise ValueError("global_limit must be >= 1")
        self.global_limit = global_limit
        self.service_quotas = dict(service_quotas)
        self.global_sem = threading.Semaphore(global_limit)
        self.service_sems = {
            sid: threading.Semaphore(q)
            for sid, q in self.service_quotas.items()}
        self.global_gauge = _InflightGauge()
        self.service_gauges = {sid: _InflightGauge()
                               for sid in self.service_quotas}

    def _worker(self, req, gate_fn, decide_fn, forward_fn):
        trace = RequestTrace(seq=req.seq)
        trace.wait_start_ms = monotonic_ms()
        if gate_fn is not None:
            # Wait for the per-service ordered decision turn BEFORE
            # acquiring any quota semaphore. Acquisition order is then
            # turn -> global -> service everywhere, so no lock-ordering
            # deadlock is possible.
            gate_fn(req)
        self.global_sem.acquire()
        self.service_sems[req.service_id].acquire()
        self.global_gauge.acquire()
        self.service_gauges[req.service_id].acquire()
        try:
            trace.wait_end_ms = monotonic_ms()
            trace.decide_start_ms = monotonic_ms()
            decision = decide_fn(req)
            trace.decide_end_ms = monotonic_ms()
            trace.decision = decision
            if decision.passed:
                trace.fwd_start_ms = monotonic_ms()
                trace.receipt = forward_fn(req)
                trace.fwd_end_ms = monotonic_ms()
        finally:
            self.service_gauges[req.service_id].release()
            self.global_gauge.release()
            self.service_sems[req.service_id].release()
            self.global_sem.release()
        return trace

    def run(self, requests, decide_fn, forward_fn, gate_fn=None):
        """Execute all requests; returns traces ordered by seq."""
        traces = {}
        with ThreadPoolExecutor(
                max_workers=self.global_limit,
                thread_name_prefix="gw-worker") as pool:
            futures = [pool.submit(self._worker, req, gate_fn, decide_fn,
                                   forward_fn)
                       for req in requests]
            for fut in futures:
                trace = fut.result()
                traces[trace.seq] = trace
        return [traces[req.seq] for req in requests]
