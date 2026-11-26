"""Lease-gated parallel batch scheduler on a real thread pool.

Every in-flight batch holds exactly one lease.  A dispatcher loop admits
queued batches only when both the global active-lease ceiling and the worker
slot ceiling allow it.  Wait time is measured with the same monotonic clock as
lease deadlines, and there is no dispatch path that bypasses the lease.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import threading

from .fixedpoint import parameter_digest
from .lease import MonotonicClock
from .receipt import decode, normalize


class Scheduler:
    def __init__(self, spec, plan, pool, *, version_map=None,
                 format_map=None, clock=None, max_workers=None):
        self.spec = spec
        self.plan = plan
        self.pool = pool
        self.version_map = version_map or {}
        self.format_map = format_map or {}
        self.clock = clock or MonotonicClock()
        self.max_workers = max_workers or max(
            1, spec.lease_policy.global_max_active
        )

    def _execute_batch(self, lease_manager, batch):
        worker = self.spec.worker_by_id[batch.worker_id]
        param_version = int(self.version_map.get(batch.batch_id, 0))
        fmt = self.format_map.get(batch.batch_id, "r3")
        param_digest = parameter_digest(
            worker.worker_id, worker.model_dim, worker.hidden_dim,
            param_version, self.plan.seed,
        )
        lease = lease_manager.grant(batch.worker_id, batch.batch_id)
        try:
            payload = {
                "op": "batch",
                "batch_id": batch.batch_id,
                "request_ids": list(batch.request_ids),
                "param_version": param_version,
                "seed": self.plan.seed,
                "format": fmt,
            }
            raw_receipt = self.pool.send_batch(batch.worker_id, payload)
            decoded = decode(
                raw_receipt,
                known_request_ids=set(
                    rid for b in self.plan.batches
                    for rid in b.request_ids
                ),
            )
            normalized = normalize(
                decoded,
                worker_id=worker.worker_id,
                param_version=param_version,
                param_digest=param_digest,
            )
            return {
                "batch": batch,
                "lease_id": lease.lease_id,
                "raw": raw_receipt,
                "normalized": normalized,
                "status": "committed",
            }
        finally:
            # Cancel is idempotent against expiry; exactly one cause owns it.
            lease_manager.cancel(lease.lease_id)

    def run(self, lease_manager, progress=None):
        """Dispatch all batches and return per-batch execution records."""
        pending = list(self.plan.batches)
        admitted = {}
        results = {}
        wait_start = {}
        records = []
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while pending or admitted:
                lease_manager.sweep_expired()
                progressed = False
                remaining = []
                for batch in pending:
                    if batch.batch_id not in wait_start:
                        wait_start[batch.batch_id] = self.clock.now_ms()
                    if lease_manager.can_grant(batch.worker_id):
                        wait_end = self.clock.now_ms()
                        future = executor.submit(
                            self._execute_batch, lease_manager, batch
                        )
                        admitted[batch.batch_id] = future
                        records.append(
                            {
                                "batch_id": batch.batch_id,
                                "worker_id": batch.worker_id,
                                "wait_start_ms": wait_start[batch.batch_id],
                                "wait_end_ms": wait_end,
                                "wait_ms": wait_end
                                - wait_start[batch.batch_id],
                            }
                        )
                        progressed = True
                    else:
                        remaining.append(batch)
                pending = remaining

                done_ids = [
                    batch_id for batch_id, future in admitted.items()
                    if future.done()
                ]
                if not done_ids and not progressed and admitted:
                    # Wait briefly for an in-flight completion so queued
                    # batches accumulate real monotonic wait time, then loop
                    # to re-evaluate expiry and capacity.
                    try:
                        next(iter(admitted.values())).result(timeout=0.02)
                    except FutureTimeout:
                        time.sleep(0.005)
                    continue
                for batch_id in done_ids:
                    future = admitted.pop(batch_id)
                    result = future.result()
                    with lock:
                        results[batch_id] = result
                        if progress is not None:
                            progress.advance(batch_id, result)
            lease_manager.sweep_expired()

        ordered = [results[batch.batch_id] for batch in self.plan.batches]
        return {"results": ordered, "waits": records}
