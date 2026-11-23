from __future__ import annotations

import hashlib
import threading
import time
import unittest

from gateway_diag.backends import InferenceReceipt
from gateway_diag.model import RequestSample
from gateway_diag.routing import BatchPlan, PlannedRequest, RouteDecision
from gateway_diag.scheduling import QuotaScheduler


class FakeRunner:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.global_running = 0
        self.backend_running = {"a": 0, "b": 0}
        self.max_global = 0
        self.max_backend = {"a": 0, "b": 0}
        self.threads: set[str] = set()

    def infer(self, backend_id, batch_id, items):
        materialized = list(items)
        with self.lock:
            self.global_running += 1
            self.backend_running[backend_id] += 1
            self.max_global = max(self.max_global, self.global_running)
            self.max_backend[backend_id] = max(
                self.max_backend[backend_id], self.backend_running[backend_id]
            )
            self.threads.add(threading.current_thread().name)
        time.sleep(0.08)
        with self.lock:
            self.global_running -= 1
            self.backend_running[backend_id] -= 1
        request_ids = tuple(request_id for request_id, _vector in materialized)
        digests = {
            request_id: hashlib.sha256(f"{backend_id}:{request_id}".encode()).hexdigest()
            for request_id in request_ids
        }
        raw = {
            "type": "receipt",
            "backend_id": backend_id,
            "parameter_digest": "a" * 64,
            "batch_id": batch_id,
            "request_ids": list(request_ids),
            "output_digests": {str(key): value for key, value in digests.items()},
            "output_digest": "b" * 64,
            "elapsed_ms": 1,
            "worker_pid": 123,
        }
        return InferenceReceipt(
            backend_id=backend_id,
            port=1,
            parameter_digest="a" * 64,
            batch_id=batch_id,
            request_ids=request_ids,
            output_digests=digests,
            output_digest="b" * 64,
            elapsed_ms=1,
            worker_pid=123,
            receipt_digest="c" * 64,
            raw=raw,
        )


def make_batch(request_id: int, backend_id: str) -> BatchPlan:
    sample = RequestSample(
        request_id=request_id,
        client_key=f"alpha:{request_id}",
        input_vector=(0.1, 0.2, 0.3, 0.4),
        arrival_ms=1000 + request_id,
    )
    decision = RouteDecision(
        request_id=request_id,
        client_key=sample.client_key,
        arrival_ms=sample.arrival_ms,
        rule_id="alpha",
        target="stable" if backend_id == "a" else "canary",
        backend_id=backend_id,
        bucket=request_id,
        weight_bp=5000,
        weight_version=1,
    )
    return BatchPlan(
        batch_id=f"batch-{request_id}",
        revision=1,
        rule_id="alpha",
        backend_id=backend_id,
        target=decision.target,
        batch_ordinal=request_id,
        items=(PlannedRequest(sample=sample, decision=decision),),
    )


class SchedulerTests(unittest.TestCase):
    def test_global_and_backend_quotas_are_enforced_on_real_threads(self) -> None:
        runner = FakeRunner()
        completed: list[str] = []
        scheduler = QuotaScheduler(
            runner,
            {"a": 1, "b": 1},
            2,
            on_complete=lambda execution: completed.append(execution.plan.batch_id),
        )
        batches = [
            make_batch(1, "a"),
            make_batch(2, "b"),
            make_batch(3, "a"),
            make_batch(4, "b"),
        ]
        executions, summary = scheduler.run(batches)
        self.assertEqual(len(executions), 4)
        self.assertEqual(runner.max_global, 2)
        self.assertEqual(runner.max_backend, {"a": 1, "b": 1})
        self.assertEqual(summary.max_global_observed, 2)
        self.assertEqual(summary.max_backend_observed, {"a": 1, "b": 1})
        self.assertEqual(set(completed), {batch.batch_id for batch in batches})
        self.assertTrue(runner.threads)
        self.assertTrue(all("gateway-batch" in name for name in runner.threads))
        for execution in executions:
            self.assertLessEqual(execution.wait_start_ms, execution.wait_end_ms)
            self.assertLessEqual(execution.wait_end_ms, execution.exec_start_ms)
            self.assertLessEqual(execution.exec_start_ms, execution.exec_end_ms)
            self.assertGreaterEqual(execution.wait_ms, 0)
        self.assertTrue(any(execution.wait_ms > 0 for execution in executions))


if __name__ == "__main__":
    unittest.main()
