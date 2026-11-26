from __future__ import annotations

import threading
import time
import unittest

from gsb.leases import LeaseManager
from gsb.model import BatchSpec
from gsb.scheduler import QuotaScheduler, SchedulerJob
from tests.helpers import make_session, make_shard


class SchedulerTests(unittest.TestCase):
    def test_candidate_order_is_shard_then_batch_order(self) -> None:
        shard_a = make_shard("a", lease_limit=1, concurrency_limit=1)
        shard_b = make_shard("b", lease_limit=1, concurrency_limit=1)
        batches = (
            BatchSpec("a-9", "a", (9,), 1, 9, waterline=100),
            BatchSpec("b-3", "b", (3,), 1, 3, waterline=100),
            BatchSpec("a-1", "a", (1,), 1, 1, waterline=100),
        )
        session = make_session((shard_a, shard_b), batches=batches, global_budget=1)
        leases = LeaseManager()
        scheduler = QuotaScheduler(session, leases, max_workers=2)
        started: list[tuple[str, int]] = []
        jobs = [
            SchedulerJob(
                batch=batch,
                shard=session.shard(batch.shard_id),
                parameter_version=session.shard(batch.shard_id).parameter_version,
                parameter_digest="d" * 64,
                error_bound=10,
            )
            for batch in batches
        ]

        outcomes = scheduler.run(
            jobs,
            lambda job, lease: {"loss": 1},
            lambda job: leases.issue(job.shard.shard_id, 30000),
            on_start=lambda outcome: started.append(
                (outcome.job.shard.shard_id, outcome.job.batch.order)
            ),
        )
        self.assertEqual(started, [("a", 1), ("a", 9), ("b", 3)])
        self.assertEqual(
            [(outcome.job.shard.shard_id, outcome.job.batch.order) for outcome in outcomes],
            [("a", 1), ("a", 9), ("b", 3)],
        )

    def test_triple_quotas_and_real_thread_pool(self) -> None:
        shard_a = make_shard("a", lease_limit=2, concurrency_limit=2)
        shard_b = make_shard("b", lease_limit=1, concurrency_limit=1)
        batches = []
        order = 1
        for shard in (shard_a, shard_b):
            for index in range(4):
                batches.append(
                    BatchSpec(
                        f"{shard.shard_id}-{index}",
                        shard.shard_id,
                        (index,),
                        1,
                        order,
                        waterline=100 + index,
                    )
                )
                order += 1
        session = make_session(
            (shard_a, shard_b), batches=tuple(batches), global_budget=2
        )
        leases = LeaseManager()
        scheduler = QuotaScheduler(session, leases, max_workers=3)
        active_units = 0
        per_shard: dict[str, int] = {"a": 0, "b": 0}
        observed = {"units": 0, "a": 0, "b": 0}
        lock = threading.Lock()

        jobs = [
            SchedulerJob(
                batch=batch,
                shard=session.shard(batch.shard_id),
                parameter_version=session.shard(batch.shard_id).parameter_version,
                parameter_digest="d" * 64,
                error_bound=10,
            )
            for batch in batches
        ]

        def execute(job, lease):
            nonlocal active_units
            with lock:
                active_units += job.batch.resource_units
                per_shard[job.shard.shard_id] += 1
                observed["units"] = max(observed["units"], active_units)
                for shard_id in per_shard:
                    observed[shard_id] = max(observed[shard_id], per_shard[shard_id])
            time.sleep(0.02)
            with lock:
                active_units -= job.batch.resource_units
                per_shard[job.shard.shard_id] -= 1
            return {"loss": 1, "loss_digest": "x", "parameter_digest": "d" * 64}

        outcomes = scheduler.run(
            jobs,
            execute,
            lambda job: leases.issue(job.shard.shard_id, 30000),
        )
        self.assertEqual(len(outcomes), 8)
        self.assertTrue(all(outcome.status == "committed" for outcome in outcomes))
        self.assertEqual(observed["units"], 2)
        self.assertLessEqual(observed["a"], 2)
        self.assertLessEqual(observed["b"], 1)
        self.assertTrue(
            all(outcome.thread_name and outcome.thread_name.startswith("gsb-batch") for outcome in outcomes)
        )
        high = scheduler.quota_high_watermarks
        self.assertEqual(high["max_global_units"], 2)
        self.assertLessEqual(high["max_leases_by_shard"]["a"], 2)
        self.assertLessEqual(high["max_leases_by_shard"]["b"], 1)
        self.assertTrue(any(outcome.error_affected for outcome in outcomes))
        self.assertTrue(all(outcome.wait_end_ms >= outcome.wait_start_ms for outcome in outcomes))


if __name__ == "__main__":
    unittest.main()
