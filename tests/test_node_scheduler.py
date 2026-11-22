import tempfile
import unittest
from pathlib import Path

from configreg.manifest import NodeSpec, Registry
from configreg.node import deterministic_summary
from configreg.nodepool import NodePool
from configreg.scheduler import CollectTask, Scheduler


class DeterminismTests(unittest.TestCase):
    def test_summary_deterministic_across_calls(self) -> None:
        first = deterministic_summary("node_a", "item", 1.25, sample_seed=99, work_repeat=2)
        second = deterministic_summary("node_a", "item", 1.25, sample_seed=99, work_repeat=2)
        self.assertEqual(first["group_mean"], second["group_mean"])
        self.assertEqual(first["checksum"], second["checksum"])

    def test_value_changes_summary_but_seed_fixes_samples(self) -> None:
        baseline = deterministic_summary("node_a", "item", 1.0, sample_seed=7, work_repeat=1)
        candidate = deterministic_summary("node_a", "item", 5.0, sample_seed=7, work_repeat=1)
        self.assertNotEqual(baseline["group_mean"], candidate["group_mean"])
        self.assertEqual(baseline["sample_count"], candidate["sample_count"])


class SchedulerSubprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.nodes = {
            "node_a": NodeSpec("node_a", "127.0.0.1", 24001, 0.0),
            "node_b": NodeSpec("node_b", "127.0.0.1", 24002, 0.0),
            "flaky": NodeSpec("flaky", "127.0.0.1", 24003, 0.5),
        }
        self.registry = Registry(self.nodes)
        self.pool = NodePool(self.registry, seed=1234, log_dir=Path(self.tmp.name) / "logs")
        self.pool.start_all()

    def tearDown(self) -> None:
        self.pool.stop_all()
        self.tmp.cleanup()

    def _task(self, index: int, node_id: str) -> CollectTask:
        return CollectTask(
            task_key=f"task-{index}",
            node=self.nodes[node_id],
            payload={
                "type": "slice",
                "node_id": node_id,
                "version": 1,
                "item": "item",
                "value": 1.0,
                "sample_seed": 1234,
                "work_repeat": 1,
            },
        )

    def test_concurrency_limits_and_retry_isolation(self) -> None:
        tasks = [self._task(index, "node_a" if index % 2 else "node_b") for index in range(8)]
        scheduler = Scheduler(
            global_concurrency=2,
            per_node_concurrency=1,
            socket_timeout_ms=500,
            max_attempts=8,
            retry_backoff_ms=10,
        )
        results = scheduler.run(tasks)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(result.ok for result in results))
        self.assertEqual(len({result.task_key for result in results}), 8)

    def test_flaky_node_retries_with_distinct_request_ids(self) -> None:
        tasks = [self._task(index, "flaky") for index in range(2)]
        scheduler = Scheduler(1, 1, socket_timeout_ms=200, max_attempts=12, retry_backoff_ms=5)
        results = scheduler.run(tasks)
        self.assertTrue(all(result.ok for result in results))
        retried = [result for result in results if len(result.attempts) > 1]
        self.assertTrue(retried, "高丢包率节点应当至少触发一次真实超时重试")
        for result in retried:
            request_ids = [attempt.request_id for attempt in result.attempts]
            self.assertEqual(len(request_ids), len(set(request_ids)))
            self.assertTrue(any(attempt.outcome == "timeout" for attempt in result.attempts))
            self.assertEqual(result.attempts[-1].outcome, "ok")


if __name__ == "__main__":
    unittest.main()
