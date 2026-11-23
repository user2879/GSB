from __future__ import annotations

import threading
import time
import unittest

from config_pipeline.manifest import manifest_from_dict
from config_pipeline.planner import compile_plan
from config_pipeline.scheduler import schedule_shards
from config_pipeline.util import digest_json
from tests.helpers import manifest


class SchedulerTests(unittest.TestCase):
    def _run(self, data: dict):
        parsed = manifest_from_dict(data)
        plan = compile_plan(parsed)
        lock = threading.Lock()
        state = {
            "global": 0,
            "peak_global": 0,
            "cluster": {cluster.id: 0 for cluster in parsed.clusters},
            "peak_cluster": {cluster.id: 0 for cluster in parsed.clusters},
            "task": 0,
            "peak_task": 0,
            "threads": set(),
        }

        def dispatch(shard):
            with lock:
                state["global"] += shard.resource_units
                state["cluster"][shard.cluster_id] += 1
                state["task"] += 1
                state["peak_global"] = max(state["peak_global"], state["global"])
                state["peak_cluster"][shard.cluster_id] = max(
                    state["peak_cluster"][shard.cluster_id],
                    state["cluster"][shard.cluster_id],
                )
                state["peak_task"] = max(state["peak_task"], state["task"])
                state["threads"].add(threading.current_thread().name)
            time.sleep(0.03)
            summary = {"shard": shard.stable_id, "value": 1}
            with lock:
                state["global"] -= shard.resource_units
                state["cluster"][shard.cluster_id] -= 1
                state["task"] -= 1
            return {
                "aggregate_summary": summary,
                "aggregate_digest": digest_json(summary),
                "elapsed_ms": 30,
            }

        report = schedule_shards(
            plan,
            dispatch,
            global_budget=parsed.global_resource_budget,
            cluster_quotas={c.id: c.concurrency for c in parsed.clusters},
        )
        return parsed, report, state

    def test_all_three_constraints_hold(self) -> None:
        data = manifest(
            ports=(41201, 41202),
            budget=3,
            task_shards=6,
            task_limit=2,
            cluster_quotas=(2, 1),
            resource_units=1,
        )
        parsed, report, state = self._run(data)
        self.assertLessEqual(state["peak_global"], 3)
        self.assertLessEqual(state["peak_cluster"]["c0"], 2)
        self.assertLessEqual(state["peak_cluster"]["c1"], 1)
        self.assertLessEqual(state["peak_task"], 2)
        self.assertGreaterEqual(state["peak_task"], 2)
        self.assertGreater(len(state["threads"]), 1)
        self.assertEqual(len(report.succeeded), 12)
        self.assertTrue(all(trace.wait_end_ms >= trace.wait_start_ms for trace in report.traces))
        self.assertTrue(any(trace.wait_ms > 0 for trace in report.traces))
        self.assertEqual(report.stats.max_observed_workers, 2)

    def test_global_budget_is_the_binding_limit(self) -> None:
        data = manifest(
            ports=(41211, 41212),
            budget=2,
            task_shards=4,
            task_limit=8,
            cluster_quotas=(8, 8),
            resource_units=1,
        )
        _, report, state = self._run(data)
        self.assertEqual(state["peak_global"], 2)
        self.assertEqual(report.stats.peak_resource_units, 2)

    def test_task_cap_is_the_binding_limit(self) -> None:
        data = manifest(
            ports=(41221, 41222),
            budget=10,
            task_shards=4,
            task_limit=1,
            cluster_quotas=(4, 4),
            resource_units=1,
        )
        _, report, state = self._run(data)
        self.assertEqual(state["peak_task"], 1)


if __name__ == "__main__":
    unittest.main()
