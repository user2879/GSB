import json
import os
import random
import tempfile
import unittest

from gpu_scheduler.engine import SimulationEngine
from gpu_scheduler.loader import load_declaration
from gpu_scheduler.model import (
    FAILED, INVALIDATED, SUCCESS, canonical_json,
)

SPEC = os.path.join("samples", "specs")


def load_sample():
    return load_declaration(
        os.path.join(SPEC, "session.json"),
        os.path.join(SPEC, "pools.json"),
        os.path.join(SPEC, "tasks.json"))


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.declaration = load_sample()
        self.engine = SimulationEngine(self.declaration)
        self.engine.run_until()

    def test_respects_global_gpu_and_quota_constraints(self):
        config = self.declaration.config
        for record in self.engine.rt.rounds:
            self.assertLessEqual(
                record.global_gpu_busy, config.global_gpu_cap)
            self.assertLessEqual(
                record.global_shares_busy, config.total_quota_shares)
            for usage in record.pool_usage:
                self.assertLessEqual(usage.gpu_busy, usage.gpu_capacity)
                self.assertLessEqual(usage.memory_busy_mib,
                                     usage.memory_capacity_mib)
                self.assertLessEqual(usage.shares_busy, usage.shares_capacity)
                self.assertLessEqual(usage.concurrent_tasks,
                                     usage.concurrent_cap)

    def test_per_task_memory_budget_never_exceeded(self):
        tasks = self.declaration.task_map()
        for record in self.engine.rt.rounds:
            task_mem = {}
            for shard_id in record.running:
                task_id, shard_label = shard_id.split("#S")
                index = int(shard_label)
                shard = self.declaration.shard_index()[(task_id, index)]
                task_mem[task_id] = task_mem.get(task_id, 0) + \
                    shard.total_memory_mib
            for task_id, used in task_mem.items():
                self.assertLessEqual(
                    used, tasks[task_id].memory_budget_mib)

    def test_deterministic_under_shuffled_input(self):
        baseline = [r.to_dict() for r in self.engine.rt.rounds]
        tasks = json.load(open(os.path.join(SPEC, "tasks.json")))
        pools = json.load(open(os.path.join(SPEC, "pools.json")))
        rng = random.Random(7)
        rng.shuffle(tasks["tasks"])
        for task in tasks["tasks"]:
            rng.shuffle(task["shards"])
        rng.shuffle(pools["pools"])
        tmp = tempfile.mkdtemp()
        tp = os.path.join(tmp, "tasks.json")
        pp = os.path.join(tmp, "pools.json")
        sp = os.path.join(SPEC, "session.json")
        json.dump(tasks, open(tp, "w"))
        json.dump(pools, open(pp, "w"))
        shuffled = SimulationEngine(load_declaration(sp, pp, tp))
        shuffled.run_until()
        self.assertEqual(
            canonical_json(baseline),
            canonical_json([r.to_dict() for r in shuffled.rt.rounds]))

    def test_preemption_priority_non_decreasing_and_recoverable(self):
        preemptions = self.engine.rt.preemptions
        self.assertTrue(preemptions, "sample must trigger preemption")
        for record in preemptions:
            self.assertGreater(record.trigger_priority,
                               record.victim_priority)
            self.assertGreaterEqual(record.released_gpu, 1)
            self.assertGreater(record.released_memory_mib, 0)
        # Victim task eventually reaches success again after being suspended.
        victim_ids = {record.victim_task_id for record in preemptions}
        states = self.declaration.shard_index()
        for victim in victim_ids:
            victim_shards = [shard for key, shard in states.items()
                             if key[0] == victim]
            self.assertTrue(any(shard.state == SUCCESS
                                for shard in victim_shards))

    def test_preemption_sequence_byte_stable(self):
        first = canonical_json(
            [p.to_dict() for p in self.engine.rt.preemptions])
        second_engine = SimulationEngine(load_sample())
        second_engine.run_until()
        second = canonical_json(
            [p.to_dict() for p in second_engine.rt.preemptions])
        self.assertEqual(first, second)

    def test_failure_propagates_through_intra_and_cross_task_graph(self):
        states = self.declaration.shard_index()
        # train-alpha shard 1 fails; shard 2 must invalidate.
        self.assertEqual(states[("train-alpha", 1)].state, FAILED)
        self.assertEqual(states[("train-alpha", 2)].state, INVALIDATED)
        # train-gamma depends on train-alpha and must be invalidated.
        self.assertTrue(all(
            states[("train-gamma", idx)].state == INVALIDATED
            for idx in range(2)))

    def test_successful_shard_versions_and_digests_preserved(self):
        before = {
            key: (shard.version, shard.output_digest)
            for key, shard in self.declaration.shard_index().items()
            if shard.state == SUCCESS
        }
        self.assertTrue(before)
        # Re-running read-only summaries must not mutate preserved outputs.
        summary = self.engine.shard_summary()
        for item in summary:
            if item["state"] == SUCCESS:
                self.assertEqual(
                    before[(item["task_id"], item["shard_index"])],
                    (item["version"], item["output_digest"]))

    def test_every_shard_has_timing_and_wait_fields(self):
        for item in self.engine.shard_summary():
            if item["state"] in (SUCCESS,):
                self.assertIsNotNone(item["started_ms"])
                self.assertIsNotNone(item["ended_ms"])
                self.assertGreaterEqual(item["wait_duration_ms"], 0)


if __name__ == "__main__":
    unittest.main()
