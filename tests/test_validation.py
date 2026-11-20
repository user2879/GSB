import json
import os
import tempfile
import unittest

from gpu_scheduler.errors import (
    E_CYCLE, E_DEPEND, E_MEMORY, E_QUOTA, E_SPEC, SchedulerError,
)
from gpu_scheduler.loader import load_declaration

SPEC = os.path.join("samples", "specs")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.session = json.load(open(os.path.join(SPEC, "session.json")))
        self.pools = json.load(open(os.path.join(SPEC, "pools.json")))
        self.tasks = json.load(open(os.path.join(SPEC, "tasks.json")))

    def _dump(self):
        paths = {
            "s": os.path.join(self.tmp, "session.json"),
            "p": os.path.join(self.tmp, "pools.json"),
            "t": os.path.join(self.tmp, "tasks.json"),
        }
        json.dump(self.session, open(paths["s"], "w"))
        json.dump(self.pools, open(paths["p"], "w"))
        json.dump(self.tasks, open(paths["t"], "w"))
        return paths

    def test_valid_declaration_loads(self):
        paths = self._dump()
        declaration = load_declaration(paths["s"], paths["p"], paths["t"])
        self.assertGreaterEqual(len(declaration.pools), 2)
        self.assertEqual(
            sum(pool.quota_shares for pool in declaration.pools),
            declaration.config.total_quota_shares)

    def test_duplicate_task_id_rejected_with_path(self):
        self.tasks["tasks"][1]["task_id"] = self.tasks["tasks"][0]["task_id"]
        paths = self._dump()
        with self.assertRaises(SchedulerError) as ctx:
            load_declaration(paths["s"], paths["p"], paths["t"])
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertTrue(any(".task_id" in p.path for p in ctx.exception.problems))

    def test_negative_integer_rejected(self):
        self.pools["pools"][0]["gpu_count"] = -2
        paths = self._dump()
        with self.assertRaises(SchedulerError):
            load_declaration(paths["s"], paths["p"], paths["t"])

    def test_quota_share_sum_violation(self):
        self.pools["pools"][0]["quota_shares"] += 5
        paths = self._dump()
        with self.assertRaises(SchedulerError) as ctx:
            load_declaration(paths["s"], paths["p"], paths["t"])
        self.assertIn(E_QUOTA, {p.code for p in ctx.exception.problems})

    def test_memory_budget_violation(self):
        task = self.tasks["tasks"][0]
        task["memory_budget_mib"] = task["gpu_need"] * \
            task["memory_per_gpu_mib"] - 1
        paths = self._dump()
        with self.assertRaises(SchedulerError) as ctx:
            load_declaration(paths["s"], paths["p"], paths["t"])
        self.assertIn(E_MEMORY, {p.code for p in ctx.exception.problems})

    def test_shard_cycle_rejected(self):
        task = next(t for t in self.tasks["tasks"] if t["shard_count"] >= 2)
        task["shards"][0]["depends_on"] = [1]
        task["shards"][1]["depends_on"] = [0]
        paths = self._dump()
        with self.assertRaises(SchedulerError) as ctx:
            load_declaration(paths["s"], paths["p"], paths["t"])
        self.assertIn(E_CYCLE, {p.code for p in ctx.exception.problems})

    def test_missing_task_reference_rejected(self):
        self.tasks["tasks"][-1]["depends_on_tasks"] = ["does-not-exist"]
        paths = self._dump()
        with self.assertRaises(SchedulerError) as ctx:
            load_declaration(paths["s"], paths["p"], paths["t"])
        self.assertIn(E_DEPEND, {p.code for p in ctx.exception.problems})

    def test_bool_is_not_integer(self):
        self.pools["pools"][0]["gpu_count"] = True
        paths = self._dump()
        with self.assertRaises(SchedulerError):
            load_declaration(paths["s"], paths["p"], paths["t"])


if __name__ == "__main__":
    unittest.main()
