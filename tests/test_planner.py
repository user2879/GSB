from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from replica_repair.errors import BudgetError
from replica_repair.planner import RepairBudget, RepairExecutor, plan_repairs
from replica_repair.truth import TruthEngine
from replica_repair.sample import init_sample
from tests.helpers import group_servers


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        init_sample(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_budget_validation_and_oversized_task(self) -> None:
        with self.assertRaises(BudgetError) as caught:
            RepairBudget(max_parallel=0, per_node_kb=3).validate()
        self.assertEqual(caught.exception.code, "E_BUDGET")
        self.assertEqual(caught.exception.path, "$.budget.max_parallel")

        with self.assertRaises(BudgetError) as caught:
            RepairBudget(max_parallel=2, per_node_kb=0).validate()
        self.assertEqual(caught.exception.code, "E_BUDGET")

        with group_servers(self.root) as (manifest, clients, _servers):
            truth = TruthEngine(manifest).run(clients)
            with self.assertRaises(BudgetError) as caught:
                plan_repairs(manifest, truth, RepairBudget(max_parallel=3, per_node_kb=1))
            self.assertEqual(caught.exception.code, "E_BUDGET")
            self.assertEqual(caught.exception.details["block_id"], "blk-0002")

    def test_plan_queues_on_quota_and_executes_atomically(self) -> None:
        with group_servers(self.root) as (manifest, clients, _servers):
            truth = TruthEngine(manifest).run(clients)
            budget = RepairBudget(max_parallel=3, per_node_kb=3, cost_per_kb=2)
            plan = plan_repairs(manifest, truth, budget)
            self.assertEqual(plan.total_repair_kb, 10)
            self.assertEqual(plan.total_cost_units, 20)
            self.assertEqual(plan.waves, (("rep-0001", "rep-0002", "rep-0003"), ("rep-0004",)))
            self.assertEqual(len(plan.queue_events), 1)
            self.assertEqual(plan.queue_events[0].reason, "per_node_quota")
            self.assertEqual(plan.queue_events[0].target_node, "v3-d")
            for wave in plan.waves:
                wave_tasks = [task for task in plan.tasks if task.task_id in wave]
                self.assertLessEqual(len(wave_tasks), budget.max_parallel)
                per_node: dict[str, int] = {}
                for task in wave_tasks:
                    per_node[task.target_node] = per_node.get(task.target_node, 0) + task.repair_kb
                self.assertTrue(all(kb <= budget.per_node_kb for kb in per_node.values()))

            progress: list[tuple[str, int]] = []
            executor = RepairExecutor(
                manifest,
                clients,
                progress_hook=lambda result, count: progress.append((result.task_id, count)),
            )
            execution = executor.execute(plan)
            self.assertEqual(execution.completed_tasks, 4)
            self.assertEqual(len(progress), 4)
            self.assertEqual([item[1] for item in progress], [1, 2, 3, 4])
            for result in execution.results:
                observed = clients[result.target_node].call("get_digest", block_id=result.block_id)
                self.assertEqual(observed["actual_digest"], result.digest)
                self.assertEqual(observed["size_bytes"], result.bytes_written)
                directory = self.root / "nodes" / result.target_node / "blocks"
                self.assertEqual(list(directory.glob("*.tmp")), [])
            self.assertTrue(all(temp.suffix != ".tmp" for node in self.root.glob("nodes/*") for temp in node.rglob("*")))


if __name__ == "__main__":
    unittest.main()
