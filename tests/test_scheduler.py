import os
import shutil
import sys
import tempfile
import unittest

from coljoin.cost import CostEstimator, choose_plan
from coljoin.errors import PortError
from coljoin.planner import enumerate_plans
from coljoin.protocol import find_free_port
from coljoin.scheduler import Scheduler
from coljoin.sharding import split_batches
from coljoin.loader import load_query, load_session, load_weights
from coljoin.storage import load_generated_specs
from scripts.generate_sample import main as generate_main


class SchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        old_argv = sys.argv
        sys.argv = ["generate_sample.py", "--workspace", cls.tmp]
        generate_main()
        sys.argv = old_argv
        cls.data_dir = os.path.join(cls.tmp, "data")
        cfg = os.path.join(cls.tmp, "configs")
        groups = load_generated_specs(cls.data_dir)
        cls.query = load_query(os.path.join(cfg, "query.json"), groups)
        weights = load_weights(os.path.join(cfg, "weights.json"))
        session = load_session(os.path.join(cfg, "session.json"))
        plans = enumerate_plans(cls.query)
        estimator = CostEstimator(groups, weights, session)
        evaluations = [estimator.evaluate(plan) for plan in plans]
        chosen = choose_plan(evaluations)["chosen_plan_id"]
        cls.plan = next(plan for plan in plans if plan["plan_id"] == chosen)
        cls.cost_vector = next(
            item for item in evaluations if item["plan_id"] == chosen
        )["vector"]
        by_id = {group.bg_id: group for group in groups}
        cls.batches = split_batches(
            cls.plan, by_id[cls.plan["body"]["order"][0]], 20
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _requests(self):
        return [
            {
                "type": "shard",
                "query_path": os.path.join(self.tmp, "configs", "query.json"),
                "plan": self.plan,
                "plan_version": 1,
                "executor_id": "executor-placeholder",
                "cost_vector": self.cost_vector,
                "batch": batch,
            }
            for batch in self.batches
        ]

    def test_real_tcp_roundtrip_and_constraints(self):
        requests = self._requests()
        with Scheduler(
            self.data_dir,
            executor_count=2,
            max_in_flight=2,
            per_executor_slots=1,
        ) as scheduler:
            for index, request in enumerate(requests):
                request["executor_id"] = scheduler.handles[
                    index % 2
                ].executor_id
            report = scheduler.run(requests)
        self.assertEqual(len(report.receipts), len(requests))
        self.assertLessEqual(report.max_global_in_flight, 2)
        for value in report.max_executor_in_flight.values():
            self.assertLessEqual(value, 1)
        for receipt in report.receipts:
            self.assertGreaterEqual(receipt["clock_end_ms"], receipt["clock_start_ms"])
            self.assertIn("executor-", receipt["executor_id"])

    def test_queueing_recorded_with_monotonic_clock(self):
        requests = self._requests()
        with Scheduler(
            self.data_dir,
            executor_count=1,
            max_in_flight=1,
            per_executor_slots=1,
        ) as scheduler:
            for request in requests:
                request["executor_id"] = scheduler.handles[0].executor_id
            report = scheduler.run(requests)
        self.assertEqual(len(report.receipts), len(requests))
        self.assertTrue(report.waits)
        for event in report.waits:
            self.assertGreaterEqual(event.wait_end_ms, event.wait_start_ms)
            self.assertIn("limit", event.reason)

    def test_declared_port_conflict_reports_business_code(self):
        port = find_free_port()
        import socket

        holder = socket.socket()
        holder.bind(("127.0.0.1", port))
        holder.listen(1)
        try:
            scheduler = Scheduler(
                self.data_dir,
                executor_count=1,
                max_in_flight=1,
                per_executor_slots=1,
                declared_ports=[port],
            )
            with self.assertRaises(PortError) as ctx:
                scheduler.start()
            self.assertEqual(ctx.exception.code, "E_PORT")
        finally:
            holder.close()


if __name__ == "__main__":
    unittest.main()
