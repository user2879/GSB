import sqlite3
import tempfile
import unittest
from pathlib import Path

from pool_workbench.app import WorkbenchSession
from pool_workbench.config import PoolConfig, Task
from pool_workbench.planner import build_plan
from pool_workbench.pool import ConnectionPool
from pool_workbench.sample_db import build_sample_db, open_connection
from pool_workbench.scheduler import (
    OUTCOME_EXEC_ERROR,
    OUTCOME_GET_TIMEOUT,
    OUTCOME_HOLD_TIMEOUT,
    OUTCOME_SUCCESS,
    run_batch,
)
from pool_workbench.structured_log import StructuredLogger


COLD_CONFIG = Path(__file__).resolve().parent.parent / "samples" / "tasks_cold.json"


class SchedulerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.db = self.workspace / "sample.sqlite"
        build_sample_db(self.db)
        self.session = WorkbenchSession(self.workspace)
        self.session.load(COLD_CONFIG, self.db)

    def tearDown(self):
        self.session.close()
        self.tmp.cleanup()

    def test_four_outcome_categories_and_counts(self):
        result = self.session.execute()
        counts = result.outcome_counts
        self.assertGreater(counts[OUTCOME_SUCCESS], 0)
        self.assertEqual(counts[OUTCOME_GET_TIMEOUT], 1)
        self.assertEqual(counts[OUTCOME_HOLD_TIMEOUT], 1)
        self.assertEqual(counts[OUTCOME_EXEC_ERROR], 1)
        self.assertEqual(sum(counts.values()), len(self.session.config.tasks))

    def test_every_task_has_a_matching_trace(self):
        result = self.session.execute()
        task_ids = {task.task_id for task in self.session.config.tasks}
        trace_ids = {trace.task_id for trace in result.traces}
        self.assertEqual(task_ids, trace_ids)
        self.assertEqual(len(result.traces), len(task_ids))

    def test_runtime_matches_deterministic_plan(self):
        plan = build_plan(self.session.config.pool, self.session.config.tasks)
        expected = {t.task_id: (t.outcome, t.slot_seq, t.wait_ms) for t in plan.tasks}
        result = self.session.execute()
        for trace in result.traces:
            # Data-dependent EXEC_ERROR (t12) cannot be predicted by the planner;
            # every other task must match outcome, slot and wait exactly.
            if trace.outcome == OUTCOME_EXEC_ERROR:
                self.assertEqual(expected[trace.task_id][0], "SUCCESS")
                continue
            self.assertEqual(
                (trace.outcome, trace.slot_seq, trace.wait_ms),
                expected[trace.task_id],
                trace.task_id,
            )

    def test_pool_cap_and_workload_quotas_never_exceeded(self):
        result = self.session.execute()
        self.assertEqual(result.constraint_violations, [])
        plan = result.plan
        self.assertLessEqual(plan.peak_active, self.session.config.pool.max_connections)
        for workload, peak in plan.peak_workload.items():
            self.assertLessEqual(
                peak, self.session.config.pool.workload_quotas[workload]
            )

    def test_get_timeout_never_occupies_connection(self):
        result = self.session.execute()
        timed_out = [t for t in result.traces if t.outcome == OUTCOME_GET_TIMEOUT]
        self.assertTrue(timed_out)
        for trace in timed_out:
            self.assertIsNone(trace.slot_seq)
            self.assertEqual(trace.error_code, "E_GETTIMEOUT")

    def test_hold_timeout_classified_and_connection_kept(self):
        result = self.session.execute()
        hold = next(t for t in result.traces if t.outcome == OUTCOME_HOLD_TIMEOUT)
        self.assertEqual(hold.error_code, "E_HOLDTIMEOUT")
        self.assertIsNotNone(hold.slot_seq)
        self.assertEqual(result.rebuild_count, 1)  # only t12 rebuilds


class ConnectionRebuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sample.sqlite"
        build_sample_db(self.db)
        self.config = PoolConfig(
            max_connections=2,
            workload_quotas={"wl_a": 2},
            hold_timeout_ms=100,
            acquire_timeout_ms=50,
        )
        self.tasks = [
            Task("a", "wl_a", "READ_WRITE", 10,
                 "INSERT INTO ops_log (workload, task_id, note) SELECT 'wl_a','a','x'"),
            Task("b", "wl_a", "READ_ONLY", 10, "SELECT item_id FROM inventory WHERE item_id=1"),
        ]
        self.logger = StructuredLogger(None)
        self.pool = ConnectionPool(self.config, str(self.db), open_connection, self.logger)

    def tearDown(self):
        self.pool.close()
        self.tmp.cleanup()

    def test_dead_connection_is_removed_and_rebuilt_without_overshoot(self):
        def close_connection(conn):
            conn.close()

        result = run_batch(self.config, self.tasks, self.pool, self.logger,
                           failure_injector=close_connection)
        self.assertEqual(result.rebuild_count, 2)
        self.assertLessEqual(self.pool.live_count(), self.config.max_connections)
        kinds = [event.kind for event in self.pool.events()]
        self.assertIn("connection_failed", kinds)
        self.assertIn("connection_rebuilt", kinds)
        self.assertEqual(result.constraint_violations, [])
        for trace in result.traces:
            self.assertEqual(trace.outcome, OUTCOME_EXEC_ERROR)
            self.assertIsNotNone(trace.rebuilt_from_seq)


class PoolAcquisitionGuardTests(unittest.TestCase):
    def test_acquire_respects_workload_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "s.sqlite"
            build_sample_db(db)
            config = PoolConfig(2, {"wl_a": 1, "wl_b": 1}, 100, 50)
            pool = ConnectionPool(config, str(db), open_connection, StructuredLogger(None))
            pool.ensure_capacity("READ_WRITE")
            pool.ensure_capacity("READ_WRITE")
            first = pool.acquire("a", "wl_a", "READ_WRITE")
            with self.assertRaises(RuntimeError):
                pool.acquire("b", "wl_a", "READ_WRITE")
            # wl_b still has quota headroom on a separate connection.
            second = pool.acquire("c", "wl_b", "READ_WRITE")
            pool.release(first, "wl_a")
            pool.release(second, "wl_b")
            pool.close()


if __name__ == "__main__":
    unittest.main()
