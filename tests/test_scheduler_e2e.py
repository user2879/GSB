import os
import shutil
import tempfile
import unittest

from sharding_diag.clock import MonotonicClock
from sharding_diag.ledger import Ledger
from sharding_diag.model import parse_declaration
from sharding_diag.recovery import rebuild_scheduler, replay_committed, rerun_unknown
from sharding_diag.scheduler import Scheduler


def declaration(budget=6):
    return parse_declaration({
        "nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": 19101, "concurrency": 1,
             "lease_limit": 1, "seed": 11, "capacity_version": 1},
            {"node_id": "n2", "host": "127.0.0.1", "port": 19102, "concurrency": 1,
             "lease_limit": 2, "seed": 22, "capacity_version": 1},
        ],
        "shards": [
            {"shard_id": "a", "rows": 20, "demand": 4, "priority": 5},
            {"shard_id": "b", "rows": 20, "demand": 4, "priority": 4},
            {"shard_id": "c", "depends_on": ["a"], "rows": 20, "demand": 3, "priority": 3},
            {"shard_id": "d", "depends_on": ["a", "b"], "rows": 20, "demand": 2, "priority": 2},
        ],
        "global": {"budget": budget, "freeze_threshold_ms": 500,
                   "revoke_tolerance": 5, "lease_ttl_ms": 30000},
    })


class SchedulerE2ETest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "e2e.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_constraints_queueing_and_deterministic_outcomes(self):
        decl = declaration()
        ledger = Ledger(self.db, decl).open()
        scheduler = Scheduler(decl, ledger, MonotonicClock())
        scheduler.start_nodes()
        try:
            result = scheduler.run()
            outcomes = ledger.outcomes()
            self.assertEqual(sorted(outcomes), ["a", "b", "c", "d"])
            self.assertTrue(ledger.verify_chain())
            # Budget 6 and demand 4: at least one shard must have queued.
            self.assertTrue(len(result["wait_records"]) >= 1)
            for record in result["wait_records"]:
                if record.get("wait_ms") is not None:
                    self.assertGreaterEqual(record["wait_ms"], 0)
                    self.assertIn(record["reason"],
                                  ("budget_exhausted", "node_slots_exhausted"))
            replay_committed(scheduler)
        finally:
            scheduler.stop_nodes()
            ledger.close()

    def test_hard_kill_marks_unknown_reclaims_budget_and_replays(self):
        decl = declaration()
        ledger = Ledger(self.db, decl).open()
        scheduler = Scheduler(decl, ledger, MonotonicClock())
        scheduler.start_nodes()
        try:
            scheduler.freeze_mgr.initialize(s.shard_id for s in decl.shards)
            scheduler.run_subset({"a", "b"})
            scheduler.start_pool()
            shard = decl.shard_map()["c"]
            migrations = scheduler.freeze_mgr.evaluate_round([shard], ledger.outcomes())
            for migration in migrations:
                ledger.record_freeze_migration(migration)
            node = next(n for n in decl.nodes if n.node_id == "n1")
            scheduler._dispatch(shard, node)
            scheduler._nodes["n1"].kill9()
            scheduler.shutdown_pool()
            # Intentionally do NOT harvest the future: the in-flight lease has
            # no committed outcome, which is exactly the hard-kill condition.

            recovery = rebuild_scheduler(scheduler)
            self.assertEqual([item["shard_id"] for item in recovery["unknown_shards"]], ["c"])
            self.assertEqual(recovery["unknown_shards"][0]["budget_reclaimed"],
                             decl.shard_map()["c"].demand)
            self.assertGreater(recovery["total_budget_reclaimed"], 0)
            self.assertEqual(recovery["budget_used_after_recovery"], 0)

            scheduler.start_nodes()
            rerun = rerun_unknown(scheduler)
            self.assertEqual(rerun["reran"], ["c"])
            replay_committed(scheduler)
            self.assertEqual(sorted(ledger.outcomes()), ["a", "b", "c"])
            self.assertTrue(ledger.verify_chain())
        finally:
            scheduler.shutdown_pool()
            scheduler.stop_nodes()
            ledger.close()


if __name__ == "__main__":
    unittest.main()
