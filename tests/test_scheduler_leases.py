import tempfile
import time
import unittest
from pathlib import Path

from shard_diag.engine import ExecutionEngine
from shard_diag.errors import LeaseError
from shard_diag.lease import LeaseRegistry
from shard_diag.ledger import Ledger
from shard_diag.node_manager import NodeManager
from shard_diag.protocol import find_free_port
from shard_diag.spec import Event, NodeSpec, RuntimeConfig, ShardSpec, Spec


def make_spec(
    *,
    budget=100,
    concurrency=1,
    max_leases=1,
    ttl_ms=10_000,
    shard_units=(1, 1),
):
    node = NodeSpec("n1", "127.0.0.1", find_free_port(), concurrency, max_leases, 99, 1)
    shards = tuple(
        ShardSpec(f"s{i}", (), 16, units, 10 - i)
        for i, units in enumerate(shard_units)
    )
    runtime = RuntimeConfig(lease_ttl_ms=ttl_ms, io_timeout_ms=2_000, batch_size=8)
    return Spec(1, (node,), shards, budget, runtime)


class SchedulerLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def start_engine(self, spec):
        ledger = Ledger.open_session(self.tmp / "ledger.sqlite", spec.digest())
        manager = NodeManager(spec, self.tmp / "runtime")
        manager.start_all()
        engine = ExecutionEngine(spec, ledger, nodes=manager, samples_dir=str(self.tmp / "samples"))
        self.addCleanup(ledger.close)
        self.addCleanup(manager.stop_all, force=True)
        return ledger, manager, engine

    def test_concurrency_capacity_forces_queue_then_runs(self):
        spec = make_spec(concurrency=1, max_leases=1, shard_units=(1, 1))
        ledger, manager, engine = self.start_engine(spec)
        events = [
            Event(1, "dispatch", {"shard_id": "s0", "node_id": "n1", "processing_delay_ms": 120}),
            Event(2, "dispatch", {"shard_id": "s1", "node_id": "n1"}),
        ]
        result = engine.run_events(events)
        self.assertEqual(result["summary"]["status_counts"].get("completed"), 2)
        blocked = [
            decision
            for decision in engine.admission_log
            if decision["shard_id"] == "s1" and not decision["admitted"]
        ]
        self.assertTrue(blocked)
        self.assertIn("node_concurrency_quota", blocked[0]["reasons"])
        self.assertIsNotNone(blocked[0]["budget"]["exact_slack_units"])

    def test_global_budget_and_lease_cap_reasons(self):
        spec = make_spec(budget=2, concurrency=4, max_leases=4, shard_units=(2, 2))
        _ledger, _manager, engine = self.start_engine(spec)
        engine.run_events([
            Event(1, "dispatch", {"shard_id": "s0", "node_id": "n1", "processing_delay_ms": 50}),
            Event(2, "dispatch", {"shard_id": "s1", "node_id": "n1"}),
        ])
        reasons = {tuple(d["reasons"]) for d in engine.admission_log if d["shard_id"] == "s1" and not d["admitted"]}
        self.assertTrue(any("global_budget" in item for item in reasons))

        spec_lease = make_spec(budget=100, concurrency=4, max_leases=1, shard_units=(1, 1))
        _ledger2, _manager2, engine2 = self.start_engine(spec_lease)
        engine2.run_events([
            Event(1, "dispatch", {"shard_id": "s0", "node_id": "n1", "processing_delay_ms": 50}),
            Event(2, "dispatch", {"shard_id": "s1", "node_id": "n1"}),
        ])
        self.assertTrue(any(
            "node_lease_cap" in d["reasons"]
            for d in engine2.admission_log
            if d["shard_id"] == "s1" and not d["admitted"]
        ))

    def test_two_workers_actually_overlap(self):
        spec = make_spec(concurrency=2, max_leases=2, shard_units=(1, 1))
        ledger, _manager, engine = self.start_engine(spec)
        engine.run_events([
            Event(1, "dispatch", {"shard_id": "s0", "node_id": "n1", "processing_delay_ms": 250}),
            Event(2, "dispatch", {"shard_id": "s1", "node_id": "n1", "processing_delay_ms": 250}),
        ])
        rows = ledger.conn.execute(
            "SELECT shard_id, started_ms, finished_ms FROM shard_outcomes WHERE status='completed' ORDER BY shard_id"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        latest_start = max(row["started_ms"] for row in rows)
        earliest_finish = min(row["finished_ms"] for row in rows)
        self.assertGreaterEqual(earliest_finish - latest_start, 100)
        self.assertEqual(len(set(engine.worker_threads.values())), 2)

    def test_real_monotonic_expiry(self):
        spec = make_spec(ttl_ms=40)
        registry = LeaseRegistry(spec)
        lease = registry.admit("s0", "n1", 1)
        self.assertFalse(lease.is_expired_at())
        time.sleep(0.07)
        expired = registry.reap_expired()
        self.assertEqual([item.lease_id for item in expired], [lease.lease_id])
        self.assertEqual(registry.used_resource_units(), 0)
        self.assertEqual(registry.reap_expired(), [])
        with self.assertRaises(LeaseError):
            registry.renew("s0", lease.version)

    def test_node_optimistic_renewal_generation(self):
        spec = make_spec()
        _ledger, manager, engine = self.start_engine(spec)
        manager.call("n1", {
            "type": "lease_grant", "lease_id": "L-opt", "shard_id": "s0",
            "lease_version": 1, "expires_monotonic_ms": 123,
        })
        with self.assertRaises(LeaseError) as ctx:
            manager.call("n1", {
                "type": "lease_renew", "lease_id": "L-opt",
                "expected_version": 999, "proposed_version": 1000,
                "expires_monotonic_ms": 456,
            })
        self.assertEqual(ctx.exception.code, "E_LEASE")
        response = manager.call("n1", {
            "type": "lease_renew", "lease_id": "L-opt",
            "expected_version": 1, "proposed_version": 2,
            "expires_monotonic_ms": 789,
        })
        self.assertEqual(response["lease_version"], 2)

        lease = engine.registry.admit("s1", "n1", 1)
        manager.call("n1", {
            "type": "lease_grant", "lease_id": lease.lease_id, "shard_id": "s1",
            "lease_version": 1, "expires_monotonic_ms": lease.expires_monotonic_ms,
        })
        renewed = engine.renew_lease("s1", 1)
        self.assertEqual(renewed.version, 2)
        with self.assertRaises(LeaseError):
            engine.renew_lease("s1", 1)


if __name__ == "__main__":
    unittest.main()
