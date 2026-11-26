from __future__ import annotations

import threading
import time
import unittest

from shard_diag.errors import LeaseError
from shard_diag.leases import LeaseManager
from shard_diag.node_protocol import NodeCluster
from shard_diag.scheduler import DispatchScheduler
from shard_diag.spec import parse_spec

from tests.factories import free_port, sample_spec_dict


class LeaseSubprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = parse_spec(sample_spec_dict())
        cls.cluster = NodeCluster(cls.spec.nodes).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cluster.stop()

    def test_grant_renew_optimistic_version_and_node_mismatch(self) -> None:
        manager = LeaseManager(
            {node.node_id: node for node in self.spec.nodes},
            self.cluster.clients,
            default_ttl_ms=5_000,
        )
        lease = manager.acquire("s1-extract-a", "node-a", 3)
        renewed = manager.renew(lease.lease_id, 1, ttl_ms=5_000)
        self.assertEqual(renewed.lease_version, 2)
        self.assertEqual(renewed.renewal_count, 1)
        expires_after_success = renewed.expires_at_ms

        with self.assertRaises(LeaseError) as raised:
            manager.renew(lease.lease_id, 1, ttl_ms=9_999)
        self.assertEqual(raised.exception.code, "E_LEASE")
        self.assertEqual(raised.exception.field, "lease_version")
        self.assertEqual(manager.get(lease.lease_id).expires_at_ms, expires_after_success)

        with self.assertRaises(LeaseError):
            self.cluster.clients["node-a"].renew_lease(lease.lease_id, 1)

    def test_expiration_uses_real_monotonic_sleep(self) -> None:
        manager = LeaseManager(
            {node.node_id: node for node in self.spec.nodes},
            self.cluster.clients,
            default_ttl_ms=60,
        )
        lease = manager.acquire("expiring", "node-a", 3, ttl_ms=60)
        self.assertEqual(manager.active_count("node-a"), 1)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and manager.active_count("node-a"):
            time.sleep(0.02)
        self.assertEqual(manager.active_count("node-a"), 0)
        self.assertEqual(manager.get(lease.lease_id).status, "EXPIRED")
        with self.assertRaises(LeaseError):
            manager.renew(lease.lease_id, lease.lease_version)

    def test_capacity_revocation_clears_active_lease(self) -> None:
        manager = LeaseManager(
            {node.node_id: node for node in self.spec.nodes},
            self.cluster.clients,
            default_ttl_ms=5_000,
        )
        first = manager.acquire("cap-1", "node-b", 7)
        second = manager.acquire("cap-2", "node-b", 7)
        revoked = manager.revoke_for_capacity("node-b", 8)
        self.assertEqual({lease.lease_id for lease in revoked}, {first.lease_id, second.lease_id})
        self.assertEqual(manager.active_count("node-b"), 0)
        ping = self.cluster.clients["node-b"].ping()
        self.assertEqual(ping["capacity_version"], 8)
        self.cluster.clients["node-b"].fence_capacity(7)


def quota_spec(*, budget: int, concurrency: int, lease_limit: int, shards: int = 3):
    raw = {
        "format_version": 1,
        "session_id": "quota-test",
        "scale_power": 0,
        "global_resource_budget": budget,
        "lease_ttl_ms": 5_000,
        "max_revision_lag": 5,
        "dispatch_delay_limit_ms": 10_000,
        "revision_chain": [1],
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": free_port(),
                "concurrency_quota": concurrency,
                "lease_limit": lease_limit,
                "seed": 7,
                "capacity_version": 1,
            }
        ],
        "shards": [
            {
                "shard_id": f"s{index}",
                "node_id": "n1",
                "dependencies": [],
                "sample_rows": 16,
                "resource_units": 1,
                "priority": 100 - index,
            }
            for index in range(shards)
        ],
        "delay_batches": [
            {
                "batch_id": "b1",
                "node_id": "n1",
                "primary_revision": 1,
                "replayed_revision": 1,
                "delay_samples_ms": [0],
            }
        ],
    }
    return parse_spec(raw)


class SchedulerQuotaTests(unittest.TestCase):
    def _run_blocking(self, spec: DispatchScheduler, expected_key: str, expected: int):
        entered = threading.Event()
        release = threading.Event()
        worker_names: set[str] = set()
        active = 0
        peak = 0
        lock = threading.Lock()

        def runner(state, lease):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                worker_names.add(threading.current_thread().name)
                if active >= expected:
                    entered.set()
            release.wait(timeout=3.0)
            with lock:
                active -= 1
            return {"summary": {"shard_id": state.spec.shard_id}}

        scheduler = DispatchScheduler(spec, work_runner=runner)
        thread = threading.Thread(target=scheduler.run, daemon=True)
        thread.start()
        self.assertTrue(entered.wait(timeout=2.0))
        time.sleep(0.15)
        snapshot = scheduler.occupancy_snapshot()
        if expected_key == "global":
            self.assertLessEqual(snapshot["global_resources"], expected)
            self.assertEqual(snapshot["peak_global_resources"], expected)
        else:
            self.assertLessEqual(snapshot["node_shards"]["n1"], expected)
            self.assertLessEqual(snapshot["node_leases"]["n1"], expected)
            self.assertEqual(snapshot["peak_node_shards"]["n1"], expected)
        self.assertGreaterEqual(len(worker_names), min(expected, 2))
        release.set()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        summary = scheduler.summary()
        self.assertEqual(summary["counts"].get("SUCCEEDED"), len(spec.shards))
        for state in scheduler.states.values():
            self.assertIsNotNone(state.wait_started_ms)
            self.assertIsNotNone(state.wait_ended_ms)
            self.assertIsNotNone(state.wait_ms)
        return scheduler

    def test_global_budget_queues_excess(self) -> None:
        spec = quota_spec(budget=2, concurrency=3, lease_limit=3)
        scheduler = self._run_blocking(spec, "global", 2)
        self.assertEqual(scheduler.peak_global_resources, 2)
        self.assertTrue(scheduler.water_comparisons)
        self.assertTrue(
            any(comparison.get("budget_boundary") for comparison in scheduler.water_comparisons)
        )
        self.assertEqual(scheduler.water_comparisons[0]["winner"], "s0")

    def test_concurrency_quota_limits_active_shards(self) -> None:
        spec = quota_spec(budget=10, concurrency=1, lease_limit=3)
        scheduler = self._run_blocking(spec, "node", 1)
        self.assertEqual(scheduler.peak_node_shards["n1"], 1)

    def test_lease_limit_limits_active_leases(self) -> None:
        spec = quota_spec(budget=10, concurrency=3, lease_limit=1)
        scheduler = self._run_blocking(spec, "node", 1)
        self.assertEqual(scheduler.peak_node_leases["n1"], 1)

    def test_unready_dependency_is_not_queued(self) -> None:
        raw = quota_spec(budget=10, concurrency=2, lease_limit=2, shards=2).to_dict()
        raw["shards"][1]["dependencies"] = ["s0"]
        spec = parse_spec(raw)
        scheduler = DispatchScheduler(spec)
        scheduler._promote_dependencies(set(scheduler.states))
        self.assertEqual(scheduler.states["s0"].status, "QUEUED")
        self.assertEqual(scheduler.states["s1"].status, "PENDING")
        self.assertIsNone(scheduler.states["s1"].wait_started_ms)


if __name__ == "__main__":
    unittest.main()
