import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from shardmesh.engine import Engine
from shardmesh.errors import ShardMeshError
from shardmesh.generator import demo_spec_dict, generate_samples
from shardmesh.ledger import Ledger
from shardmesh.leases import LeaseManager
from shardmesh.capacity import CapacityState
from shardmesh.samples import aggregate_batch
from shardmesh.spec import parse_spec
from shardmesh.timeutil import now_ns


class FakeNodeProcess:
    def __init__(self, node_id, scale, *, barrier=None, delay=0.02, fail_code=None):
        self.node_id = node_id
        self.scale = scale
        self.barrier = barrier
        self.delay = delay
        self.fail_code = fail_code
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def lease_open(self, lease, ttl_ms):
        return {"lease_id": lease["lease_id"], "version": 1}

    def lease_renew(self, *args, **kwargs):
        return {"version": 2, "renew_count": 1}

    def lease_release(self, lease_id):
        return {"ok": True}

    def aggregate(self, message):
        if self.fail_code:
            raise ShardMeshError(self.fail_code, "induced protocol failure", category="test", field_name="response")
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2.0)
            time.sleep(self.delay)
            summary = aggregate_batch(message["batch"], self.scale)
            return {
                "node_id": self.node_id,
                "shard_id": message["shard_id"],
                "capacity_version": message["capacity_version"],
                "summary": summary,
                "elapsed_ms": int(self.delay * 1000),
            }
        finally:
            with self.lock:
                self.active -= 1


class FakeCluster:
    def __init__(self, spec, **kwargs):
        self.node_specs = dict(spec.node_map)
        self.scale = spec.scale
        self.processes = {node.id: FakeNodeProcess(node.id, spec.scale, **kwargs) for node in spec.nodes}

    def get(self, node_id):
        return self.processes[node_id]

    def restart_node(self, node_id, capacity_version=None):
        return self.processes[node_id]

    def stop_all(self):
        return None


class BlockingNode:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def lease_open(self, lease, ttl_ms):
        return {"lease_id": lease["lease_id"], "version": 1}

    def lease_release(self, lease_id):
        return {"ok": True}

    def aggregate(self, message):
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("blocking test node timed out")
        return {
            "node_id": "n1",
            "shard_id": message["shard_id"],
            "capacity_version": message["capacity_version"],
            "summary": aggregate_batch(message["batch"], 10),
            "elapsed_ms": 1,
        }


class SingleNodeCluster:
    def __init__(self, spec):
        self.node_specs = dict(spec.node_map)
        self.scale = spec.scale
        self.node = BlockingNode()

    def get(self, node_id):
        return self.node

    def restart_node(self, node_id, capacity_version=None):
        return self.node

    def stop_all(self):
        return None


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = demo_spec_dict(ports=[46111, 46112, 46113])
        self.spec = parse_spec(self.raw)
        self.samples = generate_samples(self.raw)
        self.ledger = Ledger(self.root / "test.sqlite3")

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def engine(self, **fake_kwargs):
        cluster = FakeCluster(self.spec, **fake_kwargs)
        return Engine(
            self.spec,
            self.samples,
            self.ledger,
            work_dir=self.root / "runtime",
            cluster=cluster,
            manage_cluster=False,
        )

    def test_triple_limits_and_dependency_completion(self):
        engine = self.engine(delay=0.01)
        with engine:
            records = engine.run_events(self.spec.events)
            snapshot = engine.capacity.snapshot()
        self.assertEqual(len(records), len(self.spec.shards))
        self.assertLessEqual(snapshot["peak_units"], self.spec.global_budget)
        for node in self.spec.nodes:
            self.assertLessEqual(snapshot["peak_shards_by_node"][node.id], node.concurrency)
            self.assertLessEqual(snapshot["peak_leases_by_node"][node.id], node.lease_limit)
            self.assertEqual(snapshot["active_shards_by_node"][node.id], 0)
        committed = {record["shard_id"]: record for record in records}
        for shard in self.spec.shards:
            for upstream in shard.deps:
                self.assertLess(
                    committed[upstream]["cursor_seq"],
                    committed[shard.id]["cursor_seq"],
                )

    def test_work_is_projected_to_parallel_thread_pool(self):
        barrier = threading.Barrier(2)
        engine = self.engine(barrier=barrier, delay=0.02)
        with engine:
            engine.arrived.update({"s01", "s05"})
            engine.mark_wait_start("s01")
            engine.mark_wait_start("s05")
            engine.settle()
            self.assertEqual(engine.failed, {})

    def test_protocol_failure_leaves_no_capacity_change(self):
        engine = self.engine(fail_code="E_FIELD")
        with engine:
            engine.arrived.add("s01")
            engine.mark_wait_start("s01")
            engine.settle()
            self.assertIn("s01", engine.failed)
            snapshot = engine.capacity.snapshot()
        self.assertEqual(snapshot["active_units"], 0)
        self.assertEqual(snapshot["in_flight"], [])
        for active in snapshot["active_shards_by_node"].values():
            self.assertEqual(active, 0)

    def test_zero_budget_slack_is_marked_error_affected(self):
        engine = self.engine()
        engine.arrived.add("s02")
        engine.records["s01"] = {"outcome": {"summary": {"checksum": "ready"}}}
        engine.capacity.active_units = 4
        decision = engine._choose_winner()
        self.assertIsNotNone(decision)
        self.assertEqual(decision["shard"].id, "s02")
        self.assertEqual(decision["budget_boundary"]["slack_units"], 0)
        self.assertTrue(decision["budget_boundary"]["ambiguous"])
        self.assertTrue(decision["error_affected"])

    def test_indistinguishable_water_uses_stable_topology_order(self):
        raw = {
            "format_version": 1,
            "seed": 1,
            "global_budget": 10,
            "scale": 1000,
            "nodes": [
                {
                    "id": "n1",
                    "host": "127.0.0.1",
                    "port": 47111,
                    "concurrency": 2,
                    "lease_limit": 2,
                    "seed": 1,
                    "capacity_version": 1,
                }
            ],
            "shards": [
                {"id": "b-shard", "node_id": "n1", "deps": [], "rows": 8, "units": 1, "priority": 0},
                {"id": "a-shard", "node_id": "n1", "deps": [], "rows": 8, "units": 1, "priority": 0},
            ],
            "events": [
                {"seq": 1, "type": "arrive", "shard_id": "a-shard"},
                {"seq": 2, "type": "arrive", "shard_id": "b-shard"},
            ],
        }
        spec = parse_spec(raw)
        samples = generate_samples(raw)
        cluster = FakeCluster(spec)
        engine = Engine(spec, samples, self.ledger, work_dir=self.root / "tie", cluster=cluster, manage_cluster=False)
        engine.arrived.update({"a-shard", "b-shard"})
        decision = engine._choose_winner()
        self.assertEqual(decision["shard"].id, "a-shard")
        self.assertTrue(decision["adjudication"]["distinct_water_boundary"])
        self.assertTrue(decision["error_affected"])

    def test_capacity_bump_revokes_active_unexpired_lease_and_blocks_commit(self):
        raw = {
            "global_budget": 5,
            "scale": 10,
            "lease_ttl_ms": 10_000,
            "nodes": [
                {"id": "n1", "host": "127.0.0.1", "port": 47211, "concurrency": 1, "lease_limit": 1, "seed": 1, "capacity_version": 1}
            ],
            "shards": [
                {"id": "s1", "node_id": "n1", "deps": [], "rows": 8, "units": 1, "priority": 0}
            ],
            "events": [{"seq": 1, "type": "arrive", "shard_id": "s1"}],
        }
        spec = parse_spec(raw)
        samples = generate_samples(raw)
        cluster = SingleNodeCluster(spec)
        engine = Engine(
            spec,
            samples,
            self.ledger,
            work_dir=self.root / "bump",
            cluster=cluster,
            manage_cluster=False,
        )
        with engine:
            engine.arrived.add("s1")
            engine.pump()
            self.assertTrue(cluster.node.entered.wait(timeout=2.0))
            self.assertEqual(engine.capacity.active_units, 1)
            event = type("Event", (), {"seq": 2, "type": "bump_capacity_version", "params": {"node_id": "n1", "version": 2}})()
            engine.apply_parameter_event(event)
            self.assertEqual(len(engine.capacity_revocations[-1]["leases"]), 1)
            self.assertEqual(engine.capacity.active_units, 0)
            self.assertEqual(engine.leases.active_count("n1"), 0)
            cluster.node.release.set()
            deadline = time.monotonic() + 2.0
            while "s1" in engine.futures and time.monotonic() < deadline:
                engine._consume_finished()
                time.sleep(0.02)
            engine.settle()
            self.assertNotIn("s1", engine.records)
            self.assertIn("s1", engine.failed)
            self.assertEqual(engine.failed["s1"].code, "E_LEASE")


class CapacityVersionRevocationTests(unittest.TestCase):
    def test_old_version_leases_are_revoked_and_quota_released(self):
        raw = demo_spec_dict()
        spec = parse_spec(raw)
        shard = spec.shard_map["s01"]
        capacity = CapacityState(spec.global_budget, spec.nodes)
        leases = LeaseManager()
        allocation = capacity.admit(shard, "lease-s01-old", now_ns())
        self.assertIsNotNone(allocation)
        leases.issue(
            lease_id="lease-s01-old",
            shard_id="s01",
            node_id="n1",
            ttl_ms=10_000,
            capacity_version=1,
        )
        revoked = leases.revoke_capacity_version("n1", 2)
        self.assertEqual([lease.shard_id for lease in revoked], ["s01"])
        for lease in revoked:
            capacity.release(lease.shard_id, "capacity_version_bump")
        snapshot = capacity.snapshot()
        self.assertEqual(snapshot["active_units"], 0)
        self.assertEqual(snapshot["active_shards_by_node"]["n1"], 0)
        self.assertEqual(snapshot["active_leases_by_node"]["n1"], 0)
        self.assertEqual(leases.active_count("n1"), 0)


if __name__ == "__main__":
    unittest.main()
