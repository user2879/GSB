import tempfile
import time
import unittest
from pathlib import Path

from shardcast.engine import Engine
from shardcast.ledger import Ledger
from shardcast.spec import parse_spec


def make_raw() -> dict:
    return {
        "format_version": 1,
        "lease_ttl_ms": 4000,
        "global_budget": {1: 5, 2: 4},
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": 19501,
                "seed": 11,
                "initial_capacity_version": 1,
                "capacities": [
                    {"version": 1, "concurrency": 1, "lease_limit": 2},
                    {"version": 2, "concurrency": 1, "lease_limit": 1},
                ],
            },
            {
                "node_id": "n2",
                "host": "127.0.0.1",
                "port": 19502,
                "seed": 22,
                "initial_capacity_version": 1,
                "capacities": [
                    {"version": 1, "concurrency": 1, "lease_limit": 2},
                    {"version": 2, "concurrency": 2, "lease_limit": 2},
                ],
            },
        ],
        "shards": [
            {"shard_id": "a", "depends_on": [], "sample_rows": 24,
             "resource_units": 2, "priority": 0},
            {"shard_id": "b", "depends_on": ["a"], "sample_rows": 20,
             "resource_units": 3, "priority": 1},
            {"shard_id": "c", "depends_on": ["a"], "sample_rows": 16,
             "resource_units": 2, "priority": 2},
            {"shard_id": "d", "depends_on": ["b", "c"], "sample_rows": 12,
             "resource_units": 1, "priority": 3},
        ],
        "events": [
            {"kind": "arrive", "shard_id": "a"},
            {"kind": "arrive", "shard_id": "b"},
            {"kind": "arrive", "shard_id": "c"},
            {"kind": "arrive", "shard_id": "d"},
        ],
    }


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger_path = str(Path(self.tmp.name) / "ledger.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, raw: dict, batch_size: int = 64) -> object:
        spec = parse_spec(raw)
        ledger = Ledger(self.ledger_path)
        engine = Engine(spec, ledger, batch_size=batch_size)
        result = engine.run()
        ledger.close()
        return spec, result

    def test_dependency_ready_dispatch_all_completed(self) -> None:
        spec, result = self._run(make_raw())
        for shard_id in spec.shard_ids:
            self.assertEqual(result.outcomes[shard_id]["status"], "completed")
        ledger = Ledger(self.ledger_path)
        rows = ledger.all_outcomes()
        self.assertEqual(set(rows), set(spec.shard_ids))
        for shard_id in spec.shard_ids:
            self.assertIsNotNone(rows[shard_id]["summary"])
        ledger.close()

    def test_determinism_across_batch_and_engine_runs(self) -> None:
        first_spec, first = self._run(make_raw(), batch_size=7)
        ledger_two = str(Path(self.tmp.name) / "ledger2.db")
        spec = parse_spec(make_raw())
        engine_two = Engine(spec, Ledger(ledger_two), batch_size=99)
        second = engine_two.run()
        engine_two.ledger.close()
        for shard_id in first_spec.shard_ids:
            self.assertEqual(
                first.outcomes[shard_id]["summary_digest"],
                second.outcomes[shard_id]["summary_digest"],
            )

    def test_budget_constraint_forces_wait(self) -> None:
        raw = make_raw()
        raw["global_budget"] = {1: 3, 2: 3}
        raw["nodes"][0]["capacities"][0]["concurrency"] = 4
        raw["nodes"][1]["capacities"][0]["concurrency"] = 4
        _, result = self._run(raw)
        self.assertTrue(any(item["wait_end_ms"] is not None for item in result.waits.values()))
        for shard_id, outcome in result.outcomes.items():
            self.assertEqual(outcome["status"], "completed", shard_id)

    def test_capacity_replace_revokes_inflight_and_marks_stale(self) -> None:
        raw = make_raw()
        raw["events"] = [
            {"kind": "arrive", "shard_id": "a"},
            {"kind": "_sleep", "ms": 40},
            {"kind": "capacity", "capacity_version": 2},
            {"kind": "arrive", "shard_id": "b"},
            {"kind": "arrive", "shard_id": "c"},
            {"kind": "arrive", "shard_id": "d"},
        ]
        spec = parse_spec(raw)
        ledger = Ledger(self.ledger_path)
        engine = Engine(spec, ledger)
        engine.start()
        try:
            lease = engine.lease_manager.grant("lease-inflight", "a", "n1", 1)
            engine.pool.install_lease("n1", lease.to_dict())
            engine.in_flight["a"] = lease
            engine._resource_used = spec.shard("a").resource_units
            engine.runtimes["a"].status = "dispatched"
            engine.runtimes["a"].node_id = "n1"
            engine.runtimes["a"].lease_id = lease.lease_id
            engine.runtimes["a"].capacity_version = 1
            engine._replace_capacity(2)
            self.assertNotIn("a", engine.in_flight)
            self.assertEqual(engine.runtimes["a"].status, "waiting")
            actions = {decision["action"] for decision in engine.lease_manager.decisions_snapshot()}
            self.assertIn("revoke", actions)
            self.assertEqual(engine._resource_used, 0)
            self.assertEqual(engine.capacity.current.version, 2)
            self.assertTrue(engine.capacity.backups)
            self.assertTrue(engine.capacity.diffs[0]["node_changes"])
            engine.arrived = set(spec.shard_ids)
            engine._drain_until_settled()
            result = engine._build_result()
            for outcome in result.outcomes.values():
                self.assertEqual(outcome["status"], "completed")
        finally:
            engine.stop()
            ledger.close()

    def test_frozen_shard_not_dispatched_or_reserved(self) -> None:
        raw = make_raw()
        raw["events"] = [
            {"kind": "arrive", "shard_id": "a"},
            {"kind": "freeze", "shard_id": "b"},
            {"kind": "arrive", "shard_id": "b"},
            {"kind": "arrive", "shard_id": "c"},
            {"kind": "unfreeze", "shard_id": "b"},
            {"kind": "arrive", "shard_id": "d"},
        ]
        spec, result = self._run(raw)
        self.assertEqual(result.outcomes["b"]["status"], "completed")
        self.assertTrue(all(outcome["status"] == "completed" for outcome in result.outcomes.values()))

    def test_cancelled_shard_terminal_and_not_retried(self) -> None:
        raw = make_raw()
        raw["shards"][1]["resource_units"] = 6
        raw["global_budget"] = {1: 3, 2: 3}
        raw["events"] = [
            {"kind": "arrive", "shard_id": "a"},
            {"kind": "cancel", "shard_id": "b"},
            {"kind": "arrive", "shard_id": "b"},
            {"kind": "arrive", "shard_id": "c"},
            {"kind": "arrive", "shard_id": "d"},
        ]
        spec, result = self._run(raw)
        self.assertEqual(result.outcomes["b"]["status"], "cancelled")
        grants = [decision for decision in result.decisions if decision["action"] == "grant"]
        self.assertNotIn("b", [grant["shard_id"] for grant in grants])

    def test_lease_uniqueness_under_capacity_switch(self) -> None:
        raw = make_raw()
        raw["events"] = [
            {"kind": "arrive", "shard_id": "a"},
            {"kind": "capacity", "capacity_version": 2},
            {"kind": "arrive", "shard_id": "b"},
            {"kind": "arrive", "shard_id": "c"},
            {"kind": "arrive", "shard_id": "d"},
        ]
        _, result = self._run(raw)
        active_owners: dict[str, str] = {}
        for decision in result.decisions:
            if decision["action"] == "grant":
                shard_id = decision["shard_id"]
                self.assertNotIn(shard_id, active_owners)
                active_owners[shard_id] = decision["lease_id"]
        adjudications = [decision["adjudication"] for decision in result.decisions]
        self.assertEqual(adjudications, sorted(adjudications))

    def test_triple_constraints_never_violated_at_commit_time(self) -> None:
        raw = make_raw()
        raw["global_budget"] = {1: 4, 2: 4}
        for node in raw["nodes"]:
            node["capacities"][0]["concurrency"] = 1
            node["capacities"][0]["lease_limit"] = 2
        spec = parse_spec(raw)
        ledger = Ledger(str(Path(self.tmp.name) / "constraints.db"))
        engine = Engine(spec, ledger)
        engine.start()
        try:
            engine.arrived = set(spec.shard_ids)
            max_resource = 0
            max_per_node: dict[str, int] = {}
            for _ in range(100):
                engine.lease_manager.expire_due()
                launched = engine._schedule_pass()
                max_resource = max(max_resource, engine._resource_used)
                for node_id in spec.node_ids:
                    active = engine.lease_manager.active_for_node(node_id)
                    max_per_node[node_id] = max(
                        max_per_node.get(node_id, 0), len(active)
                    )
                    self.assertLessEqual(
                        len(active),
                        engine.capacity.current.quota(node_id).lease_limit,
                    )
                    dispatched = [
                        lease for lease in active if lease.shard_id in engine.in_flight
                    ]
                    self.assertLessEqual(
                        len(dispatched),
                        engine.capacity.current.quota(node_id).concurrency,
                    )
                self.assertLessEqual(max_resource, spec.global_budget[1])
                engine._run_dispatched(launched)
                if all(
                    engine.runtimes[shard_id].status in {"completed", "failed", "cancelled"}
                    for shard_id in spec.shard_ids
                ):
                    break
            for shard_id in spec.shard_ids:
                self.assertEqual(engine.runtimes[shard_id].status, "completed", shard_id)
        finally:
            engine.stop()
            ledger.close()


if __name__ == "__main__":
    unittest.main()
