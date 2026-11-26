import tempfile
import unittest
from pathlib import Path

from shardcast.engine import Engine
from shardcast.ledger import Ledger
from shardcast.recovery import recover_and_replay
from shardcast.spec import parse_spec


def recovery_raw() -> dict:
    return {
        "format_version": 1,
        "lease_ttl_ms": 4000,
        "global_budget": {1: 8},
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": 19701,
                "seed": 31,
                "initial_capacity_version": 1,
                "capacities": [{"version": 1, "concurrency": 2, "lease_limit": 3}],
            },
            {
                "node_id": "n2",
                "host": "127.0.0.1",
                "port": 19702,
                "seed": 37,
                "initial_capacity_version": 1,
                "capacities": [{"version": 1, "concurrency": 2, "lease_limit": 3}],
            },
        ],
        "shards": [
            {"shard_id": "a", "depends_on": [], "sample_rows": 30,
             "resource_units": 2, "priority": 0},
            {"shard_id": "b", "depends_on": ["a"], "sample_rows": 26,
             "resource_units": 3, "priority": 1},
            {"shard_id": "c", "depends_on": ["a"], "sample_rows": 18,
             "resource_units": 1, "priority": 2},
            {"shard_id": "d", "depends_on": ["b", "c"], "sample_rows": 14,
             "resource_units": 2, "priority": 3},
        ],
        "events": [
            {"kind": "arrive", "shard_id": "a"},
            {"kind": "arrive", "shard_id": "b"},
            {"kind": "arrive", "shard_id": "c"},
            {"kind": "arrive", "shard_id": "d"},
        ],
    }


class LedgerRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger_path = str(Path(self.tmp.name) / "ledger.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_snapshot_replaced_when_spec_digest_changes(self) -> None:
        raw = recovery_raw()
        spec = parse_spec(raw)
        ledger = Ledger(self.ledger_path)
        created, replacement = ledger.initialize_snapshot(
            spec_digest=spec.structural_digest(),
            node_set_version=spec.node_set_version(),
            capacity_version=1,
        )
        self.assertTrue(created)
        self.assertIsNone(replacement)
        ledger.close()

        changed = recovery_raw()
        changed["shards"][0]["sample_rows"] = 44
        changed_spec = parse_spec(changed)
        ledger_two = Ledger(self.ledger_path)
        created, replacement = ledger_two.initialize_snapshot(
            spec_digest=changed_spec.structural_digest(),
            node_set_version=changed_spec.node_set_version(),
            capacity_version=1,
        )
        self.assertTrue(created)
        self.assertIsNotNone(replacement)
        self.assertIn("mismatch", replacement)
        self.assertEqual(ledger_two.all_outcomes(), {})
        ledger_two.close()

    def test_atomic_commit_writes_all_fields_in_one_transaction(self) -> None:
        spec = parse_spec(recovery_raw())
        ledger = Ledger(self.ledger_path)
        ledger.initialize_snapshot(
            spec_digest=spec.structural_digest(),
            node_set_version=spec.node_set_version(),
            capacity_version=1,
        )
        outcome = {
            "shard_id": "a",
            "node_id": "n1",
            "status": "completed",
            "capacity_version": 1,
            "lease_id": "lease-000001",
            "lease_version": 1,
            "summary": {"node_id": "n1", "shard_id": "a", "groups": {}},
            "resource_units": 2,
            "committed_ms": 123,
        }
        ledger.commit_outcome(
            outcome=outcome,
            water_marks={"n1": {
                "node_id": "n1", "capacity_version": 1,
                "high_water_ms": 123, "shard_count": 1, "resource_total": 2,
            }},
            last_seq=0,
            capacity_version=1,
        )
        row = ledger.get_outcome("a")
        self.assertEqual(row["status"], "completed")
        self.assertEqual(ledger.cursor()["last_seq"], 0)
        self.assertTrue(row["summary_digest"])
        ledger.close()

    def test_recovery_after_worker_kill_replays_unknown_shards(self) -> None:
        raw = recovery_raw()
        spec = parse_spec(raw)
        ledger = Ledger(self.ledger_path)
        engine = Engine(spec, ledger)
        engine.start()
        try:
            engine.arrived = {"a", "b"}
            engine._drain_once()
            for _ in range(20):
                if all(
                    engine.runtimes[shard_id].status == "completed"
                    for shard_id in ("a", "b")
                ):
                    break
                engine._drain_once()
            committed = ledger.all_outcomes()
            self.assertIn("a", committed)
            killed_node = engine.runtimes["a"].node_id
            engine.pool.kill_node_hard(killed_node)
        finally:
            engine.stop()
            ledger.close()

        report = recover_and_replay(
            parse_spec(recovery_raw()),
            self.ledger_path,
            killed_node=killed_node,
        )
        self.assertIn("a", report.committed)
        self.assertIn("c", report.unknown)
        self.assertIn("d", report.unknown)
        self.assertEqual(report.mismatches, [])
        self.assertTrue(report.reclaimed_units > 0)

        verify_ledger = Ledger(self.ledger_path)
        rows = verify_ledger.all_outcomes()
        self.assertEqual(set(rows), set(spec.shard_ids))
        for shard_id in spec.shard_ids:
            self.assertEqual(rows[shard_id]["status"], "completed")
            self.assertIsNotNone(rows[shard_id]["summary"])
        verify_ledger.close()

    def test_replay_summary_byte_identical_to_first_run(self) -> None:
        raw = recovery_raw()
        spec = parse_spec(raw)
        first_ledger = Ledger(str(Path(self.tmp.name) / "first.db"))
        first_engine = Engine(spec, first_ledger)
        first = first_engine.run()
        first_ledger.close()

        second_ledger = Ledger(str(Path(self.tmp.name) / "second.db"))
        second_engine = Engine(parse_spec(recovery_raw()), second_ledger)
        second = second_engine.run()
        second_ledger.close()
        for shard_id in spec.shard_ids:
            self.assertEqual(
                first.outcomes[shard_id]["summary_digest"],
                second.outcomes[shard_id]["summary_digest"],
            )


if __name__ == "__main__":
    unittest.main()
