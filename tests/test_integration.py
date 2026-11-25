import json
import os
import shutil
import tempfile
import unittest

from shardflow.engine import Engine
from shardflow.errors import DuplicateError
from shardflow.replay import verify_ledger
from shardflow.spec import load_spec
from shardflow.viewmodel import build_view_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, "demo", "demo.spec.json")
EVENTS_PATH = os.path.join(ROOT, "demo", "demo.events.ndjson")


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger_path = os.path.join(self.tmp.name, "e2e.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_run_replay_and_deterministic_rerun(self):
        spec = load_spec(SPEC_PATH, EVENTS_PATH)
        engine = Engine(spec, self.ledger_path)
        snapshot = engine.run()
        self.assertEqual(snapshot["errors"], [])
        statuses = {sid: row["status"] for sid, row in
                    snapshot["shards"].items()}
        self.assertEqual(statuses["s7"], "cancelled")
        self.assertEqual(statuses["s8"], "blocked")
        completed = [sid for sid, status in statuses.items()
                     if status == "completed"]
        self.assertEqual(sorted(completed),
                         ["s1", "s2", "s3", "s4", "s5", "s6"])

        result = verify_ledger(SPEC_PATH, self.ledger_path, EVENTS_PATH)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["resource_in_use"], 0)

        model = build_view_model(SPEC_PATH, self.ledger_path, EVENTS_PATH)
        self.assertEqual(model["format_version"], 1)
        self.assertTrue(model["snapshots"])
        self.assertIn("s7", [row["shard_id"] for row in model["shards"]
                             if row["status"] == "cancelled"])

    def test_duplicate_cancel_raises_e_dup(self):
        spec = load_spec(SPEC_PATH, EVENTS_PATH)
        engine = Engine(spec, os.path.join(self.tmp.name, "dup.sqlite"),
                        start_nodes=False, enable_renewer=False)
        engine.runtimes["s7"].cancelled = True
        from shardflow.spec import Event
        event = Event({"type": "cancel", "shard_id": "s7"}, 0)
        with self.assertRaises(DuplicateError) as ctx:
            engine._apply_cancel(event)
        self.assertEqual(ctx.exception.code, "E_DUP")

    def test_budget_never_exceeded_in_ledger(self):
        spec = load_spec(SPEC_PATH, EVENTS_PATH)
        engine = Engine(spec, os.path.join(self.tmp.name, "budget.sqlite"))
        engine.run()
        replay = verify_ledger(SPEC_PATH,
                               os.path.join(self.tmp.name, "budget.sqlite"),
                               EVENTS_PATH)
        self.assertEqual(replay["status"], "ok")
        self.assertIn("resource_budget", replay["checks"])


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_unknown_shard_is_recomputed_after_restart(self):
        ledger_path = os.path.join(self.tmp.name, "recover.sqlite")
        spec = load_spec(SPEC_PATH, EVENTS_PATH)
        engine = Engine(spec, ledger_path)
        engine.ledger.mark_unknown(
            "s2", 1, "node-a",
            {"granted_ms": 1, "config_version": 1, "capacity_version": 1,
             "resource_units": 2, "wait_start_ms": 1, "wait_end_ms": 1},
            0,
        )
        engine.ledger.close()

        recovered_spec = load_spec(SPEC_PATH, EVENTS_PATH)
        recovered = Engine(recovered_spec, ledger_path)
        info = recovered.recover()
        self.assertIn("s2", info["unknown_shards"])
        self.assertTrue(recovered.runtimes["s2"].unknown)
        snapshot = recovered.rerun_unknown()
        self.assertEqual(snapshot["shards"]["s2"]["status"], "completed")
        self.assertEqual(snapshot["unknown_shards"], [])
        outcomes = recovered.ledger.fetch_outcomes()
        s2_rows = [row for row in outcomes if row["shard_id"] == "s2"]
        self.assertTrue(any(row["status"] == "completed" for row in s2_rows))
        self.assertTrue(any(row["status"] == "unknown" for row in s2_rows))
        recovered.ledger.close()

    def test_node_force_kill_and_restart_completes_handshake(self):
        spec = load_spec(SPEC_PATH, EVENTS_PATH)
        engine = Engine(spec, os.path.join(self.tmp.name, "kill.sqlite"))
        engine.start_node_processes()
        try:
            self.assertTrue(engine.node_processes["node-a"].is_alive())
            engine.force_kill_node("node-a")
            self.assertFalse(engine.node_processes["node-a"].is_alive())
            engine.restart_node("node-a")
            self.assertTrue(engine.node_processes["node-a"].is_alive())
            ack = engine.node_clients["node-a"].handshake()
            self.assertEqual(ack["node_id"], "node-a")
        finally:
            engine.stop_node_processes()


if __name__ == "__main__":
    unittest.main()
