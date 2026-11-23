import json
import tempfile
import unittest
from pathlib import Path

from diagconsole.engine import Engine, NodeKey
from diagconsole.errors import DiagError
from diagconsole.inject import (inject_corrupt_spec, inject_dangling_derives,
                                inject_snapshot_digest)
from diagconsole.ledger import Ledger
from diagconsole.model import Spec, load_spec
from diagconsole.snapshot import build_snapshot, load_snapshot, save_snapshot

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class RecoveryReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.spec = load_spec(SAMPLES / "clusters.json",
                              SAMPLES / "tasks.json",
                              SAMPLES / "revisions.json")
        self.ledger = Ledger(self.dir / "ledger.db")
        self.engine = Engine(self.spec, self.ledger)
        self.engine.apply_chain(None, "seed")
        snapshot = build_snapshot(self.spec, 6,
                                  self.ledger.list_nodes("committed"),
                                  {"stage": "test"})
        save_snapshot(self.dir / "snapshot.json", snapshot)

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def test_replay_is_reproducible(self):
        snapshot = load_snapshot(self.dir / "snapshot.json", self.spec)
        result = self.engine.replay(snapshot, use_services=False)
        self.assertTrue(result["reproducible"])

    def test_replay_detects_first_mismatch(self):
        snapshot = load_snapshot(self.dir / "snapshot.json", self.spec)
        snapshot["nodes"][0]["aggregate"]["total"] += 1
        with self.assertRaises(DiagError) as ctx:
            self.engine.replay(snapshot, use_services=False)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.location["field"], "aggregate")

    def test_forced_termination_recovery_and_rerun(self):
        key = NodeKey(self.spec.tasks[0].task_id,
                      self.spec.tasks[0].targets[0], 6)
        in_flight = self.engine.compute_node_locally(key)
        self.ledger.begin_batch("staging-crash")
        self.ledger.insert_unknown(in_flight)
        fresh_ledger = Ledger(self.dir / "ledger.db")
        engine = Engine(self.spec, fresh_ledger)
        report = engine.recover()
        self.assertTrue(report["requires_rerun"])
        self.assertFalse(report["exit_report_available"])
        self.assertEqual(len(report["unknown_nodes"]), 1)
        engine.rerun_unknown()
        self.assertEqual(engine.ledger.list_unknown_nodes(), [])
        fresh_ledger.close()

    def test_corrupt_spec_injection_is_e_spec(self):
        target = inject_corrupt_spec(SAMPLES / "revisions.json")
        with self.assertRaises(DiagError) as ctx:
            load_spec(SAMPLES / "clusters.json", SAMPLES / "tasks.json", target)
        self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_dangling_derives_injection_is_e_ref(self):
        # E_REF models a chain that became dangling inside the platform after
        # load-time validation, so construct the spec bypassing strict loading.
        from diagconsole.model import Revision
        revisions = list(self.spec.revisions)
        revisions[3] = Revision(revisions[3].version, revisions[3].parent,
                                revisions[3].values, revisions[3].strategy,
                                (999,))
        spec = Spec(clusters=self.spec.clusters, tasks=self.spec.tasks,
                    revisions=tuple(revisions))
        ledger = Ledger(self.dir / "ref.db")
        engine = Engine(spec, ledger)
        with self.assertRaises(DiagError) as ctx:
            engine.validate_references()
        self.assertEqual(ctx.exception.code, "E_REF")
        self.assertEqual(ctx.exception.location["reference"], 999)
        ledger.close()

    def test_snapshot_digest_injection_is_e_snapshot(self):
        target = inject_snapshot_digest(self.dir / "snapshot.json")
        with self.assertRaises(DiagError) as ctx:
            load_snapshot(target, self.spec)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")


if __name__ == "__main__":
    unittest.main()
