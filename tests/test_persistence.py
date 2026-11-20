import os
import tempfile
import unittest

from gpu_scheduler.errors import E_LEDGER, E_REPLAY, SchedulerError
from gpu_scheduler.injector import tamper_ledger
from gpu_scheduler.ledger import Ledger
from gpu_scheduler.loader import load_declaration
from gpu_scheduler import replay as replay_mod
from gpu_scheduler.scheduler import SchedulingSession, resume_session
from gpu_scheduler import snapshot as snapshot_mod

SPEC = os.path.join("samples", "specs")


def load_sample():
    return load_declaration(
        os.path.join(SPEC, "session.json"),
        os.path.join(SPEC, "pools.json"),
        os.path.join(SPEC, "tasks.json"))


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ledger_path = os.path.join(self.tmp, "ledger.sqlite")
        self.snapshot_dir = os.path.join(self.tmp, "snapshots")
        self.backup_dir = os.path.join(self.tmp, "backup")

    def test_rounds_persist_and_digest_verifies(self):
        declaration = load_sample()
        session = SchedulingSession(
            declaration, self.ledger_path, self.snapshot_dir, self.backup_dir)
        result = session.run()
        session.close()
        self.assertEqual(result["status"], "completed")
        ledger = Ledger(self.ledger_path)
        digest = ledger.verify()
        self.assertEqual(digest, result["ledger_digest"])
        self.assertEqual(ledger.last_round_no(), result["rounds"])
        ledger.close()

    def test_tampered_ledger_raises_e_ledger(self):
        declaration = load_sample()
        session = SchedulingSession(
            declaration, self.ledger_path, self.snapshot_dir, self.backup_dir)
        session.run()
        session.close()
        tamper_ledger(self.ledger_path)
        ledger = Ledger(self.ledger_path)
        with self.assertRaises(SchedulerError) as ctx:
            ledger.verify()
        self.assertEqual(ctx.exception.code, E_LEDGER)
        ledger.close()

    def test_crash_marks_unknown_and_replay_matches_reference(self):
        declaration = load_sample()
        session = SchedulingSession(
            declaration, self.ledger_path, self.snapshot_dir, self.backup_dir)
        crash = session.run(crash_after_round=3)
        session.close()
        self.assertEqual(crash["status"], "crashed")
        self.assertTrue(crash["unknown_shards"])
        reference = replay_mod.reference_run(load_sample())
        result = resume_session(
            load_sample(), self.ledger_path, self.snapshot_dir,
            self.backup_dir, reference_rounds=reference.rt.rounds)
        self.assertEqual(result["status"], "recovered")
        self.assertTrue(result["unknown_shards"])
        self.assertTrue(result["replay_consistent"])

    def test_forced_replay_divergence_is_reported(self):
        declaration = load_sample()
        session = SchedulingSession(
            declaration, self.ledger_path, self.snapshot_dir, self.backup_dir)
        session.run(crash_after_round=3)
        session.close()
        from gpu_scheduler.events import TraceEvent
        reference = replay_mod.reference_run(load_sample())
        # Sabotage the reference terminal outcome to force an E_REPLAY delta.
        for record in reference.rt.rounds:
            record.events[:] = [
                TraceEvent(
                    round_no=event.round_no, clock_ms=event.clock_ms,
                    kind="FAIL" if event.kind == "COMPLETE" else event.kind,
                    task_id=event.task_id, shard_index=event.shard_index,
                    pool_id=event.pool_id, gpu_count=event.gpu_count,
                    memory_mib=event.memory_mib, detail=event.detail)
                for event in record.events
            ]
        with self.assertRaises(SchedulerError) as ctx:
            resume_session(
                load_sample(), self.ledger_path, self.snapshot_dir,
                self.backup_dir, reference_rounds=reference.rt.rounds)
        self.assertEqual(ctx.exception.code, E_REPLAY)
        self.assertIn("round:", ctx.exception.problems[0].path)

    def test_snapshot_version_drift_replaces_whole_state(self):
        declaration = load_sample()
        session = SchedulingSession(
            declaration, self.ledger_path, self.snapshot_dir, self.backup_dir)
        session.run()
        session.close()
        original = snapshot_mod.read_snapshot(self.snapshot_dir)
        drifted = dict(original)
        drifted["task_version"] = "0" * 64
        self.assertFalse(snapshot_mod.versions_match(declaration, drifted))
        backup = snapshot_mod.replace_stale_snapshot(
            self.snapshot_dir, self.backup_dir)
        self.assertTrue(os.path.isdir(backup))
        self.assertFalse(os.path.exists(
            os.path.join(self.snapshot_dir, "snapshot.json")))


if __name__ == "__main__":
    unittest.main()
