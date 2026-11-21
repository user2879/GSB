import tempfile
import unittest
from pathlib import Path

from tier_mover.applog import StructuredLogger
from tier_mover.engine import TierEngine
from tier_mover.errors import ReplayError, VersionConflictError
from tier_mover.faults import inject_version_conflict
from tier_mover.ledger import Ledger
from tier_mover.planner import derive_plan
from tier_mover.recovery import RecoveryService
from tier_mover.replay import ReplayService
from tier_mover.workspace import write_sample_manifests


class RecoveryReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = write_sample_manifests(self.root, seed=99)
        self.logger = StructuredLogger(self.root / "logs" / "t.jsonl")
        self.engine = TierEngine(self.root, logger=self.logger)
        self.engine.prepare_dirs(self.manifest)
        self.ledger = Ledger(self.root / "ledger.db")
        self.ledger.register_spec(self.manifest, 100)
        self.engine.seed_objects(self.manifest)
        self.ledger.initialize_objects(self.manifest, 101)

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def _crash(self):
        plan = self.engine.plan(self.ledger, self.manifest)
        crash_key = plan.steps[0].key
        with self.assertRaises(SystemExit):
            self.engine.execute(self.ledger, self.manifest, plan=plan,
                                crash_after_key=crash_key)
        self.ledger.close()
        return crash_key

    def test_unknown_objects_detected_after_forced_termination(self):
        crash_key = self._crash()
        ledger = Ledger(self.root / "ledger.db")
        engine = TierEngine(self.root, logger=self.logger)
        report = RecoveryService(engine, ledger, self.manifest,
                                 self.logger).inspect()
        self.assertEqual(report.unknown_objects, 1)
        self.assertEqual(report.unknown_keys, [crash_key])
        self.assertEqual(len(report.open_batches), 1)
        ledger.close()

    def test_rollback_recovery_restores_source_tier(self):
        crash_key = self._crash()
        ledger = Ledger(self.root / "ledger.db")
        engine = TierEngine(self.root, logger=self.logger)
        service = RecoveryService(engine, ledger, self.manifest, self.logger)
        report = service.recover(mode="rollback")
        self.assertEqual(report.rolled_back_objects, 1)
        self.assertEqual(report.unknown_objects, 1)
        engine.reconcile_bytes(self.manifest, ledger)
        # 回退后续跑仍然成功
        ReplayService(ledger, self.manifest).run()
        ledger.close()

    def test_rebuild_recovery_adopts_target_tier(self):
        self._crash()
        ledger = Ledger(self.root / "ledger.db")
        engine = TierEngine(self.root, logger=self.logger)
        service = RecoveryService(engine, ledger, self.manifest, self.logger)
        report = service.recover(mode="rebuild")
        self.assertEqual(report.rebuilt_objects, 1)
        engine.reconcile_bytes(self.manifest, ledger)
        obj = next(o for o in self.manifest.objects
                   if o.key == report.resolutions[0]["key"])
        resolved_tier = report.resolutions[0]["resolved_tier"]
        self.assertTrue(engine.object_path(resolved_tier,
                                           obj.content_digest).exists())
        ledger.close()

    def test_replay_matches_ledger_on_happy_path(self):
        plan = self.engine.plan(self.ledger, self.manifest)
        self.engine.execute(self.ledger, self.manifest, plan=plan)
        result = ReplayService(self.ledger, self.manifest).run()
        self.assertEqual(result.steps_compared, len(plan.steps))
        self.assertEqual(result.final_placement, self.ledger.placement())

    def test_replay_reports_first_divergence_on_digest_mismatch(self):
        plan = self.engine.plan(self.ledger, self.manifest)
        self.engine.execute(self.ledger, self.manifest, plan=plan)
        # 篡改账本中第一条迁移记录的摘要，重放须在首个分歧点报 E_REPLAY
        first = self.ledger.conn.execute(
            "SELECT migration_id, key FROM migrations ORDER BY migration_id "
            "LIMIT 1").fetchone()
        self.ledger.conn.execute(
            "UPDATE migrations SET content_digest=? WHERE migration_id=?",
            ("ff" * 32, first["migration_id"]))
        self.ledger.conn.commit()
        with self.assertRaises(ReplayError) as ctx:
            ReplayService(self.ledger, self.manifest).run()
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.object_key, first["key"])
        self.assertEqual(ctx.exception.mismatch_field, "content_digest")

    def test_version_conflict_injection_raises_e_version(self):
        conflict = inject_version_conflict(self.root, self.logger)
        with self.assertRaises(VersionConflictError) as ctx:
            Ledger(conflict)
        self.assertEqual(ctx.exception.code, "E_VERSION")
        # 原账本依旧可打开、可恢复
        healthy = Ledger(self.root / "ledger.db")
        self.assertTrue(healthy.current_view())
        healthy.close()


if __name__ == "__main__":
    unittest.main()
