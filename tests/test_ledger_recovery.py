import os
import tempfile
import unittest

from workbench.engine import ScalingEngine
from workbench.errors import ReplayError
from workbench.latency import LatencyRecorder
from workbench.ledger import Ledger
from workbench.logging_utils import JsonlLogger
from workbench.sample_data import build_cluster
from workbench.testing_utils import spec_from_payload


class LedgerRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.db = os.path.join(self.base, "ledger.db")
        self.log = os.path.join(self.base, "events.jsonl")
        self.spec = spec_from_payload(build_cluster())

    def _engine(self):
        logger = JsonlLogger(self.log)
        ledger = Ledger(self.db)
        recorder = LatencyRecorder(50, 1000, logger)
        return ledger, ScalingEngine(ledger, logger, recorder), logger

    def test_transactional_minute_commit_and_recovery_unknown(self):
        ledger, engine, _ = self._engine()
        version = engine.register_declaration(self.spec)
        engine.run_pipeline(self.spec, version)
        ledger.close()

        reopened = Ledger(self.db)
        logger = JsonlLogger(self.log)
        engine2 = ScalingEngine(reopened, logger)
        state = engine2.recovery_state(self.spec, version)
        self.assertEqual(state["committed_minutes"], self.spec.minutes)
        self.assertEqual(state["unknown_minutes"], [])
        reopened.verify_snapshots(version)
        reopened.close()

    def test_simulated_crash_leaves_unknown_minutes(self):
        ledger, engine, _ = self._engine()
        version = engine.register_declaration(self.spec)
        full_rows, incremental_rows, _ = __import__(
            "workbench.estimator", fromlist=["run_both_paths"]
        ).run_both_paths(self.spec)
        for minute in range(7):
            decision = engine._build_minute_decision(
                self.spec, version, minute, full_rows, incremental_rows,
                *__import__("workbench.estimator", fromlist=["select_path"]).select_path(self.spec),
            )
            ledger.commit_minute(decision)
        ledger.close()

        reopened = Ledger(self.db)
        logger = JsonlLogger(self.log)
        engine2 = ScalingEngine(reopened, logger)
        state = engine2.recovery_state(self.spec, version)
        self.assertEqual(state["committed_minutes"], 7)
        self.assertEqual(state["unknown_minutes"], list(range(7, self.spec.minutes)))
        report = engine2.run_pipeline(self.spec, version)
        self.assertEqual(report.unknown_minutes, list(range(7, self.spec.minutes)))
        self.assertEqual(len(reopened.load_decisions(version)), self.spec.minutes)
        engine2.verify_replay(self.spec, version)
        reopened.close()

    def test_version_evolution_marks_old_version_stale(self):
        ledger, engine, _ = self._engine()
        v1 = engine.register_declaration(self.spec)
        engine.run_pipeline(self.spec, v1)

        payload = build_cluster()
        payload["global_replica_budget"] = self.spec.global_replica_budget + 5
        evolved = spec_from_payload(payload)
        v2 = engine.register_declaration(evolved, reason="budget_change")
        engine.run_pipeline(evolved, v2)
        self.assertEqual(ledger.active_version(), v2)
        self.assertEqual(ledger.stale_versions(), [v1])
        self.assertEqual(ledger.progress(v1)["status"], "complete")
        ledger.close()

    def test_replay_mismatch_raises_e_replay(self):
        ledger, engine, _ = self._engine()
        version = engine.register_declaration(self.spec)
        engine.run_pipeline(self.spec, version)

        payload = build_cluster()
        payload["services"][0]["latency_ms"] = 300
        altered = spec_from_payload(payload)
        with self.assertRaises(ReplayError) as caught:
            engine.verify_replay(altered, version)
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.details["service_id"], "svc-embedding")
        self.assertIn("field", caught.exception.details)
        ledger.close()


if __name__ == "__main__":
    unittest.main()
