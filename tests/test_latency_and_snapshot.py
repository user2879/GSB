from __future__ import annotations

import tempfile
import time
import unittest

from rope_extrap.cache import ResultCache
from rope_extrap.engine import DiagnosticEngine
from rope_extrap.models import AttentionSample, InteractionCommand
from rope_extrap.session import InteractiveSession
from rope_extrap.snapshot import SnapshotStore
from tests.helpers import CONFIG_SMALL, load_config_and_samples


class LatencyAndSnapshotTests(unittest.TestCase):
    def test_pending_response_and_later_queue_progress(self) -> None:
        config, samples, _ = load_config_and_samples(CONFIG_SMALL, count=24)
        session = InteractiveSession(config, samples)
        try:
            first = session.apply_command(
                InteractionCommand(
                    index=1,
                    command_id="switch",
                    command_type="set_scheme",
                    changes={"scheme": "ntk_aware"},
                )
            )
            self.assertTrue(first["pending"])
            self.assertIn("E_INTERACT_BUDGET_PENDING", first["business_codes"])
            self.assertLessEqual(first["summary"]["completed_samples"], 24)
            self.assertTrue(first["within_budget"])
            time.sleep(0.05)
            mid = session.apply_command(
                InteractionCommand(
                    index=2,
                    command_id="poll-mid",
                    command_type="poll",
                    changes={},
                    budget_ms=40,
                )
            )
            self.assertIn("E_INTERACT_BUDGET_PENDING", mid["business_codes"])
            queue = mid["pending_queue"]
            self.assertIsNotNone(queue["remaining_stage_tasks"])
            self.assertLess(queue["remaining_stage_tasks"], 24 * 3)
            final = session.apply_command(
                InteractionCommand(
                    index=3,
                    command_id="poll-final",
                    command_type="poll",
                    changes={},
                    budget_ms=120_000,
                )
            )
            self.assertFalse(final["pending"])
            self.assertEqual(final["summary"]["completed_samples"], 24)
            self.assertEqual(final["business_codes"], ["OK"])
        finally:
            session.close()

    def test_snapshot_resume_and_stale_detection(self) -> None:
        config, samples, _ = load_config_and_samples(count=5)
        engine = DiagnosticEngine(config, samples)
        plan = engine.run_full()
        report = engine.build_report(plan=plan)
        root = tempfile.mkdtemp(prefix="rope-snapshot-test-")
        store = SnapshotStore(root)
        manifest = store.commit(
            engine,
            global_version=7,
            completed_samples=[row["sample_id"] for row in report["samples"]],
        )
        self.assertEqual(manifest["global_version"], 7)
        loaded = store.load(
            expected_version=7,
            expected_config_hash=config.config_fingerprint(),
        )
        self.assertTrue(loaded.ok)
        self.assertEqual(len(loaded.entries), 15)
        stale = store.load(
            expected_version=8,
            expected_config_hash=config.config_fingerprint(),
        )
        self.assertFalse(stale.ok)
        self.assertEqual(stale.failures[0].code, "E_SNAPSHOT_STALE")
        wrong_hash = store.load(
            expected_version=7,
            expected_config_hash="0" * 64,
        )
        self.assertFalse(wrong_hash.ok)
        self.assertEqual(wrong_hash.failures[0].code, "E_SNAPSHOT_STALE")
        cache = ResultCache()
        loaded.apply_to(cache)
        resumed = DiagnosticEngine(config, samples, cache=cache)
        resumed_plan = resumed.build_plan()
        self.assertEqual(len(resumed_plan.reused), 15)
        self.assertEqual(
            sum(len(jobs) for jobs in resumed_plan.jobs_by_stage.values()),
            0,
        )


if __name__ == "__main__":
    unittest.main()
