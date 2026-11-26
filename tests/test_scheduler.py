from __future__ import annotations

import unittest

from tests.support import SampleWorkspace


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = SampleWorkspace()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sample.close()

    def test_pending_tasks_are_prioritized_on_next_batch(self) -> None:
        engine = self.sample.engine("priority", batch_point_budget=1)
        first = engine.advance()
        self.assertEqual(first["code"], "E_STALE_SUMMARY")
        first_computed = first["batch"]["computed_tasks"]
        self.assertEqual(len(first_computed), 1)
        pending_keys = {
            (item["frame"], item["step"])
            for item in first["pending"]
            if item["stage"] == "denoise"
        }
        second = engine.advance()
        second_computed = second["batch"]["computed_tasks"]
        self.assertTrue(second_computed)
        first_task = second_computed[0]
        self.assertEqual(first_task["stage"], "denoise")
        self.assertIn((first_task["frame"], first_task["step"]), pending_keys)

    def test_concurrency_does_not_change_pixels_or_step_summaries(self) -> None:
        serial = self.sample.engine("serial", max_concurrency=1)
        parallel = self.sample.engine("parallel", max_concurrency=4)
        self.assertEqual(serial.drain_current()["code"], "OK")
        self.assertEqual(parallel.drain_current()["code"], "OK")
        self.assertEqual(serial.results()["frames"], parallel.results()["frames"])
        self.assertEqual(
            serial.results()["denoise_steps"],
            parallel.results()["denoise_steps"],
        )
        self.assertEqual(serial.results()["rows"], parallel.results()["rows"])

    def test_snapshot_version_advances_only_when_state_changes(self) -> None:
        engine = self.sample.engine("snapshot")
        self.assertEqual(engine.drain_current()["code"], "OK")
        version = engine.snapshot_store.current_version()
        count = len(engine.snapshot_store.versions())
        again = engine.advance()
        self.assertEqual(again["code"], "OK")
        self.assertEqual(engine.snapshot_store.current_version(), version)
        self.assertEqual(len(engine.snapshot_store.versions()), count)
        self.assertFalse(again["snapshot"]["new_version_created"])

    def test_partial_snapshot_restore_is_exact(self) -> None:
        engine = self.sample.engine("restorable")
        partial = engine.advance()
        self.assertEqual(partial["code"], "E_STALE_SUMMARY")
        self.assertTrue(partial["pending"])
        restored = self.sample.engine("restorable")
        self.assertEqual(restored.cursor, engine.cursor)
        self.assertEqual(len(restored.pending_denoise), len(engine.pending_denoise))
        final = restored.drain_current()
        self.assertEqual(final["code"], "OK")

        full = self.sample.engine("restore-full")
        self.assertEqual(full.drain_current()["code"], "OK")
        self.assertEqual(restored.results()["frames"], full.results()["frames"])
        self.assertEqual(restored.results()["denoise_steps"], full.results()["denoise_steps"])


if __name__ == "__main__":
    unittest.main()
