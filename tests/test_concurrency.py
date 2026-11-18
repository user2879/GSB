import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_samples import write_bundle
from texmat.testkit import make_engine, snapshot_arrays


class ConcurrencyDeterminismTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sample = self.tmp / "sample"
        write_bundle(self.sample, size=20, workers=1, budget=100000)

    def tearDown(self):
        self._tmp.cleanup()

    def _run_with_workers(self, workers: int, snapshot: Path) -> dict[str, np.ndarray]:
        engine = make_engine(self.sample, snapshot)
        object.__setattr__(engine.config, "workers", workers)
        result = engine.execute_run()
        while not result.complete and not result.stopped_by_signal:
            engine = make_engine(self.sample, snapshot)
            object.__setattr__(engine.config, "workers", workers)
            result = engine.execute_run()
        return snapshot_arrays(snapshot)

    def test_worker_counts_produce_identical_pixels(self):
        baseline = self._run_with_workers(1, self.tmp / "w1")
        for workers in (2, 4, 8):
            other = self._run_with_workers(workers, self.tmp / f"w{workers}")
            self.assertEqual(set(baseline), set(other))
            for key in baseline:
                np.testing.assert_array_equal(baseline[key], other[key], err_msg=key)

    def test_budget_deduction_order_is_stable(self):
        # Tight budget forces many admitted/pending decisions.
        engine = make_engine(self.sample, self.tmp / "budget")
        object.__setattr__(engine.config, "point_budget", 250)
        result = engine.execute_run()
        keys = [event["order_key"] for event in result.budget_events]
        self.assertEqual(keys, sorted(keys))
        actions = [event["action"] for event in result.budget_events]
        self.assertIn("pending", actions)
        # pending failures use the same stable keys
        failure_keys = [item["order_key"] for item in result.failures]
        self.assertEqual(failure_keys, sorted(failure_keys))

    def test_pending_marker_code(self):
        engine = make_engine(self.sample, self.tmp / "pending")
        object.__setattr__(engine.config, "point_budget", 1)
        result = engine.execute_run()
        codes = {item["code"] for item in result.failures}
        self.assertIn("E_POINT_BUDGET_PENDING", codes)
        self.assertTrue(result.pending)
        self.assertFalse(result.complete)


if __name__ == "__main__":
    unittest.main()
