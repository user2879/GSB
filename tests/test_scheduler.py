from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from affinity_scheduler.errors import E_SPEC, SpecError
from affinity_scheduler.invalidation import compute_invalidations
from affinity_scheduler.samples import write_sample
from affinity_scheduler.scheduler import (
    compare_results_stable,
    conclusion_bytes,
    schedule_full,
    schedule_incremental,
)
from affinity_scheduler.scoring import ScoringManager
from affinity_scheduler.spec import load_spec


class SchedulerIncrementalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sample = write_sample(self.root, port_base=40500)
        self.spec_v1 = load_spec(self.sample["spec"])
        self.spec_v2 = load_spec(self.sample["spec_v2"])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_incremental_subset_matches_full_and_preserves_unaffected_bytes(self) -> None:
        with ScoringManager(self.spec_v1) as manager:
            state_v1 = schedule_full(self.spec_v1, manager)
        invalidation = compute_invalidations(state_v1, self.spec_v2)
        self.assertEqual(
            invalidation.affected_pods,
            ("pod-0001", "pod-0003", "pod-0004"),
        )
        self.assertNotIn("pod-0002", invalidation.affected_pods)
        self.assertEqual([step["step"] for step in invalidation.steps], [1, 2, 3])
        with ScoringManager(self.spec_v2) as manager:
            incremental = schedule_incremental(
                self.spec_v2,
                manager,
                state_v1,
                invalidation.affected_pods,
            )
            full = schedule_full(self.spec_v2, manager)
        self.assertIsNone(compare_results_stable(incremental, full))
        self.assertEqual(
            conclusion_bytes(incremental["results"]["pod-0002"]),
            conclusion_bytes(state_v1["results"]["pod-0002"]),
        )
        self.assertEqual(
            incremental["results"]["pod-0002"]["receipts"],
            state_v1["results"]["pod-0002"]["receipts"],
        )

    def test_parameter_version_must_be_monotonic(self) -> None:
        with ScoringManager(self.spec_v2) as manager:
            state_v2 = schedule_full(self.spec_v2, manager)
        with self.assertRaises(SpecError) as caught:
            compute_invalidations(state_v2, self.spec_v1)
        self.assertEqual(caught.exception.code, E_SPEC)
        self.assertEqual(caught.exception.field_path, "parameter_version")


if __name__ == "__main__":
    unittest.main()
