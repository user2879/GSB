from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stemscope.samplegen import DEFAULT_SEED, generate_samples
from stemscope.workbench import Workbench


class BudgetSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.data = Path(cls._temp.name) / "sample_data"
        generate_samples(cls.data, seed=DEFAULT_SEED)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_budget_exceeded_returns_stale_summary_without_mislabeling(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workbench = Workbench(
                self.data,
                Path(temp) / "runtime",
                self.data / "remix_config.json",
            )
            response = workbench.submit(
                {
                    "id": "probe",
                    "type": "set_gain",
                    "track": "vocals",
                    "gain": 1.3,
                    "budget_probe": {"simulated_compute_ms": 800},
                },
                budget_ms=150,
            )
            self.assertEqual(response["status"], "budget_exceeded")
            self.assertEqual(response["code"], "E_INTERACT_BUDGET_EXCEEDED")
            self.assertFalse(response["is_latest"])
            self.assertNotEqual(
                response["summary"]["param_version"],
                response["requested_param_version"],
            )
            self.assertLessEqual(response["elapsed_ms"], response["budget_ms"])
            workbench.wait_for_idle()
            self.assertEqual(workbench.latest_result.param_version, 1)
            workbench.close()

    def test_snapshot_resume_reuses_parse_gain_and_mix_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            first = Workbench(self.data, runtime, self.data / "remix_config.json")
            initial_hash = first.latest_result.pcm_sha256
            first.close()
            resumed = Workbench(
                self.data,
                runtime,
                self.data / "remix_config.json",
                resume=True,
                start_worker=False,
            )
            self.assertTrue(resumed.snapshot_audit["reusable"])
            self.assertGreater(resumed.parse_stats["cache_hits"], 0)
            self.assertGreater(resumed.latest_result.cache_stats["gain_hits"], 0)
            self.assertGreater(resumed.latest_result.cache_stats["mix_hits"], 0)
            self.assertEqual(resumed.latest_result.pcm_sha256, initial_hash)
            self.assertEqual(resumed.snapshot.audit()["temp_dirs_remaining"], 0)
            resumed.close()

    def test_stale_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            first = Workbench(self.data, runtime, self.data / "remix_config.json")
            first.close()
            stale = Workbench(
                self.data,
                runtime,
                self.data / "remix_config.json",
                resume=True,
                expected_global_version="wrong-version",
                start_worker=False,
            )
            self.assertEqual(stale.snapshot_audit["status"], "stale")
            self.assertIn("E_SNAPSHOT_STALE", [error.code for error in stale.startup_failures])
            stale.close()


if __name__ == "__main__":
    unittest.main()
