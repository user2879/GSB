from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .common import SAMPLES, adapter_payloads, copy_samples, read_report, run_workspace


class UpgradeTests(unittest.TestCase):
    def test_only_changed_target_layers_are_retrained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = copy_samples(root / "samples")
            upgraded_work = root / "upgraded"
            clean_work = root / "clean"
            self.assertEqual(run_workspace(samples, upgraded_work).exit_code, 0)
            self.assertEqual(
                run_workspace(
                    samples,
                    upgraded_work,
                    base_directory=SAMPLES / "base_v2",
                ).exit_code,
                0,
            )
            self.assertEqual(
                run_workspace(
                    samples,
                    clean_work,
                    base_directory=SAMPLES / "base_v2",
                ).exit_code,
                0,
            )
            upgraded = read_report(upgraded_work)
            clean = read_report(clean_work)
            self.assertEqual(upgraded["reuse"], ["alpha_classify", "delta_guard"])
            self.assertEqual(upgraded["retrain"], ["beta_extract", "gamma_rerank"])
            self.assertEqual(upgraded["training_executed"], ["beta_extract", "gamma_rerank"])
            self.assertEqual(
                clean["training_executed"],
                ["alpha_classify", "beta_extract", "delta_guard", "gamma_rerank"],
            )
            compatibility = {
                entry["task_name"]: entry for entry in upgraded["compatibility"]
            }
            self.assertEqual(
                compatibility["alpha_classify"]["status"], "COMPATIBLE_LAYER_UNCHANGED"
            )
            self.assertEqual(
                compatibility["delta_guard"]["status"], "COMPATIBLE_LAYER_UNCHANGED"
            )
            self.assertEqual(
                compatibility["beta_extract"]["business_code"], "E_BASE_VERSION_MISMATCH"
            )
            self.assertEqual(
                compatibility["gamma_rerank"]["business_code"], "E_BASE_VERSION_MISMATCH"
            )
            upgraded_adapters = adapter_payloads(upgraded, upgraded_work)
            clean_adapters = adapter_payloads(clean, clean_work)
            for task_name in upgraded_adapters:
                self.assertEqual(
                    upgraded_adapters[task_name]["A"], clean_adapters[task_name]["A"]
                )
                self.assertEqual(
                    upgraded_adapters[task_name]["B"], clean_adapters[task_name]["B"]
                )
                self.assertEqual(
                    upgraded_adapters[task_name]["loss_trajectory"],
                    clean_adapters[task_name]["loss_trajectory"],
                )
            self.assertGreater(upgraded["recovery"]["started_from_version"], 0)


if __name__ == "__main__":
    unittest.main()

