from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from peft_lora.base import load_base_package
from peft_lora.merge import load_merged_package, restore_base_weights
from peft_lora.training import adapter_delta

from .common import SAMPLES, adapter_payloads, copy_samples, read_report, run_workspace


class MergeTests(unittest.TestCase):
    def test_merge_is_ordered_quantized_and_reversible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = copy_samples(root / "samples")
            workspace = root / "work"
            self.assertEqual(run_workspace(samples, workspace).exit_code, 0)
            report = read_report(workspace)
            adapters = adapter_payloads(report, workspace)
            merged_path = workspace / report["merge"]["artifact"]
            manifest, merged_weights, _merged_biases = load_merged_package(merged_path)
            restored = restore_base_weights(
                merged_weights, manifest["tasks_applied"], adapters
            )
            base = load_base_package(SAMPLES / "base_v1")
            for layer_name, layer in base.layers.items():
                error = float(np.max(np.abs(restored[layer_name] - layer.weight)))
                self.assertLessEqual(error, manifest["quantization"]["allowed_restore_error"])

            expected = {
                name: np.array(layer.weight, dtype=np.float64, copy=True)
                for name, layer in base.layers.items()
            }
            for task_name in sorted(adapters):
                adapter = adapters[task_name]
                expected[adapter["target_layer"]] += adapter_delta(adapter)
            scale = int(manifest["quantization"]["scale"])
            for layer_name in expected:
                expected[layer_name] = np.rint(expected[layer_name] * scale) / scale
                self.assertTrue(np.array_equal(expected[layer_name], merged_weights[layer_name]))

    def test_budget_exhaustion_keeps_pending_queue_without_partial_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = copy_samples(
                root / "samples", [("weight_elements: 1000000", "weight_elements: 30")]
            )
            workspace = root / "work"
            result = run_workspace(samples, workspace)
            report = read_report(workspace)
            self.assertEqual(result.exit_code, 1)
            self.assertIn("E_MERGE_BUDGET_EXHAUSTED", report["business_codes"])
            self.assertEqual(
                [entry["task_name"] for entry in report["merge"]["merged_tasks"]],
                ["alpha_classify"],
            )
            self.assertEqual(
                [entry["task_name"] for entry in report["merge"]["pending_tasks"]],
                ["beta_extract", "delta_guard", "gamma_rerank"],
            )
            self.assertEqual(
                [entry["task_name"] for entry in report["failures"]],
                ["beta_extract", "delta_guard", "gamma_rerank"],
            )
            ledger = report["merge"]["budget"]["ledger"]
            self.assertEqual([entry["decision"] for entry in ledger], ["ALLOW", "DENY"])
            self.assertIn("weight_elements", ledger[-1]["deny_reasons"])

            merged_path = workspace / report["merge"]["artifact"]
            manifest, merged_weights, _ = load_merged_package(merged_path)
            base = load_base_package(SAMPLES / "base_v1")
            scale = int(manifest["quantization"]["scale"])
            self.assertTrue(
                np.array_equal(
                    merged_weights["layer_1"],
                    np.rint(base.layers["layer_1"].weight * scale) / scale,
                )
            )
            self.assertTrue(
                np.array_equal(
                    merged_weights["layer_2"],
                    np.rint(base.layers["layer_2"].weight * scale) / scale,
                )
            )

    def test_irreversible_merge_does_not_overwrite_previous_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = copy_samples(root / "samples")
            workspace = root / "work"
            self.assertEqual(run_workspace(samples, workspace).exit_code, 0)
            good_report = read_report(workspace)
            old_artifact = workspace / good_report["merge"]["artifact"]
            old_weight_file = old_artifact / "layers" / "layer_0.weight.bin"
            old_bytes = old_weight_file.read_bytes()

            config_path = samples / "peft_config.yaml"
            config = config_path.read_text(encoding="utf-8").replace(
                "tolerance_steps: 1", "tolerance_steps: 0"
            )
            config_path.write_text(config, encoding="utf-8")
            result = run_workspace(samples, workspace)
            report = read_report(workspace)
            self.assertEqual(result.exit_code, 1)
            self.assertIn("E_MERGE_NOT_REVERSIBLE", report["business_codes"])
            self.assertEqual(report["merge"]["status"], "ERROR")
            self.assertIsNone(report["merge"]["artifact"])
            self.assertEqual(old_weight_file.read_bytes(), old_bytes)


if __name__ == "__main__":
    unittest.main()

