from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from peft_lora.base import generate_base_package, load_base_package
from peft_lora.errors import E_ADAPTER_CONFIG, E_BASE_INVALID, E_TASK_INVALID

from .common import copy_samples, read_report, run_workspace


class LoadingTests(unittest.TestCase):
    def test_base_v2_changes_only_declared_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shapes = {"layer_0": (4, 3), "layer_1": (3, 4), "layer_2": (5, 4)}
            v1 = generate_base_package(root / "v1", shapes)
            v2 = generate_base_package(
                root / "v2", shapes, changed_layers={"layer_1"}, variant_salt="test"
            )
            self.assertEqual(v1.layer_fingerprints["layer_0"], v2.layer_fingerprints["layer_0"])
            self.assertEqual(v1.layer_fingerprints["layer_2"], v2.layer_fingerprints["layer_2"])
            self.assertNotEqual(
                v1.layer_fingerprints["layer_1"], v2.layer_fingerprints["layer_1"]
            )
            self.assertNotEqual(v1.fingerprint, v2.fingerprint)

    def test_invalid_base_does_not_stop_task_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = copy_samples(root / "samples")
            manifest_path = samples / "base_v1" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["layers"].append(dict(manifest["layers"][0]))
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            tasks_path = samples / "tasks.json"
            tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
            tasks["tasks"].append(
                {
                    "task_name": "invalid_layer",
                    "target_layer": "not_in_whitelist",
                    "rank": 2,
                    "scale": 1.0,
                }
            )
            tasks_path.write_text(json.dumps(tasks), encoding="utf-8")

            result = run_workspace(samples, root / "work")
            report = read_report(root / "work")
            self.assertEqual(result.exit_code, 1)
            codes = {failure["code"] for failure in report["failures"]}
            self.assertIn(E_BASE_INVALID, codes)
            self.assertIn(E_TASK_INVALID, codes)
            self.assertEqual(len(report["tasks"]), 4)

    def test_invalid_task_and_adapter_config_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = copy_samples(root / "samples")
            tasks_path = samples / "tasks.json"
            tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
            tasks["tasks"].extend(
                [
                    {
                        "task_name": "bad_rank",
                        "target_layer": "layer_0",
                        "rank": 0,
                        "scale": 1.0,
                    },
                    {
                        "task_name": "bad_layer",
                        "target_layer": "unknown_layer",
                        "rank": 2,
                        "scale": 1.0,
                    },
                ]
            )
            tasks_path.write_text(json.dumps(tasks), encoding="utf-8")

            result = run_workspace(samples, root / "work")
            report = read_report(root / "work")
            codes = {failure["code"] for failure in report["failures"]}
            self.assertEqual(result.exit_code, 1)
            self.assertIn(E_ADAPTER_CONFIG, codes)
            self.assertIn(E_TASK_INVALID, codes)
            self.assertEqual(
                report["training_executed"],
                ["alpha_classify", "beta_extract", "delta_guard", "gamma_rerank"],
            )
            self.assertEqual(report["status"], "partial")


if __name__ == "__main__":
    unittest.main()

