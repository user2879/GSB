"""异常注入隔离：真实故障命中预期业务码，单视图异常不污染其余视图。"""
from __future__ import annotations

import unittest

from recon_workbench.config import RebuildConfig
from recon_workbench.injection import apply_injections, load_injection_config
from recon_workbench.loader import load_sample
from recon_workbench.runner import ReconstructionRunner

from ._helpers import SampleHarness


class InjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = SampleHarness()

    def tearDown(self) -> None:
        self.harness.cleanup()

    def _run_injection(self, injections, config_overrides=None):
        import json

        path = self.harness.root + "/inj.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"injections": injections}, handle)
        items = load_injection_config(path)
        injected = self.harness.root + "/injected_sample"
        hits = apply_injections(
            self.harness.sample_dir, injected, items, config_overrides
        )
        load = load_sample(injected)
        # 重新挂载注入标记
        from recon_workbench.cli import _attach_injection_markers

        _attach_injection_markers(injected, load)
        config = RebuildConfig()
        if any(i["type"] == "zero_voxel_size" for i in items):
            object.__setattr__(config, "voxel_size", 0.0)
        runner = ReconstructionRunner(
            load, config, self.harness.root + "/work",
            injection_hits=hits, concurrency=1,
        )
        report = runner.execute(report_path=self.harness.root + "/r.json")
        return report.report, hits, load

    def test_truncated_png_isolated(self) -> None:
        report, hits, load = self._run_injection(
            [{"type": "truncate_png", "view": 1, "keep_ratio": 0.3}]
        )
        codes = {f["code"] for f in report["failures"]}
        self.assertIn("E_PNG_TRUNCATED", codes)
        # 其余视图仍产出特征
        self.assertIn(0, [v for v in report["views_loaded"]])
        self.assertNotIn(1, report["views_loaded"])
        self.assertTrue(
            all(h["hit"] for h in report["injection_hits"])
        )

    def test_camera_missing_isolated(self) -> None:
        report, _, _ = self._run_injection(
            [{"type": "camera_missing", "view": 3}]
        )
        self.assertIn("E_CAMERA_MISSING",
                      {f["code"] for f in report["failures"]})

    def test_nonfinite_descriptor_isolated(self) -> None:
        report, _, _ = self._run_injection(
            [{"type": "nonfinite_descriptor", "view": 0}]
        )
        self.assertIn("E_FEATURE_NONFINITE",
                      {f["code"] for f in report["failures"]})

    def test_disparity_out_of_range(self) -> None:
        report, _, _ = self._run_injection(
            [{"type": "disparity_out_of_range", "view": 4,
              "forced_disparity": 0.0}]
        )
        self.assertIn("E_MIN_DISPARITY",
                      {f["code"] for f in report["failures"]})

    def test_zero_voxel_size_mapped(self) -> None:
        report, _, _ = self._run_injection(
            [{"type": "zero_voxel_size", "view": 2}]
        )
        hit = report["injection_hits"][0]
        self.assertEqual(hit["expected_code"], "E_VOXEL_ZERO")
        # 零体素在点云阶段抛错并被隔离记录
        self.assertTrue(
            any(f["code"] == "E_VOXEL_ZERO" for f in report["failures"])
        )

    def test_injection_ids_recorded(self) -> None:
        report, hits, _ = self._run_injection(
            [{"type": "truncate_png", "view": 1},
             {"type": "camera_missing", "view": 2}]
        )
        ids = {h["injection_id"] for h in report["injection_hits"]}
        self.assertEqual(ids, {"INJ01", "INJ02"})


if __name__ == "__main__":
    unittest.main()
