"""float32/float64 双路径：逐级误差归因与体素分歧边界样本。"""
from __future__ import annotations

import unittest

from recon_workbench.artifact_store import ArtifactStore
from recon_workbench.config import RebuildConfig
from recon_workbench.engine import WorkbenchEngine
from recon_workbench.precision_analysis import analyze
from recon_workbench.runner import ReconstructionRunner

from ._helpers import SampleHarness


class PrecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = SampleHarness()
        self.config = RebuildConfig()

    def tearDown(self) -> None:
        self.harness.cleanup()

    def _both_tracks(self, work_dir: str):
        load = self.harness.load()
        store = ArtifactStore(work_dir)
        results = {}
        for precision in ("float32", "float64"):
            engine = WorkbenchEngine(
                load, self.config, store, concurrency=2
            )
            raw = engine.run(precision, version=1)
            results[precision] = {
                "features_flat": raw.features_flat,
                "matches_flat": raw.matches_flat,
                "tripoints_flat": raw.tripoints_flat,
                "depth_maps": raw.depth_maps,
                "cloud": raw.cloud,
                "raw": raw,
            }
        return results

    def test_stage_error_structure(self) -> None:
        results = self._both_tracks(self.harness.root + "/work")
        report = analyze(results, self.config)
        stages = [s["stage"] for s in report["stage_errors"]]
        self.assertEqual(
            stages,
            ["feature", "match", "triangulation", "depth", "point_cloud"],
        )
        for stage in report["stage_errors"]:
            self.assertGreaterEqual(stage["max_abs_error"], 0.0)
            self.assertGreaterEqual(stage["max_rel_error"], 0.0)

    def test_first_breach_marked_once(self) -> None:
        results = self._both_tracks(self.harness.root + "/work2")
        report = analyze(results, self.config)
        breaches = [s for s in report["stage_errors"] if s["first_breach"]]
        self.assertLessEqual(len(breaches), 1)
        # 三角化是首个受 float32 显著影响的几何环节
        if breaches:
            self.assertEqual(
                report["first_breach_stage"], breaches[0]["stage"]
            )

    def test_voxel_divergence_classified(self) -> None:
        results = self._both_tracks(self.harness.root + "/work3")
        report = analyze(results, self.config)
        divergences = [
            b for b in report["boundary_samples"]
            if b["kind"] == "voxel_divergence"
        ]
        for sample in divergences:
            self.assertIn(
                sample["source"],
                {"triangulation", "quantization_rounding"},
            )
            self.assertNotEqual(
                sample["detail"]["voxel_float32"],
                sample["detail"]["voxel_float64"],
            )

    def test_runner_report_contains_analysis(self) -> None:
        load = self.harness.load()
        runner = ReconstructionRunner(
            load, self.config, self.harness.root + "/work4",
            concurrency=1,
        )
        report = runner.execute(
            report_path=self.harness.root + "/report.json"
        ).report
        self.assertIn("precision_analysis", report)
        self.assertIn("injection_hits", report)
        self.assertIn("snapshot_version", report)


if __name__ == "__main__":
    unittest.main()
