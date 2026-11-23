import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvrb.diagnostics import (
    REASON_ROUNDING,
    REASON_SIDECAR_MISS,
    cross_version_report,
)
from mvrb.errors import INJECTION_CODES
from mvrb.exchange_v2 import read_v2
from mvrb.inject import apply_injections, load_injection_config
from mvrb.pipeline import ReconstructionPipeline
from mvrb.utils import read_json, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "demo")


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mvrb_snap_")
        self.work = os.path.join(self.tmp, "snap")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unchanged_run_keeps_stages_and_second_run_reuses(self):
        from mvrb.workbench import Workbench
        sample = os.path.join(self.tmp, "sample")
        shutil.copytree(SAMPLE, sample)
        wb = Workbench(sample, self.work)
        report1 = wb.run()
        self.assertTrue(all(a in ("replaced",) for a in report1["stage_commit_actions"].values()))
        report2 = Workbench(sample, self.work).run()
        self.assertTrue(all(a == "unchanged"
                            for a in report2["stage_commit_actions"].values()))
        self.assertFalse(report2["resume"]["stale"])

    def test_stale_snapshot_on_config_change(self):
        from mvrb.workbench import Workbench
        sample = os.path.join(self.tmp, "sample")
        shutil.copytree(SAMPLE, sample)
        Workbench(sample, self.work).run()
        cfg = read_json(os.path.join(sample, "recon_config.json"))
        cfg["feature_threshold"] = 100
        write_json(os.path.join(sample, "recon_config.json"), cfg)
        report = Workbench(sample, self.work).run()
        self.assertTrue(report["resume"]["stale"])
        self.assertIn("E_SNAPSHOT_STALE", [f["code"] for f in report["failures"]])


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mvrb_inject_")
        self.clean = os.path.join(self.tmp, "clean")
        shutil.copytree(SAMPLE, self.clean)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_injection(self, items):
        cfg_path = os.path.join(self.tmp, "injection.json")
        write_json(cfg_path, {"injections": items})
        loaded = load_injection_config(cfg_path)
        work_sample = os.path.join(self.tmp, "broken")
        manifest = apply_injections(self.clean, work_sample, loaded)
        result = ReconstructionPipeline(
            work_sample, os.path.join(self.tmp, "pipeline_work")).run()
        return manifest, result

    def test_truncated_png_is_detected(self):
        manifest, result = self._run_injection([
            {"id": "INJ-001", "fault": "truncated_png", "view": 1,
             "stage": "features", "keep_bytes": 24},
        ])
        self.assertTrue(manifest["hits"][0]["applied"])
        self.assertIn("E_TRUNCATED_PNG", [f.code for f in result.failures.sorted()])

    def test_missing_camera_is_detected(self):
        manifest, result = self._run_injection([
            {"id": "INJ-002", "fault": "missing_camera", "view": 2, "stage": "load"},
        ])
        self.assertIn("E_MISSING_CAMERA", [f.code for f in result.failures.sorted()])

    def test_disparity_overflow_is_detected(self):
        manifest, result = self._run_injection([
            {"id": "INJ-003", "fault": "disparity_overflow", "view": 0,
             "stage": "triangulate", "forced_min_disparity": 1e9},
        ])
        self.assertIn("E_DISPARITY_OUT_OF_RANGE",
                      [f.code for f in result.failures.sorted()])

    def test_zero_voxel_size_is_detected(self):
        manifest, result = self._run_injection([
            {"id": "INJ-004", "fault": "zero_voxel_size", "stage": "voxel"},
        ])
        self.assertIn("E_VOXEL_ZERO_SIZE", [f.code for f in result.failures.sorted()])

    def test_non_finite_feature_is_detected(self):
        # 非有限特征经交换文件加载校验路径
        manifest, _ = self._run_injection([
            {"id": "INJ-005", "fault": "non_finite_feature", "view": 0,
             "stage": "validate"},
        ])
        broken_v2 = os.path.join(self.tmp, "broken", "result_v2.json")
        from mvrb.exchange_v2 import parse_v2
        ir = parse_v2(read_json(broken_v2))
        import math
        self.assertTrue(any(math.isnan(f.x) for f in ir.features if f.view == 0))
        fails = ir.validate()
        self.assertIn("E_NON_FINITE", [f.code for f in fails.sorted()])

    def test_injection_ids_map_to_business_codes(self):
        items = [
            {"id": f"INJ-{i:03d}", "fault": fault, "view": 0, "stage": "diagnostic"}
            for i, fault in enumerate(INJECTION_CODES, start=1)
        ]
        manifest, result = self._run_injection(items)
        self.assertEqual(len(manifest["hits"]), len(INJECTION_CODES))
        for hit in manifest["hits"]:
            self.assertEqual(hit["code"], INJECTION_CODES[hit["fault"]])
            self.assertTrue(hit["id"].startswith("INJ-"))

    def test_each_injection_isolated_is_observed(self):
        # 每项注入独立工作目录运行，故障不得被其它故障或缓存复用屏蔽
        expected = {
            "truncated_png": "E_TRUNCATED_PNG",
            "missing_camera": "E_MISSING_CAMERA",
            "disparity_overflow": "E_DISPARITY_OUT_OF_RANGE",
            "zero_voxel_size": "E_VOXEL_ZERO_SIZE",
        }
        for idx, (fault, code) in enumerate(expected.items(), start=1):
            work_sample = os.path.join(self.tmp, f"iso_{fault}")
            work_pipe = os.path.join(self.tmp, f"iso_pipe_{fault}")
            item = [{"id": f"INJ-{idx:03d}", "fault": fault, "view": 0,
                     "stage": "diagnostic", "forced_min_disparity": 1e9,
                     "keep_bytes": 24}]
            apply_injections(self.clean, work_sample, item)
            result = ReconstructionPipeline(work_sample, work_pipe).run()
            self.assertIn(code, [f.code for f in result.failures.sorted()],
                          msg=fault)


class DiagnosticsTests(unittest.TestCase):
    def test_cross_version_report_structure_and_rounding_reason(self):
        ir = read_v2(os.path.join(SAMPLE, "result_v2.json"))
        report = cross_version_report(ir)
        self.assertTrue(report["v2_lossless"])
        self.assertTrue(report["exact_restore_with_sidecar"])
        self.assertIn("per_view_coord_histogram", report)
        self.assertGreaterEqual(report["max_score_abs_diff"], 0.0)
        self.assertTrue(all(item["ok"] for item in report["metric_restore"]))
        self.assertGreater(report["sidecar_hits"]["features"], 0)

    def test_reason_codes_distinguish_rounding_metric_sidecar(self):
        from mvrb.convert import downgrade_v2_to_v1, upgrade_v1_to_v2
        from mvrb.exchange_v1 import parse_v1, serialize_v1
        ir = read_v2(os.path.join(SAMPLE, "result_v2.json"))
        # 不带旁路升版 -> 浮点残差丢失 -> 舍入/旁路未命中原因
        ir_v1, _ = downgrade_v2_to_v1(ir)
        # 制造一个真实非整数浮点坐标，使无旁路还原产生偏差
        ir.features[0].x = ir.features[0].x + 0.37
        ir_v1, _ = downgrade_v2_to_v1(ir)
        upgrade_v1_to_v2(parse_v1(serialize_v1(ir_v1)))
        report = cross_version_report(ir)  # 带旁路，精确还原
        self.assertTrue(report["exact_restore_with_sidecar"])
        no_sidecar = cross_version_report(ir, sidecar={"features": {}, "matches": {},
                                                        "voxel_points": {}})
        joined = " ".join(str(v) for v in no_sidecar["view_reasons"].values())
        self.assertIn(REASON_ROUNDING, joined)
        self.assertIn(REASON_SIDECAR_MISS, joined)


if __name__ == "__main__":
    unittest.main()
