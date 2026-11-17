import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from mvrec.cli import run_pipeline
from mvrec.config import load_config
from mvrec.dataset import scan_views
from mvrec.errors import PipelineError
from mvrec.pngio import read_png, write_png
from mvrec.samplegen import generate_sample


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="mvrec_load_")
        self.sample_dir = os.path.join(self.workdir, "sample")
        generate_sample(self.sample_dir, 6)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_png_round_trip_and_corrupt_header_detection(self):
        path = os.path.join(self.workdir, "marker.png")
        image = np.full((12, 10), 200, dtype=np.uint8)
        write_png(path, image)
        gray, rgb = read_png(path)
        self.assertEqual(gray.shape, (12, 10))
        self.assertEqual(rgb.shape, (12, 10, 3))
        with open(path, "r+b") as handle:
            handle.seek(0)
            handle.write(b"XXXXXXXX")
        with self.assertRaises(PipelineError) as context:
            read_png(path)
        self.assertEqual(context.exception.code, "E_PNG_INVALID")

    def test_corrupt_png_view_goes_to_failure_list(self):
        image_path = os.path.join(self.sample_dir, "images", "view_001.png")
        with open(image_path, "r+b") as handle:
            handle.seek(-16, os.SEEK_END)
            handle.write(b"\x00" * 8)
        views, failures = scan_views(self.sample_dir)
        self.assertTrue(any(item["view_id"] == 1 and item["code"] == "E_PNG_INVALID"
                            for item in failures))
        self.assertTrue(all(view["view_id"] != 1 for view in views))

    def test_missing_camera_maps_to_e_camera_invalid(self):
        cameras_path = os.path.join(self.sample_dir, "cameras.json")
        cameras = json.load(open(cameras_path, encoding="utf-8"))
        cameras = [cam for cam in cameras if cam["view_id"] != 2]
        json.dump(cameras, open(cameras_path, "w", encoding="utf-8"))
        views, failures = scan_views(self.sample_dir)
        self.assertTrue(any(item["view_id"] == 2 and item["code"] == "E_CAMERA_INVALID"
                            for item in failures))

    def test_singular_intrinsics_rejected(self):
        cameras_path = os.path.join(self.sample_dir, "cameras.json")
        cameras = json.load(open(cameras_path, encoding="utf-8"))
        cameras[0]["K"] = [[1.0, 0.0, 160.0], [0.0, 1.0, 120.0], [0.0, 0.0, 0.0]]
        json.dump(cameras, open(cameras_path, "w", encoding="utf-8"))
        _, failures = scan_views(self.sample_dir)
        self.assertTrue(any(item["code"] == "E_CAMERA_INVALID" for item in failures))

    def test_illegal_config_rejected(self):
        config_path = os.path.join(self.workdir, "bad.json")
        json.dump({"feature_threshold": -1}, open(config_path, "w", encoding="utf-8"))
        with self.assertRaises(PipelineError) as context:
            load_config(config_path)
        self.assertEqual(context.exception.code, "E_CONFIG_INVALID")

    def test_zero_voxel_size_maps_to_e_voxel_invalid(self):
        base = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "recon_config.json"), encoding="utf-8"))
        base["voxel_size"] = 0
        config_path = os.path.join(self.workdir, "voxel.json")
        json.dump(base, open(config_path, "w", encoding="utf-8"))
        report_path = os.path.join(self.workdir, "report.json")
        with self.assertRaises(PipelineError) as context:
            run_pipeline(self.sample_dir, config_path, None,
                         os.path.join(self.workdir, "snap"), report_path)
        self.assertEqual(context.exception.code, "E_VOXEL_INVALID")


class InjectionReportTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="mvrec_inj_")
        self.sample_dir = os.path.join(self.workdir, "sample")
        generate_sample(self.sample_dir, 8)
        self.base = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                                 "configs", "recon_config.json"),
                                   encoding="utf-8"))

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _injection_file(self, entries):
        path = os.path.join(self.workdir, "injections.json")
        json.dump({"injections": entries}, open(path, "w", encoding="utf-8"))
        return path

    def test_all_injection_kinds_hit_expected_business_codes(self):
        entries = [
            {"id": "INJ-001", "view_id": 1, "effects": {"corrupt_png": True}},
            {"id": "INJ-002", "view_id": 3, "effects": {"missing_camera": True}},
            {"id": "INJ-003", "view_id": 4, "effects": {"parallax_violation": True}},
            {"id": "INJ-004", "view_id": 5, "effects": {"nonfinite_feature": True}},
        ]
        injection_path = self._injection_file(entries)
        config_path = os.path.join(self.workdir, "config.json")
        json.dump(self.base, open(config_path, "w", encoding="utf-8"))
        report, _ = run_pipeline(
            self.sample_dir, config_path, injection_path,
            os.path.join(self.workdir, "snap"),
            os.path.join(self.workdir, "report.json"),
        )
        codes = {failure["code"] for failure in report["failures"]}
        self.assertIn("E_PNG_INVALID", codes)
        self.assertIn("E_CAMERA_INVALID", codes)
        self.assertIn("E_PARALLAX_INVALID", codes)
        self.assertIn("E_FEATURE_NONFINITE", codes)
        hit_ids = set(report["hit_injection_ids"])
        self.assertEqual(hit_ids, {"INJ-001", "INJ-002", "INJ-003", "INJ-004"})
        for entry in report["injection_hits"]:
            self.assertTrue(entry["hit"])

    def test_budget_zero_injections_are_reported(self):
        injection_path = self._injection_file([
            {"id": "INJ-005", "view_id": 0, "effects": {"slot_budget_zero": True}},
        ])
        config_path = os.path.join(self.workdir, "config_slot.json")
        json.dump(self.base, open(config_path, "w", encoding="utf-8"))
        report, _ = run_pipeline(
            self.sample_dir, config_path, injection_path,
            os.path.join(self.workdir, "snap_slot"),
            os.path.join(self.workdir, "report_slot.json"),
        )
        codes = {failure["code"] for failure in report["failures"]}
        self.assertIn("E_SLOT_BUDGET_EXHAUSTED", codes)
        self.assertIn("INJ-005", report["hit_injection_ids"])

    def test_single_view_failure_does_not_stop_other_views(self):
        injection_path = self._injection_file([
            {"id": "INJ-004", "view_id": 5, "effects": {"nonfinite_feature": True}},
        ])
        config_path = os.path.join(self.workdir, "config_iso.json")
        json.dump(self.base, open(config_path, "w", encoding="utf-8"))
        report, _ = run_pipeline(
            self.sample_dir, config_path, injection_path,
            os.path.join(self.workdir, "snap_iso"),
            os.path.join(self.workdir, "report_iso.json"),
        )
        self.assertGreater(report["point_cloud_size"], 0)
        self.assertTrue(
            any(failure["view_id"] == 5 and failure["stage"] == "features"
                for failure in report["failures"])
        )


if __name__ == "__main__":
    unittest.main()
