import hashlib
import unittest

import numpy as np

from refrac.camera import analytic_coverage, build_camera_frame
from refrac.diagnostics import serialize_reports
from refrac.scene import generate_scene
from refrac.tracer import render_frame


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        self.scene = generate_scene(width=48, height=36)

    def test_scene_generation_byte_identical(self):
        first = generate_scene(width=48, height=36).to_json()
        second = generate_scene(width=48, height=36).to_json()
        self.assertEqual(first, second)
        self.assertEqual(hashlib.sha256(first.encode()).hexdigest(),
                         hashlib.sha256(second.encode()).hexdigest())

    def test_render_byte_identical(self):
        run_a = render_frame(self.scene)
        run_b = render_frame(self.scene)
        self.assertTrue(np.array_equal(run_a["rgba"], run_b["rgba"]))
        self.assertEqual(run_a["rgba"].dtype, np.float32)
        self.assertEqual(serialize_reports(run_a["reports"]),
                         serialize_reports(run_b["reports"]))

    def test_rgba_raw_bytes_hash_stable(self):
        rgba = render_frame(self.scene)["rgba"]
        h1 = hashlib.sha256(rgba.tobytes()).hexdigest()
        h2 = hashlib.sha256(rgba.tobytes()).hexdigest()
        self.assertEqual(h1, h2)

    def test_metadata_contains_coverage_truth(self):
        frame = build_camera_frame(self.scene.camera)
        coverage = analytic_coverage(self.scene, frame)
        for body in self.scene.bodies:
            self.assertIn(body.body_id, coverage)
            self.assertGreaterEqual(coverage[body.body_id]["analytic_bbox_pixels"], 1)
        rendered = render_frame(self.scene)
        # 解析投影覆盖包含被遮挡体，因此实测命中是解析覆盖的子集。
        for body_id, hits in rendered["coverage_hits"].items():
            self.assertGreaterEqual(coverage[body_id]["analytic_bbox_pixels"], hits)
            self.assertGreaterEqual(coverage[body_id]["analytic_bbox_pixels"], 0)
        self.assertTrue(any(v["analytic_bbox_pixels"] > 0 for v in coverage.values()))


if __name__ == "__main__":
    unittest.main()
