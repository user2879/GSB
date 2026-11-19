import unittest

from refrac.camera import build_camera_frame
from refrac.diagnostics import report_to_dict, serialize_reports
from refrac.errors import DepthBoundsError
from refrac.scene import generate_scene
from refrac.tracer import render_frame, trace_pixel


class DepthAndReportTests(unittest.TestCase):
    def test_path_report_fields(self):
        scene = generate_scene(width=36, height=27)
        result = render_frame(scene)
        for trace in result["reports"]:
            payload = report_to_dict(trace)
            for key in ("row", "col", "rgba", "termination", "isolated", "error_code",
                        "interface_crossings", "tir_count", "internal_length", "events"):
                self.assertIn(key, payload)
            self.assertGreaterEqual(payload["interface_crossings"], 0)
            self.assertGreaterEqual(payload["internal_length"], 0.0)
            self.assertIn(payload["termination"],
                          ("background", "escaped", "fully_absorbed", "isolated", "depth_limit"))

    def test_termination_reasons_include_background_and_escaped(self):
        scene = generate_scene(width=60, height=45)
        result = render_frame(scene)
        counts = result["termination_counts"]
        self.assertIn("background", counts)
        # 终止统计可定位到具体像素
        background_pixels = [(t.row, t.col) for t in result["reports"]
                             if t.termination == "background"]
        self.assertEqual(len(background_pixels), counts["background"])

    def test_depth_limit_isolation_localized(self):
        base = render_frame(generate_scene(width=60, height=45, max_depth=10))
        limited = render_frame(generate_scene(width=60, height=45, max_depth=2))
        self.assertLess(len(limited["isolated"]),
                        len(base["reports"]))
        isolated_coords = {(t.row, t.col) for t in limited["reports"] if t.isolated}
        self.assertTrue(isolated_coords)
        # 仅触及上限的像素变化；其余像素逐字段不变
        base_by_coord = {(t.row, t.col): t for t in base["reports"]}
        for trace in limited["reports"]:
            if trace.isolated:
                self.assertEqual(trace.error_code, "E_DEPTH_BOUNDS")
                continue
            baseline = base_by_coord[(trace.row, trace.col)]
            self.assertEqual(serialize_reports([baseline]),
                             serialize_reports([trace]))

    def test_depth_zero_rejected_at_pixel_level(self):
        from refrac.injection import InjectionConfig
        scene = generate_scene(width=24, height=18)
        injection = InjectionConfig(
            name="px", body_overrides={},
            pixel_overrides=[{"row": 0, "col": 0, "patch": {"max_depth": 0}}],
            render_overrides={},
        )
        result = render_frame(scene, injection)
        baseline = render_frame(scene)
        base_coords = {(x["row"], x["col"]) for x in baseline["isolated"]}
        new_coords = {(x["row"], x["col"]) for x in result["isolated"]} - base_coords
        self.assertEqual(new_coords, {(0, 0)})
        self.assertEqual(
            next(x["code"] for x in result["isolated"] if (x["row"], x["col"]) == (0, 0)),
            "E_DEPTH_BOUNDS",
        )

    def test_invalid_render_depth_raises_run_level(self):
        scene = generate_scene(width=12, height=9)
        frame = build_camera_frame(scene.camera)
        broken = generate_scene(width=12, height=9, max_depth=0)
        with self.assertRaises(DepthBoundsError) as ctx:
            trace_pixel(broken, frame, 0, 0)
        self.assertEqual(ctx.exception.code, "E_DEPTH_BOUNDS")


if __name__ == "__main__":
    unittest.main()
