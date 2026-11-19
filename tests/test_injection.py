import json
import unittest

from refrac.diagnostics import (
    build_minimal_counterexample, first_affected_pixel, replay_pixel,
    verify_counterexample_minimality,
)
from refrac.errors import ERROR_CODES, InjectConfigError
from refrac.injection import injection_from_dict
from refrac.inject_cases import build_injection_suite
from refrac.scene import dumps_canonical, generate_scene
from refrac.tracer import render_frame

EXPECTED_CODES = {
    "inj_ior_zero": "E_IOR_RANGE",
    "inj_ior_above_range": "E_IOR_RANGE",
    "inj_negative_radius": "E_INTERSECT_DEGENERATE",
    "inj_zero_direction": "E_RAY_DEGENERATE",
    "inj_depth_zero": "E_DEPTH_BOUNDS",
    "inj_absorb_negative": "E_ABSORB_NEGATIVE",
}


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.scene = generate_scene(width=48, height=36)
        self.center = (0, 0)
        self.suite = build_injection_suite(self.center)

    def test_injection_config_serialization_roundtrip(self):
        for injection in self.suite.values():
            text = injection.to_json()
            restored = injection_from_dict(json.loads(text))
            self.assertEqual(restored.to_json(), text)

    def test_expected_error_classification_and_isolation(self):
        for label, expected in EXPECTED_CODES.items():
            with self.subTest(label=label):
                result = render_frame(self.scene, self.suite[label])
                self.assertTrue(result["isolated"])
                first = first_affected_pixel(result["isolated"])
                self.assertEqual(first["code"], expected)
                # 像素级注入只在目标像素上新增隔离；其余隔离仅来自基线深度预算。
                if label in ("inj_zero_direction", "inj_depth_zero"):
                    baseline = render_frame(self.scene)
                    baseline_coords = {(x["row"], x["col"]) for x in baseline["isolated"]}
                    injected_coords = {(x["row"], x["col"]) for x in result["isolated"]}
                    self.assertEqual(injected_coords - baseline_coords, {self.center})
                    clean_pixels = [t for t in result["reports"] if not t.isolated]
                    self.assertTrue(clean_pixels)

    def test_isolated_pixels_marked_neither_silent_success_nor_loss(self):
        result = render_frame(self.scene, self.suite["inj_zero_direction"])
        isolated_trace = next(
            t for t in result["reports"]
            if t.isolated and (t.row, t.col) == self.center
        )
        self.assertEqual(isolated_trace.termination, "isolated")
        self.assertEqual(isolated_trace.error_code, "E_RAY_DEGENERATE")
        # 目标隔离像素可在终止统计中定位坐标，且记录带哨兵颜色
        record = next(x for x in result["isolated"]
                      if (x["row"], x["col"]) == self.center)
        self.assertEqual((record["row"], record["col"]), self.center)
        target_color = result["rgba"][self.center]
        self.assertEqual(tuple(target_color), (1.0, 0.0, 1.0, 1.0))

    def test_first_affected_pixel_row_major_order(self):
        result = render_frame(self.scene, self.suite["inj_ior_zero"])
        first = first_affected_pixel(result["isolated"])
        for record in result["isolated"]:
            self.assertLessEqual((first["row"], first["col"]),
                                 (record["row"], record["col"]))

    def test_minimal_counterexample_reproduces_and_is_minimal(self):
        for label, expected in EXPECTED_CODES.items():
            with self.subTest(label=label):
                injection = self.suite[label]
                result = render_frame(self.scene, injection)
                counterexample = build_minimal_counterexample(self.scene, injection, result["isolated"])
                self.assertEqual(counterexample["code"], expected)
                # 反例为可序列化 JSON 片段
                serialized = dumps_canonical(counterexample)
                json.loads(serialized)
                verification = verify_counterexample_minimality(
                    self.scene, injection, counterexample
                )
                self.assertTrue(verification["full_replay_matches"])
                self.assertTrue(verification["minimal"], verification)

    def test_replay_reproduces_same_code(self):
        injection = self.suite["inj_zero_direction"]
        result = render_frame(self.scene, injection)
        first = first_affected_pixel(result["isolated"])
        replay = replay_pixel(self.scene, injection, first["row"], first["col"])
        self.assertFalse(replay["ok"])
        self.assertEqual(replay["code"], "E_RAY_DEGENERATE")

    def test_run_level_invalid_injection_configs(self):
        # 渲染级 max_depth=0 与未知 body_id 属于运行级归类
        with self.assertRaises(Exception) as ctx_depth:
            render_frame(self.scene, self.suite["inj_depth_zero_render"])
        self.assertEqual(ctx_depth.exception.code, "E_DEPTH_BOUNDS")
        with self.assertRaises(InjectConfigError) as ctx_cfg:
            render_frame(self.scene, self.suite["inj_bad_config"])
        self.assertEqual(ctx_cfg.exception.code, "E_INJECT_CONFIG")

    def test_all_business_codes_defined(self):
        self.assertEqual(
            set(ERROR_CODES),
            {"E_IOR_RANGE", "E_RAY_DEGENERATE", "E_DEPTH_BOUNDS", "E_ABSORB_NEGATIVE",
             "E_INTERSECT_DEGENERATE", "E_PIXEL_ISOLATED", "E_INJECT_CONFIG"},
        )

    def test_injection_does_not_mutate_source_scene(self):
        before = self.scene.to_json()
        render_frame(self.scene, self.suite["inj_ior_zero"])
        self.assertEqual(self.scene.to_json(), before)

    def test_malformed_injection_config_rejected(self):
        bad_payloads = [
            {"name": ""},
            {"name": "x", "body_overrides": {"b": {}}},
            {"name": "x", "pixel_overrides": [{"row": -1, "col": 0, "patch": {"direction": [0, 0, 0]}}]},
            {"name": "x", "render_overrides": {"unknown": 1}},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(InjectConfigError) as ctx:
                    injection_from_dict(payload)
                self.assertEqual(ctx.exception.code, "E_INJECT_CONFIG")


if __name__ == "__main__":
    unittest.main()
