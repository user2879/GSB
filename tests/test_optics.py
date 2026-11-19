import math
import unittest

import numpy as np

from refrac.errors import IorRangeError, RayDegenerateError
from refrac.optics import (
    IOR_MAX, IOR_MIN, critical_angle, refract_or_reflect, validate_ior,
)
from refrac.vectors import normalize
from tests.probe_scenes import tir_scene
from refrac.tracer import render_frame


class OpticsTests(unittest.TestCase):
    def test_snell_normal_incidence(self):
        d = np.array([0.0, 0.0, -1.0])
        n = np.array([0.0, 0.0, 1.0])
        result = refract_or_reflect(d, n, 1.0, 1.5)
        self.assertEqual(result.classification, "NO_CRITICAL_ANGLE")
        np.testing.assert_allclose(result.direction, d, atol=1e-12)
        self.assertIsNone(result.critical_angle)

    def test_snell_known_angle(self):
        # 空气->玻璃 1.5，入射角 30° -> 折射角 arcsin(sin30/1.5)
        d = normalize(np.array([math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30))]))
        n = np.array([0.0, 0.0, 1.0])
        result = refract_or_reflect(d, n, 1.0, 1.5)
        expected_t = math.asin(0.5 / 1.5)
        self.assertAlmostEqual(result.direction[0], math.sin(expected_t), places=10)
        self.assertAlmostEqual(-result.direction[2], math.cos(expected_t), places=10)

    def test_critical_angle_consistency(self):
        theta_c = critical_angle(1.8, 1.0)
        self.assertAlmostEqual(theta_c, math.asin(1.0 / 1.8), places=12)
        self.assertIsNone(critical_angle(1.0, 1.5))

    def test_total_internal_reflection_classification(self):
        # 玻璃 1.8 -> 空气，入射角 45° 超过临界角 33.75°
        d = normalize(np.array([math.sin(math.radians(45)), 0.0, -math.cos(math.radians(45))]))
        n = np.array([0.0, 0.0, 1.0])
        result = refract_or_reflect(d, n, 1.8, 1.0)
        self.assertEqual(result.classification, "TIR")
        self.assertGreater(result.sin2_theta_t, 1.0)
        # 反射角等于入射角
        self.assertAlmostEqual(result.direction[0], d[0], places=12)
        self.assertAlmostEqual(result.direction[2], -d[2], places=12)

    def test_degenerate_critical_angle_equal_ior(self):
        # ior 相等时临界角退化（不存在），归类 NO_CRITICAL_ANGLE 而非丢弃光线
        d = np.array([0.3, 0.0, -0.9539])
        n = np.array([0.0, 0.0, 1.0])
        result = refract_or_reflect(d, n, 1.0, 1.0)
        self.assertEqual(result.classification, "NO_CRITICAL_ANGLE")
        self.assertIsNone(result.critical_angle)

    def test_ior_validation_range(self):
        for bad in (0.0, -1.0, 3.01, float("nan"), float("inf")):
            with self.assertRaises(IorRangeError) as ctx:
                validate_ior(bad, body_id="b", role="transmitted")
            self.assertEqual(ctx.exception.code, "E_IOR_RANGE")
        validate_ior(IOR_MIN, body_id="b", role="incident")
        validate_ior(IOR_MAX, body_id="b", role="incident")

    def test_tir_probe_scene_reports_reflections(self):
        result = render_frame(tir_scene())
        tir_total = sum(t.tir_count for t in result["reports"])
        self.assertGreater(tir_total, 0)
        self.assertEqual(result["termination_counts"].get("isolated", 0), 0)
        events = [e for t in result["reports"] for e in t.events
                  if e["type"] == "interface" and e["classification"] == "TIR"]
        self.assertTrue(events)
        for event in events:
            expected = math.asin(1.0 / 1.8)
            self.assertAlmostEqual(event["critical_angle"], expected, places=10)
            self.assertGreater(event["sin2_theta_t"], 1.0)
            self.assertGreaterEqual(event["tir_index"], 1)


class VectorDegenerateTests(unittest.TestCase):
    def test_zero_normalize_raises(self):
        from refrac.vectors import normalize
        with self.assertRaises(RayDegenerateError) as ctx:
            normalize(np.zeros(3))
        self.assertEqual(ctx.exception.code, "E_RAY_DEGENERATE")


if __name__ == "__main__":
    unittest.main()
