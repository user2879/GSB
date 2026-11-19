import unittest

import numpy as np

from refrac.absorption import ANALYTIC_TOL, integrate_transmittance, is_fully_absorbed
from refrac.errors import AbsorbNegativeError
from refrac.scene import generate_scene
from refrac.tracer import render_frame


class AbsorptionTests(unittest.TestCase):
    def test_quadrature_matches_analytic(self):
        sigma = np.array([0.12, 0.25, 0.4])
        for samples in (1, 2, 4, 8, 16):
            sampled, analytic, deviation = integrate_transmittance(sigma, 3.7, samples)
            np.testing.assert_allclose(sampled, analytic, atol=ANALYTIC_TOL)
            self.assertLessEqual(deviation, ANALYTIC_TOL)

    def test_beer_lambert_values(self):
        sigma = np.array([0.5, 0.5, 0.5])
        _, analytic, _ = integrate_transmittance(sigma, 2.0, 4)
        np.testing.assert_allclose(analytic, np.full(3, math_exp(-1.0)), atol=1e-12)

    def test_negative_absorption_rejected(self):
        with self.assertRaises(AbsorbNegativeError) as ctx:
            integrate_transmittance([0.1, -0.2, 0.3], 1.0, 4, body_id="bad")
        self.assertEqual(ctx.exception.code, "E_ABSORB_NEGATIVE")

    def test_fully_absorbed_threshold(self):
        self.assertTrue(is_fully_absorbed(np.array([1e-8, 0.5, 0.5])))
        self.assertFalse(is_fully_absorbed(np.array([0.5, 0.5, 0.5])))

    def test_render_absorption_events_analytic(self):
        scene = generate_scene(width=40, height=30)
        result = render_frame(scene)
        events = [e for t in result["reports"] for e in t.events if e["type"] == "absorption"]
        self.assertTrue(events)
        for event in events:
            self.assertLessEqual(event["absorption_deviation"], 1e-10)
            self.assertEqual(event["sample_count"], scene.sample_count)
            self.assertGreater(event["segment_length"], 0.0)

    def test_sample_count_change_only_touches_absorbing_pixels(self):
        base = render_frame(generate_scene(width=40, height=30, sample_count=8))
        variant = render_frame(generate_scene(width=40, height=30, sample_count=32))
        absorbing = {(t.row, t.col) for t in base["reports"] if t.internal_length > 0.0}
        diff_pixels = np.argwhere(np.any(base["rgba"] != variant["rgba"], axis=2))
        for row, col in diff_pixels:
            self.assertIn((int(row), int(col)), absorbing)
        # 常量吸收系数下两种采样数学等价，允许完全无差异
        outside = [
            (int(r), int(c)) for r, c in diff_pixels
            if (int(r), int(c)) not in absorbing
        ]
        self.assertEqual(outside, [])


def math_exp(value: float) -> float:
    return np.exp(value)


if __name__ == "__main__":
    unittest.main()
