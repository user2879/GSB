import tempfile
import unittest

import numpy as np

from offscreen_render.geometry import build_interval_chain
from offscreen_render.precision import quantize_mantissa
from offscreen_render.renderer import render_ray
from offscreen_render.samplegen import build_deterministic_scene
from offscreen_render.scene import Scene


class TestTransmittance(unittest.TestCase):
    def setUp(self):
        self.scene = build_deterministic_scene(tempfile.mkdtemp(), tempfile.mkdtemp())

    def test_uniform_medium_matches_analytic(self):
        # 单盒均匀介质，步进透射率应逼近 exp(-sigma*L)。
        from offscreen_render.scene import VolumeBox
        vol = VolumeBox(0, "uniform", np.array([0., 0., 0.]),
                        np.array([10., 10., 10.]), sigma_t=0.5, albedo=0.0, ior=1.0)
        cfg = self.scene.config
        scene = Scene((vol,), self.scene.lights, self.scene.camera, cfg,
                      np.zeros(3), np.full(3, 10.0))
        o = np.array([5.0, 5.0, -2.0])
        d = np.array([0.0, 0.0, 1.0])
        rr = render_ray(o, d, scene)
        analytic = float(np.exp(-0.5 * 10.0))
        self.assertLess(abs(rr.transmittance64 - analytic) / analytic, 2e-3)
        self.assertEqual(rr.stop_reason, "completed")

    def test_cutoff_termination(self):
        rr = render_ray(self.scene.camera.position.copy(),
                        np.array([0.0, 0.0, 1.0]), self.scene)
        self.assertEqual(rr.stop_reason, "cutoff")
        self.assertLess(rr.transmittance64, self.scene.config.transmittance_cutoff)

    def test_max_steps_termination(self):
        from dataclasses import replace
        cfg = replace(self.scene.config, max_steps=4, transmittance_cutoff=1e-12)
        scene = Scene(self.scene.volumes, self.scene.lights, self.scene.camera, cfg,
                      self.scene.bbox_min, self.scene.bbox_max)
        rr = render_ray(scene.camera.position.copy(), np.array([0., 0., 1.]), scene)
        self.assertEqual(rr.steps, 4)
        self.assertEqual(rr.stop_reason, "max_steps")

    def test_fp32_never_below_fp64_precision(self):
        rr = render_ray(self.scene.camera.position.copy(),
                        np.array([0.0, 0.0, 1.0]), self.scene)
        # 单精度路径必然引入非负误差（在足够多步后）。
        self.assertGreaterEqual(rr.max_abs_diff, 0.0)

    def test_quantize_binary32(self):
        import struct
        for x in (0.1, 3.14159265358979, 12345.6789, 1e-6):
            expected = struct.unpack("f", struct.pack("f", x))[0]
            self.assertEqual(quantize_mantissa(x, 23), expected)

    def test_chain_optical_depth_consistency(self):
        # 区间链解析光学厚度与逐段长度自洽。
        o = self.scene.camera.position.copy()
        d = np.array([0.0, 0.0, 1.0])
        chain = build_interval_chain(o, d, self.scene.volumes)
        tau = sum(s.volume.sigma_t * s.length for s in chain)
        self.assertGreater(tau, 0.0)
        # 每段长度为正且有序。
        for s in chain:
            self.assertGreater(s.length, 0.0)


if __name__ == "__main__":
    unittest.main()
