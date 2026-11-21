import unittest

import numpy as np

from vocoder.fixedpoint import FixedPointPhase, Float64Phase, TWO_PI


class FixedPointTest(unittest.TestCase):
    def setUp(self):
        self.frames = [
            {"frame_index": i, "f0": 120.0 + i,
             "harmonic_amplitudes": [1.0] * 6, "energy_gain": 0.5}
            for i in range(10)
        ]
        self.fps = [f"fp{i}" for i in range(10)]

    def test_ring_modulo_is_integer(self):
        fp = FixedPointPhase(8, 1)
        phases = fp.accumulate(self.frames, 160, 16000, "u", self.fps, 6)
        self.assertTrue(np.all(phases >= 0))
        self.assertTrue(np.all(phases < fp.modulus))
        self.assertEqual(phases.dtype, np.int64)

    def test_bits_change_phase_and_grid(self):
        coarse = FixedPointPhase(6, 1)
        fine = FixedPointPhase(24, 1)
        p_coarse = coarse.accumulate(self.frames, 160, 16000, "u", self.fps, 6)
        p_fine = fine.accumulate(self.frames, 160, 16000, "u", self.fps, 6)
        self.assertFalse(np.array_equal(p_coarse, p_fine))
        # Coarse grid values must be exact multiples of the quantization step.
        step_model = FixedPointPhase(8, 4)
        p_step = step_model.accumulate(self.frames, 160, 16000, "u", self.fps, 6)
        self.assertTrue(np.all(p_step % 4 == 0))

    def test_deterministic_no_global_state(self):
        a = FixedPointPhase(20, 1).accumulate(self.frames, 160, 16000, "u", self.fps, 6)
        b = FixedPointPhase(20, 1).accumulate(self.frames, 160, 16000, "u", self.fps, 6)
        np.testing.assert_array_equal(a, b)

    def test_high_bits_approach_float64(self):
        fine = FixedPointPhase(40, 1)
        fl = Float64Phase()
        p_fixed = fine.to_radians(
            fine.accumulate(self.frames, 160, 16000, "u", self.fps, 6))
        p_float = fl.accumulate(self.frames, 160, 16000, "u", self.fps, 6)
        diff = np.abs(np.angle(np.exp(1j * (p_fixed - p_float))))
        self.assertLess(diff.max(), 1e-6)

    def test_quant_step_coarsens_phase(self):
        one = FixedPointPhase(10, 1).accumulate(self.frames, 160, 16000, "u", self.fps, 6)
        eight = FixedPointPhase(10, 8).accumulate(self.frames, 160, 16000, "u", self.fps, 6)
        self.assertGreater(np.abs(one - eight).sum(), 0)


if __name__ == "__main__":
    unittest.main()
