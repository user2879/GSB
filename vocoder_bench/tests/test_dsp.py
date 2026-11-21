import math
import unittest

import numpy as np

from vocoder.dsp import (
    ENERGY_SCALE, make_window, normalized_cross_correlation, overlap_add,
    spectral_distance, synthesize_frame, waveform_rmse,
)


class DspTest(unittest.TestCase):
    def test_window_shapes(self):
        for name in ("hann", "hamming", "rect"):
            window = make_window(name, 64)
            self.assertEqual(window.shape, (64,))
            self.assertTrue(np.all(np.isfinite(window)))

    def test_ascending_harmonic_accumulation(self):
        amps = np.array([0.5, 0.25, 0.125])
        phases = np.array([0.1, 0.2, 0.3])
        frame = synthesize_frame(amps, phases, 100.0, 0.7, 32, 16000)
        self.assertEqual(frame.shape, (32,))
        t = np.arange(32)
        expected = np.zeros(32)
        for h in range(3):
            expected += amps[h] * np.sin(phases[h] + 2 * math.pi * (h + 1) * 100 * t / 16000)
        np.testing.assert_allclose(frame, expected * 0.7)

    def test_ola_cola_normalization(self):
        rng = np.random.default_rng(0)
        frames = [rng.standard_normal(32) for _ in range(8)]
        out_cola = overlap_add(frames, 16, "hann", "cola")
        out_none = overlap_add(frames, 16, "hann", "none")
        self.assertEqual(out_cola.shape, out_none.shape)
        self.assertTrue(np.all(np.isfinite(out_cola)))
        self.assertLess(np.abs(out_cola).max(), 5.0)

    def test_alignment_tie_prefers_small_offset(self):
        ref = np.sin(np.linspace(0, 20, 400))
        lag, corr = normalized_cross_correlation(ref, ref.copy(), 5.0, 16000)
        self.assertEqual(lag, 0)
        self.assertGreater(corr, 0.999)

    def test_metrics_finite_and_ordered(self):
        ref = np.sin(np.linspace(0, 30, 800)) * 1000
        cand = ref + np.random.default_rng(1).standard_normal(800) * 50
        self.assertTrue(np.isfinite(spectral_distance(ref, cand)))
        self.assertTrue(np.isfinite(waveform_rmse(ref, cand)))
        _, ncc = normalized_cross_correlation(ref, cand, 5.0, 16000)
        self.assertGreater(ncc, 0.9)

    def test_energy_scale_saturates_int16(self):
        from vocoder.wav_io import pack_pcm16
        pcm = pack_pcm16(np.array([-40000.0, 40000.0, 0.6]), "half_up")
        self.assertEqual(pcm.tolist(), [-32768, 32767, 1])
        pcm_even = pack_pcm16(np.array([0.5, 1.5, 2.5]), "half_even")
        self.assertEqual(pcm_even.tolist(), [0, 2, 2])
        pcm_trunc = pack_pcm16(np.array([0.9, -0.9]), "trunc")
        self.assertEqual(pcm_trunc.tolist(), [0, 0])
        # The three deterministic modes must be mutually distinguishable.
        ties = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5])
        outputs = {mode: pack_pcm16(ties, mode).tolist()
                   for mode in ("half_up", "half_even", "trunc")}
        self.assertEqual(len({tuple(v) for v in outputs.values()}), 3)


if __name__ == "__main__":
    unittest.main()
