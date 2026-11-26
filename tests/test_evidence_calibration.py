from tests._base import BaseCase
from vdu import calibration, evidence, features, fixedpoint as fp
from vdu.pcmio import read_pcm


class EvidenceCalibrationTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.pcm = read_pcm(self._wav())
        self.windows = features.build_windows(
            self.pcm, self.transcript, self.config, fp.COARSE)
        self.words = evidence.build_all_word_evidence(
            self.transcript, self.windows, self.config, fp.COARSE)

    def test_every_word_maps_to_windows_and_segment(self):
        word_count = sum(len(s["words"])
                         for s in self.transcript["segments"])
        self.assertEqual(len(self.words), word_count)
        for record in self.words:
            self.assertTrue(record["window_ids"])
            self.assertTrue(all(c >= 1
                                for c in record["coverage_frames"].values()))

    def test_semantic_cooccurrence_integer_and_deterministic(self):
        sequence = evidence.global_word_sequence(self.transcript)
        first = evidence.semantic_counts(sequence, 3, self.config)
        second = evidence.semantic_counts(sequence, 3, self.config)
        self.assertEqual(first, second)
        self.assertEqual(sum(first["counts"].values()), first["total"])
        self.assertIsInstance(first["total"], int)

    def test_residuals_change_with_tier(self):
        fine_windows = features.build_windows(
            self.pcm, self.transcript, self.config, fp.FINE)
        fine_words = evidence.build_all_word_evidence(
            self.transcript, fine_windows, self.config, fp.FINE)
        coarse_raw = [value for record in self.words
                      for value in record["strength"]]
        fine_raw = [value for record in fine_words
                    for value in record["strength"]]
        self.assertNotEqual(coarse_raw, fine_raw)
        self.assertTrue(any(any(record["residual"])
                            for record in fine_words) or
                        any(any(record["residual"])
                            for record in self.words))

    def test_fixedpoint_requantize_residual(self):
        src = fp.COARSE
        raw = src.to_raw(1.0) + 1
        new_raw, residual = fp.requantize(raw, src, fp.FINE)
        self.assertEqual(residual, 0)
        back, back_residual = fp.requantize(new_raw, fp.FINE, src)
        self.assertEqual(back, raw)
        self.assertGreaterEqual(back_residual, 0)

    def test_calibration_uses_integer_decay_and_inheritance(self):
        # Force an empty round by zeroing one word's strength.
        self.words[2]["strength"] = [0] * features.FEATURE_DIMS
        rounds = calibration.ordered_rounds(self.transcript)
        cals = calibration.calibrate(rounds, self.words, self.config,
                                     fp.COARSE)
        empty = cals[2]
        self.assertTrue(empty["inherited"])
        self.assertEqual(empty["gap"], 1)
        for item in cals:
            self.assertIsInstance(item["calibration"], int)

    def test_calibration_deterministic(self):
        rounds = calibration.ordered_rounds(self.transcript)
        a = calibration.calibrate(rounds, self.words, self.config, fp.COARSE)
        b = calibration.calibrate(rounds, self.words, self.config, fp.COARSE)
        self.assertEqual(a, b)

    def test_decay_never_uses_float(self):
        tier = fp.COARSE
        value = tier.to_raw(10.0)
        decayed = calibration.decay_once(value, 7, 8, tier)
        self.assertIsInstance(decayed, int)
        self.assertLessEqual(decayed, value)

    def _wav(self):
        import os
        return os.path.join(self.samples, "dialogue_demo.wav")
