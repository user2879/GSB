import os

from tests._base import BaseCase
from vdu import aggregation, calibration, error_track, evidence, features
from vdu import fixedpoint as fp
from vdu.pcmio import read_pcm


class ErrorTrackingTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.pcm = read_pcm(self._wav())

    def _stage(self, tier):
        windows = features.build_windows(
            self.pcm, self.transcript, self.config, tier)
        words = evidence.build_all_word_evidence(
            self.transcript, windows, self.config, tier)
        rounds = calibration.ordered_rounds(self.transcript)
        cals = calibration.calibrate(
            rounds, words, self.config, tier)
        summaries = aggregation.aggregate_segments(
            cals, words, self.transcript, self.config, tier)
        return windows, words, cals, summaries

    def test_four_stage_reports_and_per_segment_diffs(self):
        q_windows, q_words, q_cals, q_summary = self._stage(fp.COARSE)
        r_windows, r_words, r_cals, r_summary = self._stage(fp.REFERENCE)
        reports = [
            error_track.track_window(
                q_windows, r_windows, fp.COARSE,
                self.config["tolerance"]["window"]),
            error_track.track_words(
                q_words, r_words, fp.COARSE,
                self.config["tolerance"]["word"]),
            error_track.track_calibration(
                q_cals, r_cals,
                self.config["tolerance"]["calibration"]),
            error_track.track_segments(
                q_summary, r_summary,
                self.config["tolerance"]["segment"]),
        ]
        self.assertEqual([r["stage"] for r in reports],
                         ["window", "word", "calibration", "segment"])
        for report in reports:
            self.assertIn("per_segment_max_diff", report)
        first = error_track.first_overall_breach(reports)
        self.assertTrue(first is None or "stage" in first)

    def test_flip_attribution_returns_known_reason(self):
        c_windows, c_words, c_cals, c_summary = self._stage(fp.COARSE)
        f_windows, f_words, f_cals, f_summary = self._stage(fp.FINE)
        flips = error_track.flip_report(
            c_summary, f_summary, c_cals, f_cals, self.config,
            {"coarse": fp.COARSE, "fine": fp.FINE})
        for flip in flips:
            self.assertIn(flip["flip_reason"],
                          ("quantization_rounding", "decay_multiplication",
                           "threshold_comparison"))

    def _wav(self):
        return os.path.join(self.samples, "dialogue_demo.wav")
