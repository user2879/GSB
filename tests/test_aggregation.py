from tests._base import BaseCase
from vdu import aggregation, calibration, evidence, features, fixedpoint as fp
from vdu.pcmio import read_pcm


class AggregationTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.pcm = read_pcm(self._wav())
        self.windows = features.build_windows(
            self.pcm, self.transcript, self.config, fp.COARSE)
        self.words = evidence.build_all_word_evidence(
            self.transcript, self.windows, self.config, fp.COARSE)
        rounds = calibration.ordered_rounds(self.transcript)
        self.cals = calibration.calibrate(
            rounds, self.words, self.config, fp.COARSE)

    def test_threshold_equality_chooses_higher_band(self):
        tier = fp.COARSE
        threshold = tier.to_raw(0.0)
        decision = aggregation.classify(
            threshold, self.config["emotion_thresholds"], tier)
        self.assertTrue(decision["boundary"])
        self.assertEqual(decision["label"], "neutral")

    def test_half_step_marks_boundary(self):
        tier = fp.COARSE
        threshold = tier.to_raw(self.config["emotion_thresholds"]["positive"])
        decision = aggregation.classify(
            threshold + tier.half_step,
            self.config["emotion_thresholds"], tier)
        self.assertTrue(decision["boundary"])

    def test_overlap_arbitration_priority(self):
        summaries = aggregation.aggregate_segments(
            self.cals, self.words, self.transcript, self.config, fp.COARSE)
        # seg01 [0,7120) and seg02 [6000,13120) overlap; exactly one is owner.
        first_two = [s for s in summaries
                     if s["segment_id"] in ("seg01", "seg02")]
        suppressed = [s for s in first_two if s["suppressed"]]
        owners = [s for s in first_two if not s["suppressed"]]
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(len(owners), 1)
        self.assertTrue(suppressed[0]["suppression_reason"])
        self.assertEqual(suppressed[0]["covered_by"],
                         owners[0]["segment_id"])

    def test_references_recorded_per_segment(self):
        summaries = aggregation.aggregate_segments(
            self.cals, self.words, self.transcript, self.config, fp.COARSE)
        for summary in summaries:
            self.assertTrue(summary["word_keys"])
            self.assertTrue(summary["window_ids"])

    def _wav(self):
        import os
        return os.path.join(self.samples, "dialogue_demo.wav")
