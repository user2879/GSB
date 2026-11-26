from tests._base import BaseCase
from vdu import error_track, fixedpoint as fp


class FlipAttributionTests(BaseCase):
    def _summary(self, seg_id, label, confidence):
        return {"segment_id": seg_id, "label": label,
                "confidence": confidence, "boundary": False}

    def _cals(self, evidence_value):
        return [{"word_key": "seg01:0", "segment_id": "seg01",
                 "calibration": evidence_value,
                 "evidence_value": evidence_value}]

    def test_threshold_comparison_attribution(self):
        tier = fp.COARSE
        near = tier.to_raw(
            self.config["emotion_thresholds"]["positive"])
        coarse = self._summary("seg01", "neutral", near)
        fine = self._summary("seg01", "positive",
                             near + fp.FINE.half_step)
        flip = error_track.attribute_flip(
            coarse, fine, self._cals(near),
            self._cals(near + 1),
            self.config["emotion_thresholds"],
            (fp.COARSE, fp.FINE))
        self.assertEqual(flip["flip_reason"], "threshold_comparison")

    def test_quantization_rounding_attribution(self):
        tier = fp.COARSE
        coarse = self._summary("seg01", "neutral", tier.to_raw(0.3))
        fine = self._summary("seg01", "positive", fp.FINE.to_raw(0.31))
        flip = error_track.attribute_flip(
            coarse, fine, self._cals(tier.to_raw(0.3)),
            self._cals(fp.FINE.to_raw(0.31)),
            self.config["emotion_thresholds"],
            (fp.COARSE, fp.FINE))
        self.assertIn(flip["flip_reason"],
                      ("quantization_rounding", "decay_multiplication",
                       "threshold_comparison"))
        self.assertEqual(flip["coarse_label"], "neutral")
        self.assertEqual(flip["fine_label"], "positive")

    def test_no_flip_returns_empty(self):
        tier = fp.COARSE
        summary = self._summary("seg01", "positive", tier.to_raw(0.9))
        flips = error_track.flip_report(
            [summary], [summary], self._cals(tier.to_raw(0.9)),
            self._cals(tier.to_raw(0.9)), self.config,
            {"coarse": fp.COARSE, "fine": fp.FINE})
        self.assertEqual(flips, [])
