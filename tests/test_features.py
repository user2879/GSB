from tests._base import BaseCase
from vdu import features, fixedpoint as fp
from vdu.pcmio import read_pcm


class FeatureTests(BaseCase):
    def test_windows_repeat_byte_identical(self):
        pcm = read_pcm(self._wav())
        first = features.build_windows(pcm, self.transcript, self.config,
                                       fp.COARSE)
        second = features.build_windows(pcm, self.transcript, self.config,
                                        fp.COARSE)
        self.assertEqual([w["features"] for w in first],
                         [w["features"] for w in second])

    def test_cross_boundary_window_records_segments_and_frames(self):
        pcm = read_pcm(self._wav())
        windows = features.build_windows(pcm, self.transcript, self.config,
                                         fp.COARSE)
        spanning = [w for w in windows if len(w["segment_refs"]) > 1]
        self.assertTrue(spanning)
        for window in spanning:
            # Every referenced segment has at least one valid frame and the
            # total per-segment count covers every frame at least once.
            self.assertTrue(all(count >= 1
                                for count in window["segment_refs"].values()))
            self.assertGreaterEqual(
                sum(window["segment_refs"].values()),
                len(window["frame_indices"]))
            # Explicitly recompute from frame geometry.
            for seg_id, count in window["segment_refs"].items():
                seg = next(s for s in self.transcript["segments"]
                           if s["segment_id"] == seg_id)
                manual = 0
                for frame_idx in window["frame_indices"]:
                    f_start = frame_idx * self.config["frame_hop"]
                    f_end = f_start + self.config["frame_length"]
                    if f_start < seg["end"] and seg["start"] < f_end:
                        manual += 1
                self.assertEqual(manual, count)

    def test_dimension_order_and_width(self):
        pcm = read_pcm(self._wav())
        windows = features.build_windows(pcm, self.transcript, self.config,
                                         fp.COARSE)
        for window in windows:
            self.assertEqual(len(window["features"]), features.FEATURE_DIMS)
            bound = 1 << (fp.COARSE.total_bits - 1)
            for value in window["features"]:
                self.assertLess(abs(value), bound)

    def test_tier_change_changes_integers(self):
        pcm = read_pcm(self._wav())
        coarse = features.build_windows(pcm, self.transcript, self.config,
                                        fp.COARSE)
        fine = features.build_windows(pcm, self.transcript, self.config,
                                      fp.FINE)
        same = all(a["features"] == b["features"]
                   for a, b in zip(coarse, fine))
        self.assertFalse(same)

    def _wav(self):
        import os
        return os.path.join(self.samples, "dialogue_demo.wav")
