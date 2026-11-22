#!/usr/bin/env python3
"""Unit tests for deterministic primitives of the diarization component."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diarization import clustering, embedding, segmentation, tracking


class TestClustering(unittest.TestCase):
    def test_threshold_equality_is_same_speaker(self):
        # similarity of identical vectors is exactly 1.0; with threshold 1.0
        # the equality rule must assign the second window to the same cluster
        embs = {0: np.array([1.0, 0.0]), 1: np.array([1.0, 0.0])}
        res = clustering.cluster_windows(embs, 1.0, 0.1)
        self.assertEqual(res["assignments"][0], res["assignments"][1])
        self.assertEqual(res["unassigned"], [])

    def test_hysteresis_band_stays_unassigned(self):
        embs = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.97, 0.242535625]),  # cos ~ 0.97 -> inside band
        }
        res = clustering.cluster_windows(embs, 0.99, 0.05)
        self.assertEqual(res["unassigned"], [1])
        self.assertEqual(len(res["hysteresis_events"]), 1)
        self.assertNotIn(1, res["assignments"])

    def test_below_band_creates_new_cluster(self):
        embs = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
        res = clustering.cluster_windows(embs, 0.99, 0.05)
        self.assertNotEqual(res["assignments"][0], res["assignments"][1])

    def test_first_match_in_lexicographic_order(self):
        embs = {
            0: np.array([1.0, 0.0]),
            1: np.array([1.0, 0.0]),
            2: np.array([1.0, 0.0]),
        }
        res = clustering.cluster_windows(embs, 0.5, 0.05)
        self.assertEqual(res["assignments"][2], "SPK_000")

    def test_merge_gate_boundary_event(self):
        embs = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([0.97, 0.242535625]),
        }
        # w2 is in the hysteresis band of SPK_000 -> unassigned; the two
        # cluster representatives are orthogonal -> no merge events
        res = clustering.cluster_windows(embs, 0.99, 0.05)
        self.assertEqual(res["merge_events"], [])
        self.assertEqual(res["unassigned"], [2])


class TestQuantization(unittest.TestCase):
    def test_deterministic_bit_identical(self):
        rng = np.random.RandomState(42)
        emb = rng.normal(size=16)
        emb = emb / np.linalg.norm(emb)
        q1 = embedding.quantize(emb, 8, 65536)
        q2 = embedding.quantize(emb, 8, 65536)
        self.assertTrue(np.array_equal(q1["quantized"], q2["quantized"]))
        self.assertTrue(np.array_equal(q1["residual"], q2["residual"]))

    def test_step_and_residual_bounds(self):
        emb = np.array([0.123456, -0.987654])
        q = embedding.quantize(emb, 8, 65536)
        self.assertAlmostEqual(q["step"], 2.0 ** -8)
        self.assertTrue(np.all(np.abs(q["residual"]) <= q["step"] / 2 + 1e-18))

    def test_higher_width_reduces_residual(self):
        rng = np.random.RandomState(7)
        emb = rng.normal(size=16)
        r_low = np.abs(embedding.quantize(emb, 6, 65536)["residual"]).max()
        r_high = np.abs(embedding.quantize(emb, 10, 65536)["residual"]).max()
        self.assertLess(r_high, r_low)


class TestSegmentation(unittest.TestCase):
    def _config(self):
        return {
            "frame_ms": 25, "frame_hop_ms": 10,
            "min_segment_ms": 300, "min_silence_ms": 250,
            "boundary_rule": "midpoint",
        }

    def _windows(self, n_samples, sr=16000):
        from diarization import audio_io
        cfg = {"window_ms": 400, "hop_ms": 200, "last_window_policy": "pad"}
        return audio_io.make_windows(n_samples, sr, cfg)

    def test_empty_when_no_speech(self):
        cfg = self._config()
        n = 16000 * 2
        windows = self._windows(n)
        energy = np.zeros(200)
        segs = segmentation.build_segments({}, windows, energy, n, 16000, cfg)
        self.assertEqual(segs, [])

    def test_short_segment_dropped(self):
        cfg = self._config()
        n = 16000 * 2
        windows = self._windows(n)
        # assign only the first window -> ~0.4 s speech? one window = 400 ms
        # min_segment is 300 ms so a single window survives; use a tiny
        # assignment gap scenario instead: no window -> empty
        segs = segmentation.build_segments({}, windows, np.zeros(200), n, 16000, cfg)
        self.assertEqual(segs, [])

    def test_same_speaker_short_gap_merged(self):
        cfg = self._config()
        sr = 16000
        n = sr * 4
        windows = self._windows(n, sr)
        assignments = {}
        for w in windows:
            t = w["start_sample"] / sr
            # speech 0-1s and 1.2-3s with a 0.2s gap (< min_silence 250ms)
            if t < 1.0 or 1.2 <= t < 3.0:
                assignments[w["window_index"]] = "SPK_000"
        energy = np.ones(1 + (n - 400) // 160)
        segs = segmentation.build_segments(assignments, windows, energy, n, sr, cfg)
        speakers = [s["speaker"] for s in segs]
        self.assertEqual(speakers, ["SPK_000"])
        self.assertLess(segs[0]["end_ms"] - segs[0]["start_ms"], 3400.0)


class TestTracking(unittest.TestCase):
    def _config(self):
        return {"tolerance": 1e-6, "similarity_threshold": 0.9,
                "merge_hysteresis": 0.05}

    def _side(self, deq_diff, sim, assign, segments, margins, merges=None):
        return {
            "dequantized": {i: np.array([d]) for i, d in enumerate(deq_diff)},
            "similarities": sim,
            "assignments": assign,
            "segments": segments,
            "decision_margins": margins,
            "merge_events": merges or [],
            "merge_boundary_events": [],
        }

    def test_no_divergence_no_cause(self):
        cfg = self._config()
        side = self._side([0.1], {0: 0.5}, {0: "SPK_000"},
                          [{"start_frame": 0, "end_frame": 10}], {0: 0.5})
        out = tracking.compare_widths(side, side, cfg)
        self.assertFalse(out["flip"]["diverged"])
        self.assertIsNone(out["flip"]["cause"])

    def test_embedding_residual_cause(self):
        cfg = self._config()
        low = self._side([0.1], {0: 0.5}, {0: "SPK_000"},
                         [{"start_frame": 0, "end_frame": 10}], {0: 0.5})
        high = self._side([0.2], {0: 0.5}, {0: "SPK_000"},
                          [{"start_frame": 0, "end_frame": 11}], {0: 0.5})
        out = tracking.compare_widths(low, high, cfg)
        self.assertTrue(out["flip"]["diverged"])
        self.assertEqual(out["flip"]["cause"], "embedding_quantization_residual")
        self.assertEqual(out["first_exceed_stage"], "embedding_quantization")

    def test_similarity_rounding_cause(self):
        cfg = self._config()
        low = self._side([0.1], {0: 0.5}, {0: "SPK_000"},
                         [{"start_frame": 0, "end_frame": 10}], {0: 0.5})
        high = self._side([0.1], {0: 0.500001}, {0: "SPK_000"},
                          [{"start_frame": 0, "end_frame": 11}], {0: 0.5})
        out = tracking.compare_widths(low, high, cfg)
        self.assertEqual(out["flip"]["cause"], "similarity_rounding")

    def test_threshold_neighborhood_cause(self):
        cfg = self._config()
        low = self._side([0.1], {0: 0.5}, {0: "SPK_000", 1: "SPK_000"},
                         [{"start_frame": 0, "end_frame": 10}],
                         {0: 0.5, 1: 0.91})
        high = self._side([0.1], {0: 0.5}, {0: "SPK_000"},
                          [{"start_frame": 0, "end_frame": 12}],
                          {0: 0.5, 1: 0.89})
        out = tracking.compare_widths(low, high, cfg)
        self.assertEqual(out["flip"]["cause"], "threshold_neighborhood_decision")

    def test_merge_gate_cause(self):
        cfg = self._config()
        low = self._side([0.1], {0: 0.5}, {0: "SPK_000"},
                         [{"start_frame": 0, "end_frame": 10}], {0: 0.5},
                         merges=[{"from": "SPK_001", "into": "SPK_000",
                                  "similarity": 0.95}])
        high = self._side([0.1], {0: 0.5}, {0: "SPK_000"},
                          [{"start_frame": 0, "end_frame": 11}], {0: 0.5})
        out = tracking.compare_widths(low, high, cfg)
        self.assertEqual(out["flip"]["cause"], "merge_gate_hysteresis_boundary")


if __name__ == "__main__":
    unittest.main()
