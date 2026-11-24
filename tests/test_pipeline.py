import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diariz.audioio import read_wav, truncate_wav, write_wav
from diariz.cache import CacheStore
from diariz.clustering import incrementally_cluster
from diariz.config import (ALLOWED_FRAC_BITS, ALLOWED_WORD_BITS,
                          default_config, parse_config)
from diariz.errorprop import higher_tier_config, propagate
from diariz.faults import run_fault_injections
from diariz.features import plan_windows
from diariz.pipeline import process_audio
from diariz.quantize import (Q16_ONE, integer_similarity, q16_to_float,
                             quantize_embedding, threshold_to_q16)
from diariz.snapshot import SnapshotManager
from diariz.synth import SynthSpec, synthesize

SAMPLE_WAV = os.path.join(os.path.dirname(__file__), "..", "samples",
                          "synthetic_3spk.wav")


class BaseCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = tempfile.mkdtemp(prefix="diariz-test-")
        cls.cfg = default_config()
        if not os.path.exists(SAMPLE_WAV):
            os.makedirs(os.path.dirname(SAMPLE_WAV), exist_ok=True)
            synthesize(SynthSpec(duration_s=4.0, speaker_count=2, seed=7),
                       SAMPLE_WAV)
        cls.wave = read_wav(SAMPLE_WAV)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_root, ignore_errors=True)

    def cache(self, name):
        return CacheStore(os.path.join(self.temp_root, name))


class TestAudioConfigValidation(BaseCase):
    def test_wav_header_and_format(self):
        data = read_wav(SAMPLE_WAV)
        self.assertEqual(data.sample_rate, 16000)
        self.assertEqual(data.channels, 1)
        self.assertEqual(data.bits_per_sample, 16)
        self.assertEqual(data.data_bytes, data.num_samples * 2)

    def test_truncated_file_raises_format_code(self):
        bad = os.path.join(self.temp_root, "trunc.wav")
        write_wav(bad, self.wave.samples)
        truncate_wav(bad, os.path.getsize(bad) - 128)
        from diariz.errors import AudioFormatError
        with self.assertRaises(AudioFormatError) as ctx:
            read_wav(bad)
        self.assertEqual(ctx.exception.code, "E_AUDIO_FORMAT")

    def test_config_validation_codes(self):
        from diariz.errors import ConfigInvalidError
        bad_cases = [
            {"window_ms": 0}, {"hop_ms": 300}, {"embed_dim": 5},
            {"threshold": 1.5}, {"word_bits": 12}, {"frac_bits": 20},
            {"concurrency": 0}, {"concurrency": 99},
        ]
        for override in bad_cases:
            raw = default_config().to_dict()
            raw.update(override)
            with self.subTest(override=override):
                with self.assertRaises(ConfigInvalidError) as ctx:
                    parse_config(raw)
                self.assertEqual(ctx.exception.code, "E_CONFIG_INVALID")

    def test_enumerations_are_allowed(self):
        self.assertEqual(tuple(sorted(ALLOWED_WORD_BITS)), (8, 16, 32))
        self.assertIn(default_config().frac_bits, ALLOWED_FRAC_BITS)


class TestWindowingEmbedding(BaseCase):
    def test_deterministic_offsets_and_embeddings(self):
        r1 = process_audio(SAMPLE_WAV, self.cfg, self.temp_root,
                           cache=self.cache("det1"))
        r2 = process_audio(SAMPLE_WAV, self.cfg, self.temp_root,
                           cache=self.cache("det2"))
        offsets1 = [(w.start_sample, w.end_sample) for w in r1["windows"]]
        offsets2 = [(w.start_sample, w.end_sample) for w in r2["windows"]]
        self.assertEqual(offsets1, offsets2)
        for a, b in zip(r1["embeddings"], r2["embeddings"]):
            np.testing.assert_array_equal(a.embedding, b.embedding)
            self.assertEqual(a.accumulation_order,
                             tuple(sorted(a.accumulation_order)))

    def test_overlap_relations_recorded(self):
        windows = plan_windows(len(self.wave.samples), self.cfg)
        self.assertTrue(windows)
        self.assertGreater(len(windows[0].overlaps_with), 0)
        for window in windows:
            self.assertEqual(window.frame_indices,
                             tuple(sorted(window.frame_indices)))


class TestQuantization(BaseCase):
    def _embedding(self):
        result = process_audio(SAMPLE_WAV, self.cfg, self.temp_root,
                               cache=self.cache("quant"))
        return result["embeddings"][0]

    def test_width_change_alters_integers_and_residuals(self):
        embedding = self._embedding()
        low = quantize_embedding(0, embedding.embedding,
                                 self.cfg.with_quant(16, 10), "k-low")
        high = quantize_embedding(0, embedding.embedding,
                                  self.cfg.with_quant(32, 14), "k-high")
        self.assertNotEqual(low.integers, high.integers)
        self.assertNotEqual(low.residuals, high.residuals)
        self.assertGreaterEqual(low.max_residual_abs, 0.0)
        # residual = float - dequantized
        for float_value, integer, residual in zip(
                embedding.embedding, low.integers, low.residuals):
            self.assertAlmostEqual(
                residual, float_value - integer * low.step, places=12)

    def test_integer_similarity_range_and_threshold(self):
        result = process_audio(SAMPLE_WAV, self.cfg, self.temp_root,
                               cache=self.cache("sim"))
        q = result["quantized"]
        sim = integer_similarity(q[0].integers, q[1].integers)
        self.assertGreaterEqual(sim, -Q16_ONE)
        self.assertLessEqual(sim, Q16_ONE)
        threshold = threshold_to_q16(self.cfg.threshold)
        self.assertAlmostEqual(q16_to_float(threshold), self.cfg.threshold,
                               delta=1.0 / Q16_ONE)


class TestClustering(BaseCase):
    def test_deterministic_assignments_and_lexicographic_ids(self):
        r1 = process_audio(SAMPLE_WAV, self.cfg, self.temp_root,
                           cache=self.cache("cl1"))
        r2 = process_audio(SAMPLE_WAV, self.cfg, self.temp_root,
                           cache=self.cache("cl2"))
        self.assertEqual(r1["clustering"].assignments,
                         r2["clustering"].assignments)
        ids = list(r1["clustering"].clusters)
        self.assertEqual(ids, sorted(ids))
        for comparison in r1["clustering"].comparisons:
            self.assertEqual(comparison.accepted,
                             comparison.similarity_q16 >= comparison.threshold_q16)

    def test_equality_threshold_is_boundary(self):
        # Build synthetic integer vectors whose similarity lands exactly on a
        # chosen threshold to exercise the equality rule.
        cfg = self.cfg
        # Integer vectors whose Q16 cosine similarity equals the threshold
        # exactly: (4,3,-3,-4) vs (5,4,-3,-1) -> 58982 == threshold_to_q16(0.9).
        from diariz.quantize import QuantizedWindow
        dim = cfg.embed_dim
        a = (4, 3, -3, -4) + (0,) * (dim - 4)
        b = (5, 4, -3, -1) + (0,) * (dim - 4)
        zeros = (0.0,) * dim
        windows = [
            QuantizedWindow(0, cfg.word_bits, cfg.frac_bits,
                            2 ** -cfg.frac_bits, a, zeros, a, 0, 0.0, "q0"),
            QuantizedWindow(1, cfg.word_bits, cfg.frac_bits,
                            2 ** -cfg.frac_bits, b, zeros, b, 0, 0.0, "q1"),
        ]
        self.assertEqual(integer_similarity(a, b),
                         threshold_to_q16(cfg.threshold))
        result = incrementally_cluster(windows, cfg, {0: "q0", 1: "q1"})
        self.assertEqual(len(result.clusters), 1)
        self.assertIn(1, result.boundary_windows)


class TestSegmentation(BaseCase):
    def test_segment_fields_and_arbitration(self):
        result = process_audio(SAMPLE_WAV, self.cfg, self.temp_root,
                               cache=self.cache("seg"))
        segments = result["segmentation"].segments
        self.assertTrue(segments)
        for segment in segments:
            self.assertGreaterEqual(segment.end_ms, segment.start_ms)
            self.assertTrue(segment.arbitration_windows)
            self.assertTrue(segment.boundary_basis)
            self.assertGreaterEqual(segment.avg_similarity, 0.0)
        votes = result["segmentation"].frame_votes
        for vote in votes:
            self.assertIn(vote.winner_window, vote.participating_windows)


class TestCacheInvalidation(BaseCase):
    def test_single_window_replacement_precision(self):
        cache = self.cache("invalidate")
        base = process_audio(SAMPLE_WAV, self.cfg, self.temp_root, cache=cache)
        windows = plan_windows(len(self.wave.samples), self.cfg)
        target = len(windows) // 2
        length = windows[target].end_sample - windows[target].start_sample
        replacement = np.random.default_rng(3).normal(0.0, 0.3, length)
        changed = process_audio(
            SAMPLE_WAV, self.cfg, self.temp_root, cache=cache,
            window_audio_overrides={target: replacement})
        reused = changed["reuse_report"]["reused"]["embed"]
        recomputed = changed["reuse_report"]["recomputed"]["embed"]
        # Only the replaced content and its overlapping duplicate entries move.
        self.assertGreater(len(reused), 0)
        self.assertLess(len(recomputed), base["window_count"])
        self.assertGreater(len(changed["reuse_report"]["recomputed"]["compare"]),
                           0)

    def test_threshold_change_reuses_embeddings(self):
        cache = self.cache("threshold")
        process_audio(SAMPLE_WAV, self.cfg, self.temp_root, cache=cache)
        changed_cfg = self.cfg.__class__(
            **{**self.cfg.to_dict(), "threshold": 0.8})
        changed = process_audio(SAMPLE_WAV, changed_cfg, self.temp_root,
                                cache=cache)
        self.assertEqual(
            len(changed["reuse_report"]["reused"]["embed"]),
            changed["window_count"])
        self.assertEqual(
            len(changed["reuse_report"]["reused"]["quant"]),
            changed["window_count"])
        self.assertGreater(
            len(changed["reuse_report"]["recomputed"]["compare"]), 0)


class TestErrorPropagation(BaseCase):
    def test_two_tier_diff_and_attribution(self):
        base = process_audio(SAMPLE_WAV, self.cfg, self.temp_root,
                             cache=self.cache("ep-base"))
        high_cfg = higher_tier_config(self.cfg)
        high = process_audio(SAMPLE_WAV, high_cfg, self.temp_root,
                             cache=self.cache("ep-high"))
        diffs, attribution = propagate(
            {"quantized": base["quantized"],
             "clustering": base["clustering"],
             "segmentation": base["segmentation"]},
            {"quantized": high["quantized"],
             "clustering": high["clustering"],
             "segmentation": high["segmentation"]},
            self.cfg)
        self.assertEqual(len(diffs), base["window_count"])
        self.assertIn("attributed_reasons", attribution)
        self.assertGreaterEqual(
            attribution["criteria"]["threshold_gap_q16_neutral_band"], 0)
        valid_stages = {"within_tolerance", "quantization", "similarity",
                        "clustering", "segmentation"}
        self.assertTrue({d.first_exceed_stage for d in diffs} <= valid_stages)


class TestSnapshotAndFaults(BaseCase):
    def test_snapshot_version_and_conflict(self):
        manager = SnapshotManager(os.path.join(self.temp_root, "snap"))
        manager.begin()
        manager.stage_artifact("embed", "a.json", {"k": 1})
        manifest = manager.commit("cfg", ["audio"], {"embed": 1})
        self.assertEqual(manifest["global_version"], 1)
        from diariz.errors import SnapshotConflictError
        manager.begin()
        with self.assertRaises(SnapshotConflictError):
            manager.commit("cfg", ["audio"], {"embed": 2},
                           expected_version=99)

    def test_fault_injection_isolation(self):
        report = run_fault_injections(
            SAMPLE_WAV, self.wave.samples, self.cfg, self.temp_root)
        hits = {item["id"]: item for item in report}
        self.assertTrue(all(item["hit"] for item in report))
        self.assertEqual(hits["INJ-01"]["error_code"], "E_AUDIO_FORMAT")
        self.assertEqual(hits["INJ-03"]["error_code"], "E_CONFIG_INVALID")
        self.assertEqual(hits["INJ-04"]["error_code"], "E_AUDIO_FORMAT")
        # Silence remains a processable legal input.
        self.assertEqual(hits["INJ-02"]["error_code"], "")


class TestConcurrency(BaseCase):
    def test_serial_vs_concurrent_equivalence(self):
        serial = process_audio(
            SAMPLE_WAV, self.cfg.__class__(
                **{**self.cfg.to_dict(), "concurrency": 1}),
            self.temp_root, cache=self.cache("conc-serial"))
        parallel = process_audio(
            SAMPLE_WAV, self.cfg, self.temp_root,
            cache=self.cache("conc-parallel"))
        self.assertEqual(serial["clustering"].assignments,
                         parallel["clustering"].assignments)
        self.assertEqual(
            [as_seg(s) for s in serial["segmentation"].segments],
            [as_seg(s) for s in parallel["segmentation"].segments])
        self.assertEqual(
            serial["traces"]["embed"]["budget_deduction_order"],
            parallel["traces"]["embed"]["budget_deduction_order"])


def as_seg(segment):
    return (segment.speaker, segment.start_ms, segment.end_ms,
            segment.avg_similarity, segment.arbitration_windows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
