import dataclasses
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from diar.audio import read_wav, read_wav_bytes
from diar.cache import make_cache_key
from diar.clusters import run_attribution, run_clustering
from diar.codes import (
    E_AUDIO_FORMAT, E_AUDIO_TRUNCATED, E_NONFINITE_SAMPLE, E_SNAPSHOT_STALE,
    E_THRESHOLD_ZERO, PipelineError,
)
from diar.config import PipelineConfig
from diar.features import extract_features
from diar.injection import InjectionRegistry
from diar.pipeline import DiarizationPipeline
from diar.segments import segment_audio
from diar.synth import generate_dataset
from diar.util import quantize_vector


class DemoFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.mkdtemp(prefix="diar_test_")
        cls.input_dir = os.path.join(cls.workspace, "input")
        cls.out_dir = os.path.join(cls.workspace, "output")
        generate_dataset(cls.input_dir, n_files=6)
        cls.config = PipelineConfig.load(os.path.join(cls.input_dir, "config.json"))
        with open(os.path.join(cls.input_dir, "truth.json"), encoding="utf-8") as fh:
            cls.truth = json.load(fh)
        with open(os.path.join(cls.input_dir, "injections.json"), encoding="utf-8") as fh:
            cls.injections = json.load(fh)
        cls.registry = InjectionRegistry(cls.injections)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def _pipeline(self, out_dir=None, config=None, fresh=False, allow_edit=False):
        return DiarizationPipeline(
            self.input_dir, out_dir or self.out_dir,
            config or self.config, self.truth,
            InjectionRegistry(self.injections),
            allow_edit=allow_edit, fresh=fresh,
        )


class TestAudioAndFeatures(DemoFixture):
    def test_wav_header_validation(self):
        record = read_wav(os.path.join(self.input_dir, "mix_006.wav"))
        self.assertEqual(record["sample_rate"], 16000)
        self.assertEqual(record["channels"], 1)
        self.assertEqual(record["bits"], 16)
        self.assertEqual(record["data_bytes"], record["n_samples"] * 2)

    def test_bad_container_raises_format_code(self):
        with self.assertRaises(PipelineError) as ctx:
            read_wav_bytes(b"NOT_RIFF_DATA" * 4, "bad.wav")
        self.assertEqual(ctx.exception.code, E_AUDIO_FORMAT)
        self.assertEqual(ctx.exception.stage, "audio")

    def test_truncated_bytes_maps_to_truncated_code(self):
        path = os.path.join(self.input_dir, "mix_006.wav")
        with open(path, "rb") as handle:
            raw = handle.read()
        with self.assertRaises(PipelineError) as ctx:
            read_wav_bytes(raw[: len(raw) // 2], "cut.wav")
        self.assertEqual(ctx.exception.code, E_AUDIO_TRUNCATED)

    def test_frame_feature_shapes_and_quantization(self):
        record = read_wav(os.path.join(self.input_dir, "mix_006.wav"))
        product = extract_features(record, self.config)
        self.assertEqual(len(product["frames"]), product["n_frames"])
        for frame in product["frames"]:
            self.assertEqual(len(frame["cepstrum_q"]), self.config.feature_dim)
            self.assertEqual(len(frame["cepstrum"]), self.config.feature_dim)
            self.assertTrue(all(np.isfinite(frame["cepstrum"])))
        # 定点量化：整数 = round(浮点/步长)，残差可回放
        frame = product["frames"][0]
        ints, step, residual = quantize_vector(frame["cepstrum"], self.config.frac_bits)
        self.assertTrue(np.allclose(ints, frame["cepstrum_q"]))
        rebuilt = np.asarray(frame["cepstrum_q"]) * step + np.asarray(residual)
        self.assertTrue(np.allclose(rebuilt, frame["cepstrum"], atol=1e-12))

    def test_nonfinite_detected_at_feature_stage(self):
        record = read_wav(os.path.join(self.input_dir, "mix_006.wav"))
        registry = InjectionRegistry([{
            "injection_id": "INJ-T", "audio": record["audio"], "stage": "features",
            "fault_type": "nonfinite_samples",
        }])
        with self.assertRaises(PipelineError) as ctx:
            extract_features(record, self.config, registry, record["audio"])
        self.assertEqual(ctx.exception.code, E_NONFINITE_SAMPLE)
        self.assertEqual(ctx.exception.stage, "features")


class TestSegmentationClustering(DemoFixture):
    def setUp(self):
        self.record = read_wav(os.path.join(self.input_dir, "mix_006.wav"))
        self.features = extract_features(self.record, self.config)

    def test_segment_timing_and_overlap_flag(self):
        product = segment_audio(self.features, self.config)
        self.assertGreater(len(product["segments"]), 1)
        for segment in product["segments"]:
            self.assertLess(segment["start_frame"], segment["end_frame"])
            self.assertEqual(len(segment["frame_set"]),
                             segment["end_frame"] - segment["start_frame"])
        self.assertTrue(any(s["overlap_candidate"] for s in product["segments"]))

    def test_equal_threshold_is_non_overlap_boundary(self):
        # 构造一个使相邻段均值 L1 距离恰等于量化阈值的阈值，验证等距按非重叠决断
        product = segment_audio(self.features, self.config)
        means = product["segments"]
        from diar.util import l1_distance
        if len(means) >= 2:
            exact = l1_distance(means[0]["mean_cepstrum_q"],
                                means[1]["mean_cepstrum_q"])
            cfg = dataclasses.replace(
                self.config,
                cluster_distance_threshold=exact / (2 ** self.config.frac_bits),
            )
            tuned = segment_audio(self.features, cfg)
            self.assertEqual(tuned["boundary_records"][0]["decision"],
                             "non_overlap_equal_threshold")
            self.assertFalse(tuned["segments"][1]["overlap_candidate"])

    def test_clustering_tie_and_threshold_zero(self):
        segment_product = segment_audio(self.features, self.config)
        segment_product["_frames"] = self.features["frames"]
        cluster_product = run_clustering(segment_product, self.config)
        self.assertEqual(len(cluster_product["clusters"]),
                         self.truth["mix_006.wav"]["expected_speakers"])
        with self.assertRaises(PipelineError) as ctx:
            run_clustering(segment_product,
                           dataclasses.replace(self.config, cluster_distance_threshold=0.0))
        self.assertEqual(ctx.exception.code, E_THRESHOLD_ZERO)

    def test_quantization_gear_changes_integers_and_confidence(self):
        segment_product = segment_audio(self.features, self.config)
        segment_product["_frames"] = self.features["frames"]
        cluster_a = run_clustering(segment_product, self.config)
        gear_a = run_attribution(segment_product, cluster_a, self.config)
        cfg_b = dataclasses.replace(self.config, frac_bits=8)
        cluster_b = run_clustering(segment_product, cfg_b)
        gear_b = run_attribution(segment_product, cluster_b, cfg_b)
        interval_a = next(iter(gear_a["speaker_intervals"].values()))[0]
        interval_b = next(iter(gear_b["speaker_intervals"].values()))[0]
        self.assertNotEqual(interval_a["confidence_q"], interval_b["confidence_q"])
        self.assertNotEqual(interval_a["quant_step"], interval_b["quant_step"])
        # 档位变化通过单帧倒谱整数与置信度步长体现
        self.assertNotEqual(self.features["frames"][0]["cepstrum_q"],
                            extract_features(self.record, cfg_b)["frames"][0]["cepstrum_q"])
        self.assertEqual(interval_b["frac_bits"], 8)


class TestCacheAndSnapshot(DemoFixture):
    def test_cache_key_derivation_and_selective_invalidation(self):
        key_a = make_cache_key("features", "upstream", self.config)
        key_b = make_cache_key("features", "upstream",
                               dataclasses.replace(self.config, energy_threshold=0.05))
        # 特征键不依赖能量门限
        self.assertEqual(key_a, key_b)
        key_c = make_cache_key("segments", "upstream", self.config)
        key_d = make_cache_key("segments", "upstream",
                               dataclasses.replace(self.config, energy_threshold=0.05))
        self.assertNotEqual(key_c, key_d)
        key_e = make_cache_key("clusters", "upstream", self.config)
        key_f = make_cache_key("clusters", "upstream",
                               dataclasses.replace(self.config, cluster_distance_threshold=10.0))
        self.assertNotEqual(key_e, key_f)

    def test_stale_snapshot_rejected_then_accepted_with_edit_flag(self):
        out = os.path.join(self.workspace, "stale_out")
        self._pipeline(out, fresh=True).run()
        changed = dataclasses.replace(self.config, energy_threshold=0.04)
        with self.assertRaises(PipelineError) as ctx:
            self._pipeline(out, config=changed, allow_edit=False).run()
        self.assertEqual(ctx.exception.code, E_SNAPSHOT_STALE)
        report = self._pipeline(out, config=changed, allow_edit=True).run()
        self.assertIn("mix_006.wav", report["results"])

    def test_resume_reuses_everything(self):
        report_one = self._pipeline(fresh=True).run()
        self.assertGreater(report_one["stats"]["compute"], 0)
        report_two = self._pipeline().run()
        self.assertEqual(report_two["stats"]["compute"], 0)
        self.assertEqual(report_two["stats"]["reuse"], report_one["stats"]["compute"])


if __name__ == "__main__":
    unittest.main()
