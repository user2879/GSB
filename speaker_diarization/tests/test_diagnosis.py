import dataclasses
import json
import os
import shutil
import tempfile
import unittest

from diar.audio import read_wav
from diar.config import PipelineConfig
from diar.features import extract_features
from diar.injection import InjectionRegistry
from diar.pipeline import DiarizationPipeline
from diar.segments import segment_audio
from diar.synth import generate_dataset
from diar.verification import verify_audio


class TestCountDeviationDiagnosis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.mkdtemp(prefix="diar_diag_")
        cls.input_dir = os.path.join(cls.workspace, "input")
        generate_dataset(cls.input_dir, n_files=6)
        cls.config = PipelineConfig.load(os.path.join(cls.input_dir, "config.json"))
        with open(os.path.join(cls.input_dir, "truth.json"), encoding="utf-8") as fh:
            cls.truth = json.load(fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def _products(self, config, audio="mix_006.wav"):
        record = read_wav(os.path.join(self.input_dir, audio))
        features = extract_features(record, config)
        segments = segment_audio(features, config)
        segments["_frames"] = features["frames"]
        segments["_frame_lookup"] = {
            f["frame_index"]: {"start_ms": f["start_ms"], "end_ms": f["end_ms"]}
            for f in features["frames"]
        }
        from diar.clusters import run_attribution, run_clustering
        clusters = run_clustering(segments, config)
        attribution = run_attribution(segments, clusters, config)
        return features, segments, clusters, attribution

    def test_diagnosis_distinguishes_cluster_merge(self):
        # 极大聚类阈值导致所有段合并为一个簇 -> cluster_merge 诊断
        cfg = dataclasses.replace(self.config, cluster_distance_threshold=10.0)
        features, segments, clusters, attribution = self._products(cfg)
        truth_entry = self.truth["mix_006.wav"]
        report = verify_audio(
            truth_entry, attribution, segments, cfg, None, "mix_006.wav",
            segments["_frame_lookup"], features["n_frames"],
        )
        self.assertNotEqual(report["speaker_count_deviation"], 0)
        self.assertEqual(report["diagnosis"]["source"], "cluster_merge")
        self.assertIsInstance(report["diagnosis"]["first_divergence_frame"], int)

    def test_diagnosis_distinguishes_segmentation(self):
        # 极高能量门限切掉大量语音 -> 区间切分不足 -> segmentation 诊断
        cfg = dataclasses.replace(self.config, energy_threshold=0.9)
        features, segments, clusters, attribution = self._products(cfg)
        truth_entry = self.truth["mix_006.wav"]
        report = verify_audio(
            truth_entry, attribution, segments, cfg, None, "mix_006.wav",
            segments["_frame_lookup"], features["n_frames"],
        )
        if report["speaker_count_deviation"] != 0:
            self.assertEqual(report["diagnosis"]["source"], "segmentation")


class TestAudioDeletion(unittest.TestCase):
    def test_delete_audio_is_reflected_incrementally(self):
        workspace = tempfile.mkdtemp(prefix="diar_del_")
        input_dir = os.path.join(workspace, "input")
        out_dir = os.path.join(workspace, "output")
        generate_dataset(input_dir, n_files=6)
        cfg = PipelineConfig.load(os.path.join(input_dir, "config.json"))
        with open(os.path.join(input_dir, "truth.json"), encoding="utf-8") as fh:
            truth = json.load(fh)
        with open(os.path.join(input_dir, "injections.json"), encoding="utf-8") as fh:
            injections = json.load(fh)
        pipeline = DiarizationPipeline(input_dir, out_dir, cfg, truth,
                                       InjectionRegistry(injections), fresh=True)
        first = pipeline.run()
        self.assertIn("mix_006.wav", first["results"])

        # 删除一个干净音频后，授权编辑重跑应不再包含该音频
        os.remove(os.path.join(input_dir, "mix_006.wav"))
        pipeline2 = DiarizationPipeline(input_dir, out_dir, cfg, truth,
                                        InjectionRegistry(injections), allow_edit=True)
        second = pipeline2.run()
        self.assertNotIn("mix_006.wav", second["results"])
        self.assertNotIn("mix_006.wav", second["audio_order"])
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
