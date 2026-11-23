from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from voice_conversion.audio import read_wav
from voice_conversion.config import PipelineConfig
from voice_conversion.editor import Workspace
from voice_conversion.pipeline import run_pipeline
import generate_samples

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


class SampleDeterminismTests(unittest.TestCase):
    def test_generation_is_byte_reproducible(self):
        tmp = Path(tempfile.mkdtemp(prefix="vc-sample-"))
        try:
            generate_samples.main(str(tmp))
            pcm_a, rate_a = read_wav(tmp / "wav" / "source.wav")
            shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            generate_samples.main(str(tmp))
            pcm_b, rate_b = read_wav(tmp / "wav" / "source.wav")
            self.assertEqual(rate_a, 16000)
            self.assertEqual(rate_a, rate_b)
            self.assertTrue((pcm_a == pcm_b).all())
            self.assertEqual(
                (tmp / "exchange" / "segments.v1.txt").read_text("utf-8"),
                (SAMPLES / "exchange" / "segments.v1.txt").read_text("utf-8"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_wav_is_mono_16bit_16k(self):
        pcm, rate = read_wav(SAMPLES / "wav" / "source.wav")
        self.assertEqual(rate, 16000)
        self.assertEqual(pcm.dtype.kind, "i")
        self.assertGreater(len(pcm), 0)


class SnapshotResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vc-snap-"))
        self.config = PipelineConfig.load(SAMPLES / "config" / "acoustic.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_resume_reuses_products_without_rebuild(self):
        ws = Workspace(self.tmp / "ws", self.config)
        ws.reset_to(SAMPLES / "exchange" / "segments.v1.txt",
                    SAMPLES / "config" / "acoustic.json", "v1")
        common = dict(exchange_path=ws.exchange_path, version="v1",
                      source_wav=SAMPLES / "wav" / "source.wav",
                      config=self.config, snapshot_dir=self.tmp / "snap")
        first = run_pipeline(output_wav=self.tmp / "a.wav",
                             full_rebuild=True, **common)
        second = run_pipeline(output_wav=self.tmp / "b.wav",
                              full_rebuild=False, **common)
        self.assertEqual(first["output_fingerprint"],
                         second["output_fingerprint"])
        self.assertEqual(second["recomputed_segments"], 0)
        self.assertEqual(second["reused_segments"],
                         second["valid_segment_count"])
        self.assertEqual(
            [f["code"] for f in second["failures"]], [])

    def test_structural_config_change_forces_rebuild(self):
        ws = Workspace(self.tmp / "ws", self.config)
        ws.reset_to(SAMPLES / "exchange" / "segments.v1.txt",
                    SAMPLES / "config" / "acoustic.json", "v1")
        run_pipeline(exchange_path=ws.exchange_path, version="v1",
                     source_wav=SAMPLES / "wav" / "source.wav",
                     config=self.config, snapshot_dir=self.tmp / "snap",
                     output_wav=self.tmp / "a.wav", full_rebuild=True)
        changed = PipelineConfig.from_dict(
            {**self.config.to_dict(), "frame_shift_ms": 5.0})
        report = run_pipeline(
            exchange_path=ws.exchange_path, version="v1",
            source_wav=SAMPLES / "wav" / "source.wav", config=changed,
            snapshot_dir=self.tmp / "snap",
            output_wav=self.tmp / "b.wav", full_rebuild=False)
        self.assertIn("E_SNAPSHOT_MISMATCH",
                      [f["code"] for f in report["failures"]])
        self.assertEqual(report["recomputed_segments"],
                         report["valid_segment_count"])


if __name__ == "__main__":
    unittest.main()
