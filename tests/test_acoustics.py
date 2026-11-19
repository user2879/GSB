from __future__ import annotations

import copy
import unittest
from pathlib import Path

from tests.helpers import SAMPLES, ROOT
from voicecal.config import load_config
from voicecal.constants import E_AUDIO_FORMAT
from voicecal.errors import VoicecalError
from voicecal.evidence import build_initial_evidence
from voicecal.features import WorkStats, aggregate_audio, extract_audio_frames
from voicecal.util import canonical_json
from voicecal.wav_io import parse_wav_bytes, read_wav


class AcousticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(SAMPLES / "calibration.json")
        cls.payload = read_wav(SAMPLES / "audio" / "t0001.wav")

    def test_riff_validation_rejects_truncated_or_wrong_headers(self) -> None:
        truncated = self.payload.raw[:-1]
        with self.assertRaises(VoicecalError) as context:
            parse_wav_bytes(truncated, Path("broken.wav"))
        self.assertEqual(context.exception.code, E_AUDIO_FORMAT)
        wrong_rate = bytearray(self.payload.raw)
        wrong_rate[24:28] = (8000).to_bytes(4, "little")
        with self.assertRaises(VoicecalError) as context:
            parse_wav_bytes(bytes(wrong_rate), Path("wrong-rate.wav"))
        self.assertEqual(context.exception.code, E_AUDIO_FORMAT)
        self.assertIn("sample rate", str(context.exception))

    def test_frame_features_are_deterministic_and_padding_is_marked(self) -> None:
        stats_a = WorkStats()
        stats_b = WorkStats()
        frames_a = extract_audio_frames(self.payload.samples, "audio/t0001.wav", self.config["frame"], 3, stats_a)
        frames_b = extract_audio_frames(self.payload.samples, "audio/t0001.wav", self.config["frame"], 3, stats_b)
        self.assertEqual(canonical_json(frames_a), canonical_json(frames_b))
        self.assertGreater(len(frames_a), 0)
        self.assertTrue(any(frame["padded"] for frame in frames_a))
        self.assertEqual([frame["frame_index"] for frame in frames_a], list(range(len(frames_a))))
        self.assertEqual(frames_a, sorted(frames_a, key=lambda frame: frame["sort_key"]))
        self.assertEqual(stats_a.frame_tasks, len(frames_a))
        self.assertGreaterEqual(stats_a.max_active_workers, 1)
        self.assertLessEqual(stats_a.max_active_workers, 3)

    def test_initial_evidence_is_fixed_and_repeatable(self) -> None:
        stats = WorkStats()
        frames = extract_audio_frames(self.payload.samples, "audio/t0001.wav", self.config["frame"], 2, stats)
        aggregate = aggregate_audio(frames, int(self.payload.samples.size), "audio/t0001.wav", self.config["frame"], stats)
        key_a, record_a = build_initial_evidence(aggregate, "extract-key", self.config)
        key_b, record_b = build_initial_evidence(aggregate, "extract-key", self.config)
        self.assertEqual(key_a, key_b)
        self.assertEqual(canonical_json(record_a), canonical_json(record_b))
        self.assertTrue(all(isinstance(value, int) for value in record_a["intensities_fixed"].values()))
        self.assertEqual(record_a["frames_used"], len(frames))
        self.assertEqual(set(record_a["scoring_feature_subset"]), set(record_a["normalized_features_fixed"]))
        self.assertEqual(record_a["status"], "ok")


if __name__ == "__main__":
    unittest.main()
