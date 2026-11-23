from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from timbre_workbench.config import load_config
from timbre_workbench.segment import detect_speech_segments
from timbre_workbench.synth import silence, synthesize_segment, TIMBRES
from timbre_workbench.wavio import parse_wav_bytes, read_wav, scan_wav_directory, wav_bytes, write_wav


class WavValidationTests(unittest.TestCase):
    def test_valid_wav_round_trip_and_strict_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = synthesize_segment(16000, 120.0, TIMBRES["alpha"], 1)
            good = root / "good.wav"
            write_wav(good, samples)
            parsed = read_wav(good)
            self.assertEqual(parsed.sample_count, len(samples))
            self.assertEqual(parsed.sample_rate, 16000)
            self.assertEqual(parsed.channels, 1)
            self.assertEqual(parsed.bits_per_sample, 16)

            with self.assertRaisesRegex(Exception, "RIFF size"):
                parse_wav_bytes(good.read_bytes()[:-1])
            with self.assertRaisesRegex(Exception, "sample rate"):
                parse_wav_bytes(wav_bytes(samples, sample_rate=8000))

    def test_scan_keeps_processing_after_bad_file_and_sorts_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_wav(root / "z_good.wav", synthesize_segment(16000, 80.0, TIMBRES["beta"], 2))
            write_wav(root / "a_good.wav", synthesize_segment(16000, 80.0, TIMBRES["gamma"], 3))
            (root / "m_bad.wav").write_bytes(b"RIFF\x10\x00\x00\x00WAVEbad!")
            (root / "n_bad.wav").write_bytes(b"not a wav")
            assets, failures = scan_wav_directory(root)
            self.assertEqual([asset.logical_name for asset in assets], ["a_good.wav", "z_good.wav"])
            self.assertEqual([failure["file"] for failure in failures], ["m_bad.wav", "n_bad.wav"])
            self.assertTrue(all(failure["error_code"] == "E_AUDIO_FORMAT" for failure in failures))


class SegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.sr = 16000

    def _signal(self, gap_ms: float, duration_ms: float = 260.0) -> np.ndarray:
        return np.concatenate(
            [
                silence(self.sr, 100.0),
                synthesize_segment(self.sr, duration_ms, TIMBRES["alpha"], 10),
                silence(self.sr, gap_ms),
                synthesize_segment(self.sr, duration_ms, TIMBRES["alpha"], 11),
                silence(self.sr, 120.0),
            ]
        )

    def test_short_gap_is_merged_and_long_gap_is_split(self) -> None:
        merged = detect_speech_segments(self._signal(25.0), self.sr, self.config["segmentation"])
        split = detect_speech_segments(self._signal(220.0), self.sr, self.config["segmentation"])
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(split), 2)
        self.assertLess(merged[0]["start_sample"], merged[0]["end_sample"])
        self.assertLess(split[0]["end_sample"], split[1]["start_sample"])

    def test_short_segment_is_discarded(self) -> None:
        signal = np.concatenate(
            [
                silence(self.sr, 100.0),
                synthesize_segment(self.sr, 40.0, TIMBRES["alpha"], 12),
                silence(self.sr, 150.0),
            ]
        )
        segments = detect_speech_segments(signal, self.sr, self.config["segmentation"])
        self.assertEqual(segments, [])


if __name__ == "__main__":
    unittest.main()

