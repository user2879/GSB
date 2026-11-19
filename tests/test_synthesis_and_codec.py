import json
import os
import tempfile

import numpy as np
import unittest

from voice_repair_workbench import audio_io, lossy_codec, synthesis
from voice_repair_workbench.errors import E_CHANNEL_MISMATCH, E_FORMAT_UNSUPPORTED, RepairError


class TestSynthesisAndCodec(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.paths = synthesis.generate_material(self.tmp)

    def test_generation_is_byte_deterministic(self):
        other = tempfile.mkdtemp()
        paths2 = synthesis.generate_material(other)
        for key in ("mono16", "stereo24", "segments"):
            with open(self.paths[key], "rb") as a, open(paths2[key], "rb") as b:
                assert a.read() == b.read()

    def test_format_detection(self):
        mono = audio_io.parse_audio(self.paths["mono16"], expected_channels=1)
        assert mono.sample_rate == 48000
        assert mono.bit_depth == 16 and mono.channels == 1 and mono.lossy is False
        stereo = audio_io.parse_audio(self.paths["stereo24"], expected_channels=2)
        assert stereo.bit_depth == 24 and stereo.channels == 2

    def test_lossy_roundtrip_is_lossy_and_deterministic(self):
        mono = audio_io.parse_audio(self.paths["mono16"], expected_channels=1)
        lrx = lossy_codec.encode_file(
            mono.mono_analysis, mono.sample_rate, os.path.join(self.tmp, "a.lrx")
        )
        parsed = audio_io.parse_audio(lrx, expected_channels=1)
        assert parsed.lossy is True and parsed.lossy_format == "LRX1"
        decoded, _ = lossy_codec.decode_file(lrx)
        assert not np.array_equal(decoded, mono.mono_analysis)
        assert lossy_codec.encode_file(
            mono.mono_analysis, mono.sample_rate, os.path.join(self.tmp, "b.lrx")
        )
        with open(os.path.join(self.tmp, "a.lrx"), "rb") as a, open(
            os.path.join(self.tmp, "b.lrx"), "rb"
        ) as b:
            assert a.read() == b.read()

    def test_channel_mismatch_is_explicit(self):
        try:
            audio_io.parse_audio(self.paths["stereo24"], expected_channels=1)
        except RepairError as exc:
            assert exc.code == E_CHANNEL_MISMATCH and exc.field == "channels"
        else:  # pragma: no cover
            raise AssertionError("expected E_CHANNEL_MISMATCH")

    def test_unsupported_extension(self):
        try:
            audio_io.parse_audio(self.paths["mono16"] + ".bogus")
        except RepairError as exc:
            assert exc.code == E_FORMAT_UNSUPPORTED
        else:  # pragma: no cover
            raise AssertionError("expected E_FORMAT_UNSUPPORTED")

    def test_segments_present(self):
        with open(self.paths["segments"], encoding="utf-8") as fh:
            data = json.load(fh)
        kinds = [s["kind"] for s in data["segments"]]
        assert "dialogue" in kinds and "silence" in kinds and "room_noise" in kinds


if __name__ == '__main__':
    unittest.main()
