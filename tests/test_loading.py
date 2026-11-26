import json
import os
import struct
import wave

from tests._base import BaseCase
from vdu.errors import AudioFormatError, InputInvalidError
from vdu.loading import (load_sample, validate_config, validate_transcript,
                         validate_wav)


class LoadingTests(BaseCase):
    def test_good_sample_loads(self):
        sample = load_sample(self.samples)
        self.assertEqual(sample["frames"], 27120)
        self.assertEqual(len(sample["transcript"]["segments"]), 4)

    def test_bad_sample_rate(self):
        path = self._wav(rate=8000)
        with self.assertRaises(AudioFormatError) as ctx:
            validate_wav(path)
        self.assertEqual(ctx.exception.code, "E_AUDIO_FORMAT")

    def test_stereo_rejected(self):
        path = self._wav(channels=2)
        with self.assertRaises(AudioFormatError) as ctx:
            validate_wav(path)
        self.assertEqual(ctx.exception.code, "E_AUDIO_FORMAT")

    def test_8bit_rejected(self):
        path = self._wav(width=1)
        with self.assertRaises(AudioFormatError) as ctx:
            validate_wav(path)
        self.assertEqual(ctx.exception.code, "E_AUDIO_FORMAT")

    def test_data_length_mismatch(self):
        good = os.path.join(self.samples, 'dialogue_demo.wav')
        with open(good, "rb") as handle:
            raw = handle.read()
        pos = raw.find(b"data")
        broken = raw[:pos + 4] + struct.pack("<I", 999) + raw[pos + 8:]
        path = os.path.join(self.samples, 'bad.wav')
        with open(path, "wb") as handle:
            handle.write(broken)
        with self.assertRaises(AudioFormatError) as ctx:
            validate_wav(path)
        self.assertEqual(ctx.exception.code, "E_AUDIO_FORMAT")

    def test_word_outside_vocab(self):
        self.transcript["segments"][0]["words"][0]["word"] = "zzz"
        self.write_transcript(self.transcript)
        with self.assertRaises(InputInvalidError) as ctx:
            load_sample(self.samples)
        self.assertEqual(ctx.exception.code, "E_INPUT_INVALID")
        self.assertIsNotNone(ctx.exception.line)

    def test_word_offset_escapes_segment(self):
        word = self.transcript["segments"][0]["words"][0]
        word["end"] = self.transcript["segments"][0]["end"] + 10
        self.write_transcript(self.transcript)
        with self.assertRaises(InputInvalidError) as ctx:
            load_sample(self.samples)
        self.assertEqual(ctx.exception.code, "E_INPUT_INVALID")

    def test_duplicate_interval_rejected(self):
        seg = self.transcript["segments"][0]
        dup = dict(seg)
        self.transcript["segments"].append(dup)
        self.write_transcript(self.transcript)
        with self.assertRaises(InputInvalidError) as ctx:
            load_sample(self.samples)
        self.assertEqual(ctx.exception.code, "E_INPUT_INVALID")

    def test_label_version_decrease_rejected(self):
        self.transcript["segments"][2]["label_version"] = 1
        for word in self.transcript["segments"][2]["words"]:
            word["annotator_version"] = 1
        self.write_transcript(self.transcript)
        with self.assertRaises(InputInvalidError) as ctx:
            load_sample(self.samples)
        self.assertEqual(ctx.exception.code, "E_INPUT_INVALID")

    def test_config_missing_key(self):
        del self.config["scheduler"]
        with open(self.config_path, "w") as handle:
            json.dump(self.config, handle)
        with self.assertRaises(InputInvalidError) as ctx:
            validate_config(self.config, self.config_path)
        self.assertEqual(ctx.exception.code, "E_INPUT_INVALID")

    def _wav(self, rate=16000, channels=1, width=2):
        path = os.path.join(self.samples, 'bad.wav')
        frames = 100
        with wave.open(path, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(width)
            wav.setframerate(rate)
            if width == 2:
                wav.writeframes(struct.pack("<%dh" % (frames * channels),
                                            *([0] * frames * channels)))
            else:
                wav.writeframes(bytes(frames * channels))
        return path

