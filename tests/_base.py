"""Shared deterministic test fixtures."""

import json
import os
import shutil
import struct
import tempfile
import unittest
import wave

from vdu.synth import (generate_demo, render_pcm, build_transcript,
                       default_config, write_wav)


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="vdu_test_")
        self.samples = os.path.join(self.root, "samples")
        self.configs = os.path.join(self.root, "configs")
        self.snapshots = os.path.join(self.root, "snapshots")
        generate_demo(self.samples, self.configs)
        self.transcript_path = os.path.join(self.samples,
                                            "dialogue_demo.json")
        self.config_path = os.path.join(self.configs, "calibration.json")
        with open(self.transcript_path, encoding="utf-8") as handle:
            self.transcript = json.load(handle)
        with open(self.config_path, encoding="utf-8") as handle:
            self.config = json.load(handle)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write_transcript(self, transcript):
        with open(self.transcript_path, "w", encoding="utf-8") as handle:
            json.dump(transcript, handle, indent=2, ensure_ascii=False)

    def rewrite_audio(self, transcript):
        write_wav(os.path.join(self.samples, transcript["audio"]),
                  render_pcm(transcript))

    def write_wav_bytes(self, name, raw_bytes, params=None):
        path = os.path.join(self.samples, name)
        with open(path, "wb") as handle:
            handle.write(raw_bytes)
        return path
