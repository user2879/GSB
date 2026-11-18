"""Synthetic dataset generation is seed deterministic and spec compliant."""
from __future__ import annotations

import os
import wave

from voiceprint.audio import read_wav
from voiceprint.synth import generate_dataset, synth_utterance, SpeakerProfile
from voiceprint.util import sha256_bytes


def test_synth_utterance_deterministic():
    profile = SpeakerProfile("spk", 150.0, 5.0, (700.0, 1100.0, 2400.0),
                             (1.0, 0.7, 0.4), 4.0)
    a = synth_utterance(profile, "u1")
    b = synth_utterance(profile, "u1")
    assert a.tolist() == b.tolist()
    c = synth_utterance(profile, "u2")
    assert a.tolist() != c.tolist()


def test_generated_wav_is_16k_mono_16bit(dataset_dir):
    path = os.path.join(dataset_dir, "spk_01_utt_01.wav")
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
    wave_data = read_wav(path)
    assert wave_data.n_frames_pcm == 16000
    with wave.open(path, "rb") as wf:
        pcm_blob = wf.readframes(wf.getnframes())
    assert wave_data.data_fp == sha256_bytes(pcm_blob)


def test_dataset_is_byte_reproducible(dataset_dir, tmp_path):
    from tests.conftest import INJECTION_PATH
    second = tmp_path / "samples2"
    second.mkdir()
    generate_dataset(str(second), INJECTION_PATH)
    for name in sorted(os.listdir(dataset_dir)):
        if not name.endswith(".wav"):
            continue
        a = open(os.path.join(dataset_dir, name), "rb").read()
        b = open(os.path.join(str(second), name), "rb").read()
        assert a == b, name
