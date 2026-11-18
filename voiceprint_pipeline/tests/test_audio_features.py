"""Requirements 1: strict WAV validation and deterministic frame features."""
from __future__ import annotations

import os

import numpy as np
import pytest

from voiceprint.audio import read_wav, list_wav_files
from voiceprint.errors import AudioFormatError
from voiceprint.features import extract_frames, frame_grid


def test_valid_wav_loads(dataset_dir, base_config):
    path = os.path.join(dataset_dir, "spk_01_utt_01.wav")
    wave = read_wav(path)
    assert wave.n_frames_pcm == 16000
    assert wave.samples.dtype == np.float64
    frames = extract_frames(wave, base_config)
    assert frames.n_frames >= base_config.min_valid_frames
    assert frames.mel_energy.shape == (frames.n_frames, base_config.mel_bands)
    assert frames.cepstrum.shape == (frames.n_frames, base_config.cepstrum_order)
    assert frames.f0.shape == (frames.n_frames,)


def test_frame_grid_zero_pads_tail(dataset_dir, base_config):
    wave = read_wav(os.path.join(dataset_dir, "spk_01_utt_01.wav"))
    frame_size = int(round(base_config.frame_length_ms * 16000 / 1000))
    hop = int(round(base_config.frame_shift_ms * 16000 / 1000))
    n, starts, ends = frame_grid(wave.n_frames_pcm, frame_size, hop)
    assert n == len(starts) == len(ends)
    frames = extract_frames(wave, base_config)
    # at most one tail frame and it is the last one
    tail_idx = [i for i, flag in enumerate(frames.is_tail) if flag]
    assert tail_idx == [] or tail_idx == [n - 1]


def test_feature_extraction_is_elementwise_deterministic(dataset_dir, base_config):
    path = os.path.join(dataset_dir, "spk_02_utt_01.wav")
    first = extract_frames(read_wav(path), base_config)
    second = extract_frames(read_wav(path), base_config)
    assert first.feature_fp == second.feature_fp
    np.testing.assert_array_equal(first.f0, second.f0)
    np.testing.assert_array_equal(first.mel_energy, second.mel_energy)
    np.testing.assert_array_equal(first.cepstrum, second.cepstrum)
    np.testing.assert_array_equal(first.energy_db, second.energy_db)


def test_invalid_sample_rate_rejected(dataset_dir):
    with pytest.raises(AudioFormatError) as exc:
        read_wav(os.path.join(dataset_dir, "bad_rate_8k.wav"))
    assert exc.value.code == "E_AUDIO_FORMAT"


def test_stereo_rejected(dataset_dir):
    with pytest.raises(AudioFormatError) as exc:
        read_wav(os.path.join(dataset_dir, "bad_stereo.wav"))
    assert exc.value.code == "E_AUDIO_FORMAT"


def test_list_wav_sorted(dataset_dir):
    paths = list_wav_files(dataset_dir)
    assert paths == sorted(paths)
    assert len(paths) == 14
