"""Requirement 2: silence gating, deterministic statistics and fingerprints."""
from __future__ import annotations

import os

import pytest

from voiceprint.aggregation import aggregate
from voiceprint.audio import read_wav
from voiceprint.errors import InsufficientFramesError
from voiceprint.features import extract_frames


def test_aggregation_shape_and_fingerprint(dataset_dir, base_config):
    wave = read_wav(os.path.join(dataset_dir, "spk_01_utt_01.wav"))
    frames = extract_frames(wave, base_config)
    agg = aggregate(frames, base_config)
    # f0(1) + cep(order) + energy(1) columns, each with mean/var + 3 quantiles
    expected_cols = (1 + base_config.cepstrum_order + 1)
    assert agg.stats_vector.shape[0] == expected_cols * (2 + len(base_config.quantile_points))
    assert agg.valid_frame_count >= base_config.min_valid_frames
    assert agg.valid_frame_indices == sorted(agg.valid_frame_indices)
    assert agg.aggregation_fp


def test_aggregation_deterministic(dataset_dir, base_config):
    wave = read_wav(os.path.join(dataset_dir, "spk_03_utt_02.wav"))
    a = aggregate(extract_frames(wave, base_config), base_config)
    b = aggregate(extract_frames(wave, base_config), base_config)
    assert a.aggregation_fp == b.aggregation_fp
    assert a.stats_vector.tolist() == b.stats_vector.tolist()


def test_insufficient_fragments_raises(dataset_dir, make_config):
    cfg = make_config(min_valid_frames=10_000)
    wave = read_wav(os.path.join(dataset_dir, "spk_01_utt_01.wav"))
    frames = extract_frames(wave, cfg)
    with pytest.raises(InsufficientFramesError) as exc:
        aggregate(frames, cfg)
    assert exc.value.code == "E_INSUFFICIENT_FRAMES"
    # partial summary is retained rather than an empty downstream vector
    assert exc.value.context["frames_above_gate"] < cfg.min_valid_frames
