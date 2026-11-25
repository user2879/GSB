"""Shared pytest fixtures for the streaming VAD diagnostics test suite."""

from __future__ import annotations

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vad_diag.config import build_config
from vad_diag.samplegen import build_segments
from vad_diag.segments import build_in_memory_segment, write_wav


BASE_CONFIG = {
    "sample_rate": 16000,
    "frame_length": 400,
    "frame_shift": 160,
    "window_length": 800,
    "window_shift": 320,
    "energy_low": 0.01,
    "energy_high": 0.05,
    "flatness_low": 0.20,
    "flatness_high": 0.05,
    "min_speech_frames": 3,
    "min_silence_frames": 3,
    "frac_bits": 15,
    "quant_mode": "ROUND_HALF_EVEN",
    "speech_threshold": 0.5,
    "latency_budget_ms": 10_000,
    "segment_cost_ms": 1,
    "window_cost_ms": 1,
    "max_workers": 4,
    "snapshot_dir": "snapshots",
    "ledger_dir": "ledger",
}


@pytest.fixture
def workdir(tmp_path):
    yield tmp_path


@pytest.fixture
def config_factory(workdir):
    def make(**overrides):
        raw = dict(BASE_CONFIG)
        raw["snapshot_dir"] = str(workdir / "snapshots")
        raw["ledger_dir"] = str(workdir / "ledger")
        raw.update(overrides)
        return build_config(raw)
    return make


@pytest.fixture
def segment_factory():
    def make(seg_id="S0000", samples=None, capture_ms=0, arrival_ms=10,
             process_ms=5):
        if samples is None:
            entries, _ = build_segments(segment_count=1,
                                        segment_samples=3200)
            samples = entries[0]["samples"]
        return build_in_memory_segment(
            seg_id, samples, capture_ms, arrival_ms, process_ms)
    return make


@pytest.fixture
def dataset_segments():
    entries, ids = build_segments(segment_count=20, segment_samples=3200)
    return {e["segment_id"]: build_in_memory_segment(
        e["segment_id"], e["samples"], e["capture_ms"], e["arrival_ms"],
        e["process_ms"]) for e in entries}


@pytest.fixture
def wav_file(workdir):
    def make(samples, name="seg.wav", rate=16000):
        path = os.path.join(workdir, name)
        write_wav(path, samples, rate)
        return path
    return make
