"""Requirement 1: loading and validation layer with explicit failure codes."""

from __future__ import annotations

import json
import os
import struct

import pytest

from vad_diag.config import build_config
from vad_diag.errors import E_AUDIO_FORMAT, E_FIELD_INVALID
from vad_diag.loader import load_command_stream, load_segment_manifest
from vad_diag.segments import (
    build_in_memory_segment, parse_wav_segment, write_wav,
)
from vad_diag.samplegen import build_segments


def _manifest(tmp_path, entries):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"segments": entries}))
    return str(path)


def test_valid_wav_roundtrip(workdir, wav_file):
    entries, _ = build_segments(segment_count=2, segment_samples=800)
    path = wav_file(entries[0]["samples"], "ok.wav")
    seg = parse_wav_segment(path, "S0000", 0, 10, 5)
    assert seg.sample_rate == 16000
    assert seg.channels == 1
    assert seg.sample_width == 2
    assert seg.num_samples == 800
    assert seg.data_byte_len == 1600


def test_truncated_riff_is_audio_format(workdir, wav_file):
    entries, _ = build_segments(segment_count=1, segment_samples=800)
    path = wav_file(entries[0]["samples"], "trunc.wav")
    blob = bytearray(open(path, "rb").read())
    # Lie about the RIFF size and truncate the data chunk.
    struct.pack_into("<I", blob, 4, len(blob) + 100)
    bad = workdir / "bad_riff.wav"
    bad.write_bytes(bytes(blob))
    with pytest.raises(Exception) as exc:
        parse_wav_segment(str(bad), "S0000", 0, 10, 5)
    assert exc.value.code == E_AUDIO_FORMAT


def test_wrong_sample_rate_rejected(workdir):
    samples = [0] * 800
    path = os.path.join(workdir, "rate.wav")
    write_wav(path, samples, rate=8000)
    with pytest.raises(Exception) as exc:
        parse_wav_segment(path, "S0000", 0, 10, 5)
    assert exc.value.code == E_AUDIO_FORMAT


def test_negative_timestamp_is_field_invalid(segment_factory):
    entries, _ = build_segments(segment_count=1)
    with pytest.raises(Exception) as exc:
        build_in_memory_segment("S0000", entries[0]["samples"],
                                -1, 10, 5)
    assert exc.value.code == E_FIELD_INVALID


def test_sample_out_of_range_rejected():
    with pytest.raises(Exception) as exc:
        build_in_memory_segment("S0000", [40000], 0, 10, 5)
    assert exc.value.code == E_AUDIO_FORMAT


def test_manifest_collects_failures_without_silent_skip(workdir):
    good_entries, _ = build_segments(segment_count=2, segment_samples=400)
    entries = [
        {"segment_id": "S0000", "capture_ms": 0, "arrival_ms": 10,
         "samples": good_entries[0]["samples"]},
        {"segment_id": "S0001", "capture_ms": -5, "arrival_ms": 10,
         "samples": good_entries[1]["samples"]},
        {"segment_id": "S0000", "capture_ms": 0, "arrival_ms": 10,
         "samples": good_entries[0]["samples"]},
    ]
    result = load_segment_manifest(_manifest(workdir, entries))
    assert set(result.segments) == {"S0000"}
    codes = [f.code for f in result.failures]
    assert E_FIELD_INVALID in codes
    assert len(result.failures) == 2
    # Failure sequence numbers are deterministic and contiguous.
    assert [f.seq for f in result.failures] == [0, 1]


def test_config_frame_geometry_validation(workdir):
    raw = {
        "frame_length": 100, "frame_shift": 200,
        "window_length": 400, "window_shift": 100,
        "energy_low": 0.1, "energy_high": 0.05,
        "flatness_low": 0.2, "flatness_high": 0.1,
        "min_speech_frames": 1, "min_silence_frames": 1,
        "frac_bits": 15, "quant_mode": "ROUND_HALF_EVEN",
        "speech_threshold": 0.5, "latency_budget_ms": 10,
        "max_workers": 2,
        "snapshot_dir": str(workdir / "s"),
        "ledger_dir": str(workdir / "l"),
    }
    with pytest.raises(Exception) as exc:
        build_config(raw)
    assert exc.value.code == E_FIELD_INVALID


def test_config_quant_enum_validation(workdir):
    raw = {
        "frame_length": 400, "frame_shift": 160,
        "window_length": 800, "window_shift": 320,
        "energy_low": 0.01, "energy_high": 0.05,
        "flatness_low": 0.2, "flatness_high": 0.05,
        "min_speech_frames": 1, "min_silence_frames": 1,
        "frac_bits": 12, "quant_mode": "ROUND_HALF_EVEN",
        "speech_threshold": 0.5, "latency_budget_ms": 10,
        "max_workers": 2,
        "snapshot_dir": str(workdir / "s"),
        "ledger_dir": str(workdir / "l"),
    }
    with pytest.raises(Exception) as exc:
        build_config(raw)
    assert exc.value.code == E_FIELD_INVALID


def test_command_stream_rejects_unknown_op(workdir):
    path = workdir / "cmds.json"
    path.write_text(json.dumps({"commands": [{"op": "bogus"}]}))
    with pytest.raises(Exception) as exc:
        load_command_stream(str(path))
    assert exc.value.code == E_FIELD_INVALID
