from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG
from .synth import synthesize_levels
from .wavio import write_wav_from_levels


DEMO_RATES = [11025, 22050, 44100]
DEMO_DEPTHS = [8, 16, 24, 32]


def _library_clip(clip_id: str, track_id: str, version: int, seed: int, rate: int, depth: int) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "track_id": track_id,
        "version": version,
        "role": "library",
        "seed": seed,
        "path": f"clips/{clip_id}.wav",
        "internal_sample_rate": 11025,
        "internal_sample_count": 12288,
        "source_sample_rate": rate,
        "bit_depth": depth,
        "declared_sample_rates": DEMO_RATES,
        "declared_bit_depths": DEMO_DEPTHS,
    }


def build_demo_manifest() -> dict[str, Any]:
    clips = [
        _library_clip("c-alpha-v1", "trk-alpha", 1, 101, 22050, 16),
        _library_clip("c-alpha-v2-fmt", "trk-alpha", 2, 101, 44100, 8),
        _library_clip("c-beta-v1", "trk-beta", 1, 202, 44100, 16),
        _library_clip("c-gamma-v1", "trk-gamma", 1, 303, 22050, 24),
        _library_clip("c-delta-v1", "trk-delta", 1, 404, 11025, 8),
        _library_clip("c-epsilon-v1", "trk-epsilon", 1, 505, 22050, 32),
        {
            "clip_id": "q-alpha-offset2",
            "track_id": "zz-query-alpha",
            "version": 1,
            "role": "query",
            "expected_track_id": "trk-alpha",
            "seed": 101,
            "path": "queries/q-alpha-offset2.wav",
            "internal_sample_rate": 11025,
            "internal_sample_count": 12288,
            "start_sample": 256,
            "excerpt_sample_count": 4096,
            "source_sample_rate": 44100,
            "bit_depth": 24,
            "declared_sample_rates": DEMO_RATES,
            "declared_bit_depths": DEMO_DEPTHS,
        },
        {
            "clip_id": "q-beta-offset3",
            "track_id": "zz-query-beta",
            "version": 1,
            "role": "query",
            "expected_track_id": "trk-beta",
            "seed": 202,
            "start_sample": 384,
            "excerpt_sample_count": 3584,
            "path": "queries/q-beta-offset3.wav",
            "internal_sample_rate": 11025,
            "internal_sample_count": 12288,
            "source_sample_rate": 22050,
            "bit_depth": 8,
            "declared_sample_rates": DEMO_RATES,
            "declared_bit_depths": DEMO_DEPTHS,
        },
        {
            "clip_id": "q-gamma-offset4",
            "track_id": "zz-query-gamma",
            "version": 1,
            "role": "query",
            "expected_track_id": "trk-gamma",
            "seed": 303,
            "start_sample": 512,
            "excerpt_sample_count": 3840,
            "path": "queries/q-gamma-offset4.wav",
            "internal_sample_rate": 11025,
            "internal_sample_count": 12288,
            "source_sample_rate": 11025,
            "bit_depth": 16,
            "declared_sample_rates": DEMO_RATES,
            "declared_bit_depths": DEMO_DEPTHS,
        },
        {
            "clip_id": "bad-missing",
            "track_id": "trk-bad-missing",
            "version": 1,
            "role": "invalid_fixture",
            "path": "invalid/missing.wav",
            "source_sample_rate": 22050,
            "bit_depth": 16,
            "declared_sample_rates": [22050],
            "declared_bit_depths": [16],
        },
        {
            "clip_id": "bad-header",
            "track_id": "trk-bad-header",
            "version": 1,
            "role": "invalid_fixture",
            "path": "invalid/bad-header.wav",
            "source_sample_rate": 22050,
            "bit_depth": 16,
            "declared_sample_rates": [22050],
            "declared_bit_depths": [16],
        },
        {
            "clip_id": "bad-rate",
            "track_id": "trk-bad-rate",
            "version": 1,
            "role": "invalid_fixture",
            "path": "invalid/bad-rate.wav",
            "source_sample_rate": 8000,
            "bit_depth": 16,
            "declared_sample_rates": [22050, 44100],
            "declared_bit_depths": [16],
        },
        {
            "clip_id": "bad-depth",
            "track_id": "trk-bad-depth",
            "version": 1,
            "role": "invalid_fixture",
            "path": "invalid/bad-depth.wav",
            "source_sample_rate": 22050,
            "bit_depth": 8,
            "declared_sample_rates": [22050],
            "declared_bit_depths": [16],
        },
        {
            "clip_id": "bad-data",
            "track_id": "trk-bad-data",
            "version": 1,
            "role": "invalid_fixture",
            "path": "invalid/bad-data.wav",
            "source_sample_rate": 22050,
            "bit_depth": 16,
            "declared_sample_rates": [22050],
            "declared_bit_depths": [16],
        },
        {
            "clip_id": "bad-duplicate-pair",
            "track_id": "trk-alpha",
            "version": 1,
            "role": "invalid_fixture",
            "path": "clips/c-alpha-v1.wav",
            "source_sample_rate": 22050,
            "bit_depth": 16,
            "declared_sample_rates": DEMO_RATES,
            "declared_bit_depths": DEMO_DEPTHS,
        },
    ]
    return {"manifest_version": 1, "clips": clips}


def build_edit_commands() -> dict[str, Any]:
    return {
        "commands": [
            {"op": "replace_clip", "clip_id": "c-beta-v1", "seed": 209, "command_id": "cmd-0000"},
            {
                "op": "add_clip",
                "clip": _library_clip("c-zeta-v1", "trk-zeta", 1, 606, 22050, 16),
                "command_id": "cmd-0001",
            },
            {"op": "set_peak_threshold", "value": 1200, "command_id": "cmd-0002"},
            {"op": "set_frame_threshold", "clip_id": "c-alpha-v1", "frame_index": 12, "value": 15000, "command_id": "cmd-0003"},
            {"op": "set_bucket_capacity", "value": 4, "command_id": "cmd-0004"},
            {"op": "set_time_quant_gear", "value": "tq_coarse", "command_id": "cmd-0005"},
            {"op": "set_clip_format", "clip_id": "c-delta-v1", "sample_rate": 44100, "bit_depth": 16, "command_id": "cmd-0006"},
            {"op": "delete_clip", "clip_id": "c-epsilon-v1", "command_id": "cmd-0007"},
        ]
    }


def build_invalid_command_stream() -> dict[str, Any]:
    return {
        "commands": [
            {"op": "set_peak_threshold", "value": 0},
            {"op": "set_bucket_capacity", "value": -2},
            {"op": "set_time_quant_gear", "value": "tq_unknown"},
            {"op": "set_frame_threshold", "clip_id": "c-alpha-v1", "frame_index": -1, "value": 10},
            {"op": "unknown_op"},
            {"op": "set_peak_threshold", "value": 1500},
        ]
    }


def create_demo_workspace(root: str | Path = "sample_workspace") -> dict[str, Any]:
    root = Path(root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_demo_manifest()

    from .synth import generate_clip_file

    generated: list[dict[str, Any]] = []
    for clip in manifest["clips"]:
        if clip.get("role") == "invalid_fixture":
            continue
        generated.append(generate_clip_file(clip, data_dir))

    bad_header = data_dir / "invalid" / "bad-header.wav"
    bad_header.parent.mkdir(parents=True, exist_ok=True)
    bad_header.write_bytes(b"NOT-A-WAV" + bytes(64 - 9))

    levels = synthesize_levels(777, 2048, 11025)
    write_wav_from_levels(data_dir / "invalid" / "bad-rate.wav", levels[:1024], 8000, 16)
    write_wav_from_levels(data_dir / "invalid" / "bad-depth.wav", levels[:1024], 22050, 8)
    bad_data_path = data_dir / "invalid" / "bad-data.wav"
    info = write_wav_from_levels(bad_data_path, levels[:2048], 22050, 16)
    raw = bytearray(bad_data_path.read_bytes())
    data_chunk_offset = info["data_offset"] - 8
    import struct

    struct.pack_into("<I", raw, data_chunk_offset + 4, int(info["data_size"]) + 128)
    bad_data_path.write_bytes(bytes(raw))

    config = dict(DEFAULT_CONFIG)
    config["candidate_limit"] = 4
    config["bucket_capacity"] = 6
    config["max_concurrency"] = 3
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "edit_commands.json").write_text(
        json.dumps(build_edit_commands(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "invalid_commands.json").write_text(
        json.dumps(build_invalid_command_stream(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"root": str(root), "clip_count": len(manifest["clips"]), "data_dir": str(data_dir)}
