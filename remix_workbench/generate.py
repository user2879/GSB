from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .fingerprint import canonical_json
from .formats import (
    V1_AUDIO_FILE,
    V2_MANIFEST_FILE,
    V3_HEADER,
    V3_MAGIC,
    V3_PACKAGE_FILE,
    V3_TABLE_ENTRY,
    V3_TABLE_PREFIX,
)
from .models import V1_TRACK_ORDER
from .pcm import (
    decode_pcm_bytes,
    q23_to_i16,
    read_wav,
    write_pcm_wav,
    write_wav,
)

DEFAULT_SEED = 20261118
CLIP_FRAMES = (12800, 14080, 15360)
TRACK_GAIN_DB = {"bass": -2.5, "drums": -1.0, "vocal": -3.5}


def _fade(length: int, sample_rate: int) -> np.ndarray:
    fade_len = min(int(sample_rate * 0.012), length // 2)
    envelope = np.ones(length, dtype=np.float64)
    if fade_len > 0:
        ramp = np.linspace(0.0, 1.0, fade_len, endpoint=False)
        envelope[:fade_len] = ramp
        envelope[-fade_len:] = ramp[::-1]
    return envelope


def _normalize_track(values: np.ndarray, peak: float = 0.72) -> np.ndarray:
    maximum = float(np.max(np.abs(values)))
    if maximum == 0:
        return values
    return values * (peak / maximum)


def _synthesize_tracks(
    frames: int, sample_rate: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    t = np.arange(frames, dtype=np.float64) / sample_rate
    global_fade = _fade(frames, sample_rate)

    beat_period = 0.32
    beat_phase = np.mod(t, beat_period)
    beat_index = np.floor(t / beat_period).astype(np.int64)
    kick_freq = 58.0 + 72.0 * np.exp(-beat_phase * 28.0)
    kick = np.sin(2.0 * np.pi * kick_freq * beat_phase) * np.exp(-beat_phase * 14.0)
    snare_gate = (beat_index % 2 == 1) * np.exp(-beat_phase * 22.0)
    snare = rng.standard_normal(frames) * snare_gate * 0.36
    hat = rng.standard_normal(frames) * np.exp(
        -np.mod(t, beat_period / 2.0) * 75.0
    ) * 0.12
    drums_env = np.clip(0.76 + 0.18 * np.sin(2 * np.pi * 0.43 * t), 0.45, 0.98)
    drums = _normalize_track((kick + snare + hat) * drums_env)

    bass_f0 = 82.4069
    bass_env = np.clip(0.58 + 0.32 * np.sin(2 * np.pi * 0.27 * t + 0.4), 0.2, 1.0)
    bass_phase = 2 * np.pi * bass_f0 * t + 0.18 * np.sin(2 * np.pi * 0.7 * t)
    bass = (
        np.sin(bass_phase)
        + 0.32 * np.sin(2 * bass_phase + 0.15)
        + 0.12 * np.sin(3 * bass_phase)
    ) * bass_env
    bass = _normalize_track(bass, peak=0.64)

    vibrato = 1.0 + 0.008 * np.sin(2 * np.pi * 5.2 * t)
    vocal_f0 = 220.0 * vibrato
    vocal_phase = 2 * np.pi * np.cumsum(vocal_f0) / sample_rate
    syllables = np.clip(
        0.28
        + 0.55 * np.maximum(0.0, np.sin(2 * np.pi * 2.65 * t))
        + 0.14 * np.sin(2 * np.pi * 0.38 * t + 1.1),
        0.08,
        1.0,
    )
    vocal = (
        np.sin(vocal_phase)
        + 0.28 * np.sin(2 * vocal_phase + 0.08)
        + 0.17 * np.sin(3 * vocal_phase + 0.4)
        + 0.08 * np.sin(5 * vocal_phase)
    ) * syllables
    vocal = _normalize_track(vocal, peak=0.68)

    tracks = {
        "bass": bass * global_fade,
        "drums": drums * global_fade,
        "vocal": vocal * global_fade,
    }
    envelopes = {
        "bass": bass_env * global_fade,
        "drums": drums_env * global_fade,
        "vocal": syllables * global_fade,
    }
    return tracks, envelopes


def _q23_from_float(values: np.ndarray) -> np.ndarray:
    scaled = np.floor(values * 8_388_608.0 + 0.5)
    return np.clip(scaled, -8_388_608, 8_388_607).astype(np.int32)


def _write_v1(frames: int, tracks: dict[str, np.ndarray], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    matrix = np.empty((frames, len(V1_TRACK_ORDER)), dtype=np.int16)
    for channel, name in enumerate(V1_TRACK_ORDER):
        matrix[:, channel] = q23_to_i16(tracks[name])
    write_wav(
        directory / V1_AUDIO_FILE,
        sample_rate=16000,
        channels=len(V1_TRACK_ORDER),
        sample_width=2,
        raw_pcm=matrix.astype("<i2").tobytes(order="C"),
    )
    descriptor = {
        "format_version": "v1",
        "sample_rate": 16000,
        "bit_depth": 16,
        "frames": frames,
        "channel_map": {name: index for index, name in enumerate(V1_TRACK_ORDER)},
        "note": "Audio is the authoritative v1 payload; names follow v1 fixed order.",
    }
    (directory / "package.json").write_text(
        canonical_json(descriptor) + "\n", encoding="utf-8"
    )


def _write_v2(
    frames: int,
    tracks: dict[str, np.ndarray],
    metadata: dict[str, Any],
    directory: Path,
) -> None:
    tracks_dir = directory / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    mapping = {"bass": 0, "drums": 2, "vocal": 1}
    entries: dict[str, Any] = {}
    for name in sorted(tracks):
        rel = f"tracks/{name}.wav"
        write_pcm_wav(
            directory / rel,
            tracks[name],
            sample_rate=16000,
            sample_width=3,
        )
        entries[name] = {
            "file": rel,
            "channel": mapping[name],
            "frames": frames,
            "bit_depth": 24,
            "gain_db": metadata["track_gain_metadata"][name],
            "source_frames": metadata["source_frame_annotations"][name],
        }
    manifest = {
        "format_version": "v2",
        "sample_rate": 16000,
        "bit_depth": 24,
        "frames": frames,
        "channel_map": mapping,
        "tracks": entries,
        "metadata": metadata,
    }
    (directory / V2_MANIFEST_FILE).write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )


def _write_v3(frames: int, tracks: dict[str, np.ndarray], directory: Path) -> None:
    names = ["bass", "vocal", "drums"]
    payloads = [
        q23_to_i16(tracks[name]).astype("<i2").tobytes(order="C") for name in names
    ]
    header_size = V3_HEADER.size + sum(
        V3_TABLE_PREFIX.size + len(name.encode("utf-8")) + V3_TABLE_ENTRY.size
        for name in names
    )
    cursor = header_size
    table = bytearray()
    for name, payload in zip(names, payloads):
        encoded = name.encode("utf-8")
        table.extend(V3_TABLE_PREFIX.pack(len(encoded)))
        table.extend(encoded)
        table.extend(V3_TABLE_ENTRY.pack(cursor, len(payload)))
        cursor += len(payload)
    header = V3_HEADER.pack(V3_MAGIC, 3, len(names), frames, 16000, 16, 0)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / V3_PACKAGE_FILE).write_bytes(header + bytes(table) + b"".join(payloads))


def _envelope_points(envelope: np.ndarray, sample_rate: int) -> list[dict[str, Any]]:
    step = int(sample_rate * 0.1)
    points = []
    for frame in range(0, envelope.size, step):
        points.append({"frame": frame, "value": round(float(envelope[frame]), 8)})
    if points[-1]["frame"] != envelope.size - 1:
        points.append(
            {"frame": envelope.size - 1, "value": round(float(envelope[-1]), 8)}
        )
    return points


def generate_workspace(root: str | Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    root = Path(root)
    samples_root = root / "samples"
    configs_root = root / "configs"
    samples_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)
    clips_manifest = []
    config_clips = []
    for index, frames in enumerate(CLIP_FRAMES):
        clip_id = f"clip_{index:03d}"
        clip_seed = seed + index * 1009
        track_float, envelopes = _synthesize_tracks(frames, 16000, clip_seed)
        tracks = {name: _q23_from_float(values) for name, values in track_float.items()}
        clip_dir = samples_root / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        float_mix = sum(track_float.values()) * 0.52
        float_mix = np.clip(float_mix, -1.0, 0.999969)
        write_pcm_wav(
            clip_dir / "mix.wav",
            q23_to_i16(_q23_from_float(float_mix)),
            sample_rate=16000,
            sample_width=2,
        )
        source_annotations = {
            name: {
                "original_frames": frames,
                "declared_frames": frames,
                "annotation": "full-length deterministic source",
            }
            for name in sorted(tracks)
        }
        metadata = {
            "source_frame_annotations": source_annotations,
            "track_gain_metadata": dict(TRACK_GAIN_DB),
            "synthesis_seed": clip_seed,
            "synthesis": "numpy deterministic additive synthesis",
        }
        _write_v1(frames, tracks, clip_dir / "packages" / "v1")
        _write_v2(frames, tracks, metadata, clip_dir / "packages" / "v2")
        _write_v3(frames, tracks, clip_dir / "packages" / "v3")
        envelope_data = {
            name: _envelope_points(values, 16000) for name, values in envelopes.items()
        }
        (clip_dir / "envelopes.json").write_text(
            json.dumps(envelope_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        package_paths = {
            version: f"{clip_id}/packages/{version}" for version in ("v1", "v2", "v3")
        }
        clips_manifest.append(
            {
                "clip_id": clip_id,
                "seed": clip_seed,
                "sample_rate": 16000,
                "channels": 1,
                "bit_depth": 16,
                "frames": frames,
                "duration_ms": round(frames * 1000 / 16000, 6),
                "mix_wav": f"{clip_id}/mix.wav",
                "envelopes": f"{clip_id}/envelopes.json",
                "packages": package_paths,
                "track_gain_metadata": dict(TRACK_GAIN_DB),
                "source_frame_annotations": source_annotations,
            }
        )
        clip_start = (8.0, 12.0, 6.0)[index]
        clip_end = (780.0, 850.0, 940.0)[index]
        config_clips.append(
            {
                "clip_id": clip_id,
                "source_format": ("v1", "v2", "v3")[index],
                "source_package": package_paths[("v1", "v2", "v3")[index]],
                "start_ms": clip_start,
                "end_ms": clip_end,
                "tracks": {
                    "bass": {
                        "gain": 0.68,
                        "mute": False,
                        "start_ms": 0.0,
                        "end_ms": None,
                    },
                    "drums": {
                        "gain": 0.82,
                        "mute": False,
                        "start_ms": 0.0,
                        "end_ms": None,
                    },
                    "vocal": {
                        "gain": 0.76,
                        "mute": False,
                        "start_ms": 30.0,
                        "end_ms": (720.0, 800.0, 900.0)[index],
                    },
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "generator": "remix_workbench.generate",
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16,
        "clips": clips_manifest,
    }
    (samples_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    replacement_source = samples_root / "clip_001" / "packages" / "v2" / "tracks" / "drums.wav"
    replacement_dir = samples_root / "replacements"
    replacement_dir.mkdir(parents=True, exist_ok=True)
    replacement_path = replacement_dir / "clip_001_drums.wav"
    _, channels, width, frames, raw = read_wav(replacement_source)
    if channels != 1:
        raise RuntimeError("generated v2 drum track must be mono")
    replaced = decode_pcm_bytes(raw, width)
    replacement = (
        np.roll(replaced, 37) * 0.73 + np.roll(replaced, 151) * 0.19
    ).astype(np.int32)
    write_pcm_wav(
        replacement_path,
        replacement,
        sample_rate=16000,
        sample_width=3,
    )

    config = {
        "schema_version": 1,
        "sample_rate": 16000,
        "gain_quantization": 1024,
        "render_frame_length": 512,
        "thread_pool_size": 4,
        "snapshot_interval_fragments": 3,
        "snapshot_interval_clips": 3,
        "defaults": {"start_ms": 0.0, "end_ms": None, "tracks": {}},
        "clips": config_clips,
        "track_overrides": {},
        "conversions": [
            {
                "id": "CONV_V1_TO_V3",
                "clip_id": "clip_000",
                "source_format": "v1",
                "target_format": "v3",
                "source_package": "clip_000/packages/v1",
            },
            {
                "id": "CONV_V3_TO_V2",
                "clip_id": "clip_000",
                "source_format": "v3",
                "target_format": "v2",
                "source_package": "clip_000/packages/v3",
            },
            {
                "id": "CONV_V2_TO_V1",
                "clip_id": "clip_000",
                "source_format": "v2",
                "target_format": "v1",
                "source_package": "clip_000/packages/v2",
            },
        ],
    }
    (configs_root / "remix.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    edits = {
        "schema_version": 1,
        "base_config": "configs/remix.json",
        "edits": [
            {
                "id": "E01_GAIN",
                "op": "set_gain",
                "clip_id": "clip_000",
                "track": "drums",
                "gain": 0.55,
            },
            {
                "id": "E02_MUTE",
                "op": "set_mute",
                "clip_id": "clip_001",
                "track": "bass",
                "mute": True,
            },
            {
                "id": "E03_TRIM",
                "op": "set_trim",
                "clip_id": "clip_002",
                "target": "track",
                "track": "vocal",
                "start_ms": 120.0,
                "end_ms": 800.0,
            },
            {
                "id": "E04_REPLACE",
                "op": "replace_track",
                "clip_id": "clip_001",
                "track": "drums",
                "wav": "replacements/clip_001_drums.wav",
            },
            {
                "id": "E05_SWITCH_V3",
                "op": "set_format_version",
                "clip_id": "clip_000",
                "format_version": "v3",
            },
            {
                "id": "E06_SWITCH_V2",
                "op": "set_format_version",
                "clip_id": "clip_002",
                "format_version": "v2",
            },
            {
                "id": "E07_SWITCH_V3",
                "op": "set_format_version",
                "clip_id": "clip_001",
                "format_version": "v3",
            },
        ],
    }
    (configs_root / "edit_sequence.json").write_text(
        json.dumps(edits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    faults = {
        "schema_version": 1,
        "injections": [
            {
                "id": "INJ001",
                "stage": "parse",
                "kind": "truncate_wav_bytes",
                "clip_id": "clip_001",
                "track": "bass",
                "format_version": "v2",
                "bytes": 11,
                "expected_code": "E_TRACK_FORMAT",
            },
            {
                "id": "INJ002",
                "stage": "parse",
                "kind": "offset_table_out_of_bounds",
                "clip_id": "clip_002",
                "track": "vocal",
                "format_version": "v3",
                "expected_code": "E_TRACK_FORMAT",
            },
            {
                "id": "INJ003",
                "stage": "parse",
                "kind": "frame_count_mismatch",
                "clip_id": "clip_001",
                "track": "drums",
                "format_version": "v2",
                "expected_code": "E_TRACK_FORMAT",
            },
            {
                "id": "INJ004",
                "stage": "gain",
                "kind": "gain_nonfinite",
                "clip_id": "clip_000",
                "track": "drums",
                "value": "NaN",
                "expected_code": "E_MIX_CONFIG",
            },
            {
                "id": "INJ005",
                "stage": "config",
                "kind": "quantization_out_of_range",
                "clip_id": "clip_002",
                "value": 0,
                "expected_code": "E_MIX_CONFIG",
            },
        ],
    }
    (configs_root / "fault_injections.json").write_text(
        json.dumps(faults, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def manifest_digest(root: str | Path) -> str:
    path = Path(root) / "samples" / "manifest.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()
