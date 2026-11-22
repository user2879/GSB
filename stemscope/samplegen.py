from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import canonical_json_bytes, hash_directory
from .model import CANONICAL_TRACK_ORDER
from .v3codec import encode_v3
from .wavio import write_multichannel_wav, write_pcm16_wav


DEFAULT_SEED = 20261122
SAMPLE_RATE = 16000
CLIP_COUNT = 4
CLIP_MS = 750


def _envelope(frames: int, attack: int, decay: int) -> np.ndarray:
    attack = max(1, min(attack, frames))
    decay = max(1, min(decay, frames))
    rising = np.linspace(0.0, 1.0, attack, endpoint=False)
    falling = np.exp(np.linspace(0.0, -4.0, decay))
    sustain_length = max(0, frames - attack - decay)
    sustain = np.full(sustain_length, math.exp(-4.0), dtype=np.float64)
    result = np.concatenate([rising, falling, sustain])
    if result.shape[0] < frames:
        result = np.pad(result, (0, frames - result.shape[0]))
    return result[:frames]


def _add_at(target: np.ndarray, start: int, signal: np.ndarray, gain: float = 1.0) -> None:
    if start >= target.shape[0]:
        return
    end = min(target.shape[0], start + signal.shape[0])
    target[start:end] += signal[: end - start] * gain


def _synth_drums(rng: np.random.Generator, frames: int, variation: int) -> np.ndarray:
    t = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    result = np.zeros(frames, dtype=np.float64)
    beat = int(SAMPLE_RATE * 0.25)
    for index, start in enumerate(range(0, frames, beat)):
        local_frames = min(int(SAMPLE_RATE * 0.18), frames - start)
        if local_frames <= 0:
            continue
        local_t = np.arange(local_frames, dtype=np.float64) / SAMPLE_RATE
        kick_freq = 62.0 + 22.0 * np.exp(-local_t * 30.0)
        kick = np.sin(2 * np.pi * np.cumsum(kick_freq) / SAMPLE_RATE)
        kick *= np.exp(-local_t * 22.0)
        _add_at(result, start, kick, 0.9 if index % 2 == 0 else 0.55)
    for index, start in enumerate(range(beat // 2, frames, beat)):
        local_frames = min(int(SAMPLE_RATE * 0.10), frames - start)
        if local_frames <= 0:
            continue
        noise = rng.standard_normal(local_frames)
        local_t = np.arange(local_frames) / SAMPLE_RATE
        snare = 0.58 * noise * np.exp(-local_t * 35.0)
        snare += 0.25 * np.sin(2 * np.pi * 185.0 * local_t) * np.exp(-local_t * 20.0)
        _add_at(result, start, snare, 0.55 + 0.03 * variation)
    hat_interval = max(1, int(SAMPLE_RATE * 0.0625))
    for start in range(0, frames, hat_interval):
        local_frames = min(int(SAMPLE_RATE * 0.025), frames - start)
        noise = rng.standard_normal(local_frames)
        local_t = np.arange(local_frames) / SAMPLE_RATE
        _add_at(result, start, noise * np.exp(-local_t * 90.0) * 0.15, 0.8)
    result *= _envelope(frames, int(SAMPLE_RATE * 0.008), int(SAMPLE_RATE * 0.05))
    return result


def _synth_bass(rng: np.random.Generator, frames: int, variation: int) -> np.ndarray:
    t = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    notes = [55.0, 73.42, 82.41, 61.74, 55.0, 65.41]
    result = np.zeros(frames, dtype=np.float64)
    note_length = int(SAMPLE_RATE * 0.18)
    for index, start in enumerate(range(0, frames, note_length)):
        local_frames = min(note_length, frames - start)
        local_t = np.arange(local_frames, dtype=np.float64) / SAMPLE_RATE
        frequency = notes[(index + variation) % len(notes)]
        phase = 2 * np.pi * frequency * local_t
        tone = np.sin(phase) + 0.32 * np.sin(2 * phase) + 0.16 * np.sin(3 * phase)
        env = np.exp(-local_t * 3.2) * _envelope(local_frames, 40, 180)
        _add_at(result, start, tone * env, 0.8)
    result += rng.standard_normal(frames) * 0.008
    result *= _envelope(frames, int(SAMPLE_RATE * 0.015), int(SAMPLE_RATE * 0.06))
    return result


def _synth_vocals(rng: np.random.Generator, frames: int, variation: int, alternate: bool = False) -> np.ndarray:
    t = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    base = [196.0, 220.0, 246.94, 261.63][(variation + (2 if alternate else 0)) % 4]
    vibrato = 1.0 + 0.012 * np.sin(2 * np.pi * (5.0 + 0.2 * variation) * t)
    phase = 2 * np.pi * np.cumsum(base * vibrato) / SAMPLE_RATE
    result = (
        0.62 * np.sin(phase)
        + 0.24 * np.sin(2.01 * phase + 0.2)
        + 0.12 * np.sin(3.02 * phase + 0.7)
        + 0.06 * np.sin(4.99 * phase)
    )
    syllable = 0.5 + 0.5 * np.sin(2 * np.pi * 2.1 * t + variation * 0.4)
    envelope = _envelope(frames, int(SAMPLE_RATE * 0.035), int(SAMPLE_RATE * 0.08))
    result *= (0.35 + 0.65 * syllable) * envelope
    result += rng.standard_normal(frames) * 0.005
    if alternate:
        result = np.roll(result, int(SAMPLE_RATE * 0.017)) * 0.93
    return result


def _to_pcm16(signal: np.ndarray, peak: float = 0.86) -> np.ndarray:
    max_abs = float(np.max(np.abs(signal))) if signal.size else 0.0
    if max_abs <= 0:
        return np.zeros(signal.shape[0], dtype="<i2")
    normalized = signal / max_abs * peak
    return np.clip(np.rint(normalized * 32767.0), -32768, 32767).astype("<i2")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_v2_package(clip_root: Path, tracks: dict[str, np.ndarray], sample_rate: int) -> None:
    target = clip_root / "packages" / "v2"
    entries = []
    for channel, name in enumerate(CANONICAL_TRACK_ORDER):
        filename = f"{name}.wav"
        write_pcm16_wav(target / filename, tracks[name], sample_rate)
        entries.append(
            {
                "name": name,
                "channel": channel,
                "file": filename,
                "frame_count": int(tracks[name].shape[0]),
            }
        )
    manifest = {
        "format": "stemscope/v2",
        "version": 2,
        "sample_rate": sample_rate,
        "frame_count": int(tracks[CANONICAL_TRACK_ORDER[0]].shape[0]),
        "tracks": entries,
    }
    _write_json(target / "manifest.json", manifest)


def _write_v3_package(clip_root: Path, tracks: dict[str, np.ndarray], sample_rate: int, clip_id: str) -> None:
    from .model import Track

    track_objects = [
        Track(
            name=name,
            pcm=tracks[name],
            sample_rate=sample_rate,
            source_format="v3",
        )
        for name in CANONICAL_TRACK_ORDER
    ]
    raw = encode_v3(track_objects, sample_rate)
    target = clip_root / "packages" / "v3"
    target.mkdir(parents=True, exist_ok=True)
    (target / "stems.v3").write_bytes(raw)


def generate_samples(output: str | Path, seed: int = DEFAULT_SEED, force: bool = True) -> dict[str, Any]:
    root = Path(output)
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    frames = int(SAMPLE_RATE * CLIP_MS / 1000)
    clip_ids: list[str] = []
    for clip_index in range(CLIP_COUNT):
        rng = np.random.Generator(np.random.PCG64(seed + clip_index * 1009))
        clip_id = f"seg_{clip_index:03d}"
        clip_ids.append(clip_id)
        clip_root = root / "segments" / clip_id
        drums = _to_pcm16(_synth_drums(rng, frames, clip_index), 0.82)
        bass = _to_pcm16(_synth_bass(rng, frames, clip_index), 0.76)
        vocals = _to_pcm16(_synth_vocals(rng, frames, clip_index), 0.73)
        alt_vocals = _to_pcm16(_synth_vocals(rng, frames, clip_index, alternate=True), 0.69)
        tracks = {"bass": bass, "drums": drums, "vocals": vocals}
        source_root = clip_root / "source"
        for name in CANONICAL_TRACK_ORDER:
            write_pcm16_wav(source_root / f"{name}.wav", tracks[name], SAMPLE_RATE)
        write_pcm16_wav(source_root / "alt_vocals.wav", alt_vocals, SAMPLE_RATE)
        mix = (
            bass.astype(np.int32) * 0.32
            + drums.astype(np.int32) * 0.42
            + vocals.astype(np.int32) * 0.36
        )
        mix = np.clip(np.rint(mix), -32768, 32767).astype("<i2")
        write_pcm16_wav(clip_root / "mix.wav", mix, SAMPLE_RATE)
        v1_target = clip_root / "packages" / "v1"
        channel_matrix = np.stack([tracks[name] for name in CANONICAL_TRACK_ORDER], axis=0)
        write_multichannel_wav(v1_target / "stems.wav", channel_matrix, SAMPLE_RATE)
        _write_json(
            Path(str(v1_target / "stems.wav") + ".sidecar.json"),
            {
                "schema": "stemscope/sidecar/v1",
                "channel_names": list(CANONICAL_TRACK_ORDER),
                "provenance": {"generator": "stemscope.samplegen", "seed": seed, "clip": clip_id},
            },
        )
        _write_v2_package(clip_root, tracks, SAMPLE_RATE)
        _write_v3_package(clip_root, tracks, SAMPLE_RATE, clip_id)

    config = {
        "schema_version": 1,
        "sample_rate": SAMPLE_RATE,
        "gain_quantization_steps": 1000,
        "render_frames": 2048,
        "workers": 4,
        "interaction_budget_ms": 2000,
        "snapshot_interval_segments": 2,
        "format_version": "v1",
        "tracks": {
            "bass": {"gain": 0.9073, "muted": False, "start_ms": 0, "end_ms": CLIP_MS},
            "drums": {"gain": 0.7531, "muted": False, "start_ms": 0, "end_ms": CLIP_MS},
            "vocals": {"gain": 1.0137, "muted": False, "start_ms": 0, "end_ms": CLIP_MS},
        },
    }
    _write_json(root / "remix_config.json", config)

    interactions = {
        "schema": "stemscope/interactions/v1",
        "commands": [
            {"id": "cmd_001", "type": "set_gain", "track": "vocals", "gain": 1.25},
            {"id": "cmd_002", "type": "set_mute", "track": "drums", "muted": True},
            {"id": "cmd_003", "type": "set_crop", "track": "bass", "start_ms": 60, "end_ms": 690},
            {
                "id": "cmd_004",
                "type": "replace_track",
                "clip": "seg_001",
                "track": "vocals",
                "file": "segments/seg_001/source/alt_vocals.wav",
            },
            {"id": "cmd_005", "type": "switch_format", "format": "v3"},
            {"id": "cmd_006", "type": "set_quantization", "steps": 2000},
            {"id": "cmd_007", "type": "set_gain", "track": "bass", "gain": 0.65},
            {"id": "cmd_008", "type": "switch_format", "format": "v2"},
            {"id": "cmd_009", "type": "set_mute", "track": "drums", "muted": False},
            {
                "id": "cmd_010",
                "type": "set_gain",
                "track": "vocals",
                "gain": 1.40,
                "budget_probe": {"simulated_compute_ms": 3000},
            },
        ],
    }
    _write_json(root / "interactions.json", interactions)

    injection = {
        "schema": "stemscope/injections/v1",
        "cases": [
            {
                "id": "corrupt_v1_wav",
                "format": "v1",
                "clip_id": "seg_000",
                "operation": "patch_bytes",
                "path": "segments/seg_000/packages/v1/stems.wav",
                "offset": 40,
                "data_hex": "ffffffff",
                "expected_code": "E_TRACK_PARSE_FAILED",
            },
            {
                "id": "v3_offset_oob",
                "format": "v3",
                "clip_id": "seg_000",
                "operation": "patch_v3_record_offset",
                "path": "segments/seg_000/packages/v3/stems.v3",
                "record_index": 1,
                "offset": 4294967280,
                "expected_code": "E_TRACK_PARSE_FAILED",
            },
            {
                "id": "v2_frame_mismatch",
                "format": "v2",
                "clip_id": "seg_000",
                "operation": "patch_json",
                "path": "segments/seg_000/packages/v2/manifest.json",
                "json_path": ["tracks", 1, "frame_count"],
                "value_delta": 1,
                "expected_code": "E_TRACK_PARSE_FAILED",
            },
        ],
    }
    _write_json(root / "injection_manifest.json", injection)

    manifest = {
        "schema": "stemscope/sample-manifest/v1",
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "clip_ms": CLIP_MS,
        "frames_per_clip": frames,
        "clip_count": CLIP_COUNT,
        "clip_ids": clip_ids,
        "track_order": list(CANONICAL_TRACK_ORDER),
        "generator": "scripts/generate_samples.py",
        "content_sha256": hash_directory(root),
    }
    _write_json(root / "generator_manifest.json", manifest)
    return manifest


def load_sample_manifest(data_root: str | Path) -> dict[str, Any]:
    path = Path(data_root) / "generator_manifest.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
