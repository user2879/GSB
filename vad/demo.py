"""Deterministic local demo-audio and injection-manifest generator."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .features import frame_geometry
from .util import atomic_write_json


SAMPLE_RATE = 16000


def generate_demo(
    data_dir: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    *,
    seed: int = 20261120,
) -> dict[str, Any]:
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    frame_samples, _hop_samples = frame_geometry(config.raw, SAMPLE_RATE)
    specifications = _sample_specifications(frame_samples)
    manifest_samples: list[dict[str, Any]] = []
    for specification in specifications:
        name = specification["file"]
        rng = np.random.default_rng(_seed_for(seed, name))
        samples = _synthesize(specification, rng, frame_samples)
        _write_pcm16_wav(output_dir / name, samples)
        manifest_samples.append(
            {
                "file": name,
                "shard_count": 1,
                "expected_segments": list(specification["expected_segments"]),
                "scale_profile": specification["scale_profile"],
            }
        )
    manifest = {
        "schema_version": "1.0",
        "generator": {
            "name": "vad-demo-generator",
            "seed": int(seed),
            "sample_rate": SAMPLE_RATE,
            "deterministic": True,
        },
        "noise_reference_samples": ["noise_only.wav"],
        "samples": sorted(manifest_samples, key=lambda item: item["file"]),
    }
    atomic_write_json(Path(manifest_path), manifest)
    return manifest


def _seed_for(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _sample_specifications(frame_samples: int) -> list[dict[str, Any]]:
    return [
        {
            "file": "dense_six.wav",
            "duration": 15.0,
            "noise_sigma": 68.0,
            "segments": [(1.2, 0.8, 156.0), (3.2, 0.8, 188.0), (5.2, 0.8, 142.0),
                          (7.2, 0.8, 205.0), (9.2, 0.8, 174.0), (11.2, 0.8, 162.0)],
            "spikes": [2.75, 6.65, 12.8],
            "expected_segments": [6, 6],
            "scale_profile": "medium",
        },
        {
            "file": "long_sixteen.wav",
            "duration": 200.0,
            "noise_sigma": 52.0,
            "segments": [(7.0, 1.2, 132.0), (19.0, 1.15, 168.0),
                          (31.0, 1.25, 145.0), (43.0, 1.1, 196.0),
                          (55.0, 1.2, 154.0), (67.0, 1.15, 179.0),
                          (79.0, 1.2, 137.0), (91.0, 1.1, 211.0),
                          (103.0, 1.2, 164.0), (115.0, 1.15, 151.0),
                          (127.0, 1.2, 173.0), (139.0, 1.1, 142.0),
                          (151.0, 1.2, 189.0), (163.0, 1.15, 157.0),
                          (175.0, 1.2, 201.0), (187.0, 1.1, 166.0)],
            "spikes": [13.0, 37.0, 73.0, 109.0, 145.0, 181.0, 196.5],
            "expected_segments": [16, 16],
            "scale_profile": "large",
        },
        {
            "file": "medium_three.wav",
            "duration": 12.0,
            "noise_sigma": 61.0,
            "segments": [(1.5, 1.0, 148.0), (4.6, 1.0, 183.0), (8.0, 1.0, 126.0)],
            "spikes": [3.35, 6.4, 10.4],
            "expected_segments": [3, 3],
            "scale_profile": "medium",
        },
        {
            "file": "noise_only.wav",
            "duration": 3.0,
            "noise_sigma": 80.0,
            "segments": [],
            "spikes": [],
            "expected_segments": [0, 0],
            "scale_profile": "small",
        },
        {
            "file": "padded_single.wav",
            "duration": 2.005,
            "noise_sigma": 72.0,
            "segments": [(0.55, 0.62, 171.0)],
            "spikes": [1.55],
            "expected_segments": [1, 1],
            "scale_profile": "small",
        },
        {
            "file": "short_two.wav",
            "duration": 2.0,
            "noise_sigma": 76.0,
            "segments": [(0.35, 0.28, 158.0), (1.25, 0.30, 192.0)],
            "spikes": [0.95],
            "expected_segments": [2, 2],
            "scale_profile": "small",
        },
    ]


def _synthesize(
    specification: dict[str, Any], rng: np.random.Generator, frame_samples: int
) -> np.ndarray:
    sample_count = int(round(float(specification["duration"]) * SAMPLE_RATE))
    samples = rng.normal(0.0, float(specification["noise_sigma"]), sample_count)
    for start_seconds, duration_seconds, f0 in specification["segments"]:
        start = int(round(start_seconds * SAMPLE_RATE))
        length = int(round(duration_seconds * SAMPLE_RATE))
        samples[start : start + length] += _harmonic_voice(length, float(f0), rng)
    for spike_seconds in specification["spikes"]:
        start = int(round(float(spike_seconds) * SAMPLE_RATE))
        length = frame_samples
        burst = rng.normal(0.0, 30000.0, length)
        samples[start : start + length] += burst
    return np.clip(np.rint(samples), -32768, 32767).astype(np.int16)


def _harmonic_voice(length: int, f0: float, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / float(SAMPLE_RATE)
    glide = 1.0 + 0.012 * np.sin(2.0 * np.pi * 0.9 * time + float(rng.uniform(0, 2 * np.pi)))
    instantaneous = f0 * glide
    phase = 2.0 * np.pi * np.cumsum(instantaneous) / float(SAMPLE_RATE)
    signal = np.zeros(length, dtype=np.float64)
    for harmonic in range(1, 22):
        if f0 * harmonic >= 7600.0:
            break
        signal += np.sin(harmonic * phase + 0.17 * harmonic) / (
            harmonic ** 1.18
        )
    peak = float(np.max(np.abs(signal))) or 1.0
    signal = signal / peak
    attack = min(int(0.045 * SAMPLE_RATE), max(1, length // 5))
    release = min(int(0.060 * SAMPLE_RATE), max(1, length // 5))
    envelope = np.ones(length, dtype=np.float64)
    envelope[:attack] = np.sin(0.5 * np.pi * np.arange(attack) / attack) ** 2
    envelope[-release:] = np.cos(
        0.5 * np.pi * np.arange(release) / release
    ) ** 2
    sustain_time = np.arange(length) / float(SAMPLE_RATE)
    envelope *= 1.0 + 0.035 * np.sin(2.0 * np.pi * 4.3 * sustain_time)
    amplitude = float(rng.uniform(4300.0, 5200.0))
    return amplitude * signal * envelope


def _write_pcm16_wav(path: Path, samples: np.ndarray) -> None:
    payload = np.asarray(samples, dtype="<i2").tobytes(order="C")
    fmt = struct.pack(
        "<HHIIHH", 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16
    )
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    blob = b"RIFF" + struct.pack("<I", len(body)) + body
    path.write_bytes(blob)
