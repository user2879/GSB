"""Deterministic synthetic WAV scenarios for tests and the demo workbench."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG, load_config
from .segment import detect_speech_segments
from .util import atomic_write_json
from .wavio import write_wav


@dataclass(frozen=True)
class Timbre:
    name: str
    f0: float
    formants: tuple[float, float, float]
    syllable_rate: float
    brightness: float = 1.0


TIMBRES: dict[str, Timbre] = {
    "alpha": Timbre("alpha", 118.0, (720.0, 1220.0, 2540.0), 4.1, 1.0),
    "beta": Timbre("beta", 212.0, (510.0, 1780.0, 2680.0), 5.7, 0.9),
    "gamma": Timbre("gamma", 158.0, (860.0, 1420.0, 2380.0), 3.4, 1.15),
    "delta": Timbre("delta", 142.0, (650.0, 1050.0, 2200.0), 4.7, 1.05),
}


def synthesize_segment(sample_rate: int, duration_ms: float, timbre: Timbre, seed: int) -> np.ndarray:
    sample_count = int(round(sample_rate * duration_ms / 1000.0))
    rng = np.random.RandomState(seed)
    t = np.arange(sample_count, dtype=np.float64) / sample_rate
    signal = np.zeros(sample_count, dtype=np.float64)
    max_harmonic = min(42, int((sample_rate / 2.0 - 200.0) / timbre.f0))
    for harmonic in range(1, max_harmonic + 1):
        frequency = harmonic * timbre.f0
        spectral_gain = 0.035 / math.sqrt(harmonic)
        for center, bandwidth in zip(timbre.formants, (150.0, 220.0, 280.0)):
            spectral_gain += 0.085 * timbre.brightness * math.exp(
                -0.5 * ((frequency - center) / bandwidth) ** 2
            )
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        signal += spectral_gain * np.sin(2.0 * math.pi * frequency * t + phase)

    vibrato = 1.0 + 0.006 * np.sin(2.0 * math.pi * 5.2 * t + 0.31)
    syllable = 0.74 + 0.26 * np.sin(2.0 * math.pi * timbre.syllable_rate * t - 0.7)
    noise = rng.normal(0.0, 0.006, sample_count)
    envelope = syllable * vibrato
    fade_samples = min(sample_count // 5, int(0.025 * sample_rate))
    if fade_samples > 1:
        fade = np.linspace(0.0, 1.0, fade_samples, endpoint=True)
        envelope[:fade_samples] *= fade
        envelope[-fade_samples:] *= fade[::-1]
    signal = signal * envelope + noise
    peak = float(np.max(np.abs(signal))) if sample_count else 0.0
    if peak > 0:
        signal = signal * (0.24 / peak)
    return np.clip(np.round(signal * 32767.0), -32768, 32767).astype(np.int16)


def silence(sample_rate: int, duration_ms: float) -> np.ndarray:
    return np.zeros(int(round(sample_rate * duration_ms / 1000.0)), dtype=np.int16)


def build_file(
    sample_rate: int,
    parts: list[tuple[str, float, int]],
    gap_ms: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    chunks: list[np.ndarray] = [silence(sample_rate, 120.0)]
    cursor = len(chunks[0])
    labels: list[dict[str, Any]] = []
    for timbre_name, duration_ms, seed in parts:
        samples = synthesize_segment(sample_rate, duration_ms, TIMBRES[timbre_name], seed)
        labels.append(
            {
                "start_sample": cursor,
                "end_sample": cursor + len(samples),
                "speaker_id": f"spk_{timbre_name}",
            }
        )
        chunks.append(samples)
        cursor += len(samples)
        gap = silence(sample_rate, gap_ms)
        chunks.append(gap)
        cursor += len(gap)
    chunks.append(silence(sample_rate, 140.0))
    return np.concatenate(chunks).astype(np.int16), labels


def generate_demo(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    files: dict[str, list[dict[str, Any]]] = {}
    file_specs: dict[str, list[tuple[str, float, int]]] = {
        "voice_alpha.wav": [("alpha", 520.0, 11), ("alpha", 470.0, 12), ("alpha", 560.0, 13)],
        "voice_beta.wav": [("beta", 440.0, 21), ("beta", 510.0, 22), ("beta", 480.0, 23)],
        "voice_gamma.wav": [("gamma", 500.0, 31), ("gamma", 450.0, 32), ("gamma", 530.0, 33)],
        "voice_mixed.wav": [("alpha", 460.0, 41), ("beta", 480.0, 42), ("alpha", 500.0, 43)],
    }
    generated: dict[str, np.ndarray] = {}
    for file_name, parts in file_specs.items():
        samples, labels = build_file(sample_rate, parts, gap_ms=150.0)
        write_wav(audio_dir / file_name, samples, sample_rate=sample_rate)
        generated[file_name] = samples
        files[file_name] = labels

    config = load_config(
        overlay={
            "runtime": {
                "first_response_budget_ms": 1,
                "first_response_scheduling_tolerance_ms": 20.0,
                "max_workers": 4,
                "snapshot_after_each_command": True,
                "snapshot_keep": 5,
            },
            "assignment": {"similarity_threshold": 0.92, "rbf_gamma": 8.0},
        }
    )
    labels_payload = {"files": files}
    config_path = out_dir / "config.json"
    labels_path = out_dir / "labels.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    labels_path.write_text(json.dumps(labels_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    alpha_intervals = detect_speech_segments(generated["voice_alpha.wav"], sample_rate, config["segmentation"])
    first = alpha_intervals[0]
    split_at = (first["start_sample"] + first["end_sample"]) // 2
    replacement_length = first["end_sample"] - split_at
    replacement = synthesize_segment(
        sample_rate,
        replacement_length / sample_rate * 1000.0,
        Timbre("alpha_shift", 126.0, (760.0, 1280.0, 2600.0), 4.4, 1.05),
        seed=901,
    )
    write_wav(variants_dir / "alpha_split_replacement.wav", replacement, sample_rate=sample_rate)

    beta_intervals = detect_speech_segments(generated["voice_beta.wav"], sample_rate, config["segmentation"])
    boundary_target = beta_intervals[1]["end_sample"] - 240 if len(beta_intervals) > 1 else 0
    commands = {
        "schema_version": 1,
        "commands": [
            {
                "op": "split_segment",
                "file": "voice_alpha.wav",
                "ordinal": 0,
                "at_sample": split_at,
            },
            {
                "op": "replace_segment_audio",
                "file": "voice_alpha.wav",
                "ordinal": 1,
                "source": "variants/alpha_split_replacement.wav",
            },
            {"op": "set_quantization_tier", "dimension": "f0_mean", "tier": 4},
            {"op": "set_similarity_threshold", "value": 0.95},
            {"op": "merge_segments", "file": "voice_alpha.wav", "ordinals": [0, 1]},
            {
                "op": "set_segment_boundary",
                "file": "voice_beta.wav",
                "ordinal": 1,
                "edge": "end",
                "sample": boundary_target,
            },
            {"op": "noop"},
        ],
    }
    atomic_write_json(out_dir / "commands.json", commands)
    return {
        "out_dir": str(out_dir),
        "audio_files": sorted(file_specs),
        "config": str(config_path),
        "labels": str(labels_path),
        "commands": str(out_dir / "commands.json"),
        "split_at_sample": split_at,
        "command_count": len(commands["commands"]),
    }


def write_minimal_scenario(root: Path, *, invalid_audio: bool = False) -> dict[str, Any]:
    """Small scenario used by unit tests."""
    root.mkdir(parents=True, exist_ok=True)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    samples, labels = build_file(
        16000,
        [("alpha", 360.0, 101), ("beta", 420.0, 102), ("alpha", 380.0, 103)],
        gap_ms=160.0,
    )
    write_wav(audio_dir / "sample.wav", samples)
    if invalid_audio:
        (audio_dir / "broken.wav").write_bytes(b"RIFF\x10\x00\x00\x00WAVEbad!")
    config = load_config(overlay={"runtime": {"first_response_budget_ms": 0, "max_workers": 3}})
    config_path = root / "config.json"
    labels_path = root / "labels.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    labels_path.write_text(json.dumps({"files": {"sample.wav": labels}}, indent=2) + "\n", encoding="utf-8")
    return {
        "root": str(root),
        "audio_dir": str(audio_dir),
        "config": str(config_path),
        "labels": str(labels_path),
        "labels_data": {"files": {"sample.wav": labels}},
    }
