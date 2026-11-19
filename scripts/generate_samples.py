"""Deterministic synthetic WAV + config + command stream + injection generator.

Run:  python -m scripts.generate_samples --out samples/dataset
Everything derives from fixed NumPy seeds, so repeated runs regenerate
byte-identical WAV and JSON payloads.
"""

import argparse
import json
import os

import numpy as np

from speaker_pipeline.audioio import SAMPLE_RATE, write_wav

SEED = 20260517
SPEAKERS = {
    "spk_a": {"f0": 120.0, "formants": (700.0, 1220.0), "amp": 0.42},
    "spk_b": {"f0": 180.0, "formants": (520.0, 1700.0), "amp": 0.38},
    "spk_c": {"f0": 260.0, "formants": (900.0, 2400.0), "amp": 0.35},
}


def speaker_voice(speaker, n_samples, rng):
    cfg = SPEAKERS[speaker]
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    phase = 2.0 * np.pi * cfg["f0"] * t
    vibrato = 1.0 + 0.01 * np.sin(2.0 * np.pi * 5.0 * t)
    voice = np.sin(phase * vibrato)
    for harmonic in (2, 3):
        voice += (1.0 / harmonic) * np.sin(
            phase * harmonic * vibrato + rng.uniform(0, np.pi)
        )
    for formant in cfg["formants"]:
        voice += 0.3 * np.sin(2.0 * np.pi * formant * t)
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t + rng.uniform(0, 6))
    return cfg["amp"] * voice * (0.7 + 0.3 * envelope)


def add_silence(buffer, duration_ms, rng):
    n = int(SAMPLE_RATE * duration_ms / 1000)
    noise = rng.normal(0.0, 0.004, size=n)
    return np.concatenate([buffer, noise])


def add_speech(buffer, speaker, duration_ms, rng):
    n = int(SAMPLE_RATE * duration_ms / 1000)
    voice = speaker_voice(speaker, n, rng)
    noise = rng.normal(0.0, 0.006, size=n)
    return np.concatenate([buffer, voice + noise])


def add_overlap(buffer, speaker_a, speaker_b, duration_ms, rng):
    n = int(SAMPLE_RATE * duration_ms / 1000)
    mix = (
        speaker_voice(speaker_a, n, rng)
        + speaker_voice(speaker_b, n, rng)
    ) * 0.5
    noise = rng.normal(0.0, 0.006, size=n)
    return np.concatenate([buffer, mix + noise])


def build_file(plan, rng):
    buffer = np.zeros(0, dtype=np.float64)
    for action in plan:
        kind = action[0]
        if kind == "silence":
            buffer = add_silence(buffer, action[1], rng)
        elif kind == "speech":
            buffer = add_speech(buffer, action[1], action[2], rng)
        elif kind == "overlap":
            buffer = add_overlap(buffer, action[1], action[2], action[3], rng)
    return buffer.astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples/dataset")
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    plans = {
        "sample_001.wav": [
            ("silence", 150),
            ("speech", "spk_a", 700),
            ("silence", 200),
            ("speech", "spk_b", 650),
            ("overlap", "spk_a", "spk_b", 400),
            ("silence", 180),
            ("speech", "spk_a", 600),
            ("silence", 120),
        ],
        "sample_002.wav": [
            ("silence", 120),
            ("speech", "spk_b", 600),
            ("silence", 220),
            ("speech", "spk_c", 700),
            ("silence", 160),
            ("overlap", "spk_b", "spk_c", 350),
            ("silence", 200),
            ("speech", "spk_b", 550),
            ("silence", 100),
        ],
        "sample_003.wav": [
            ("silence", 180),
            ("speech", "spk_c", 650),
            ("silence", 240),
            ("speech", "spk_a", 600),
            ("silence", 200),
            ("speech", "spk_c", 500),
            ("silence", 140),
        ],
    }
    for name, plan in plans.items():
        samples = build_file(plan, rng)
        write_wav(os.path.join(out, name), samples)

    config = {
        "frame_length_ms": 25,
        "frame_shift_ms": 10,
        "feature_dim": 12,
        "energy_threshold": 0.01,
        "spectral_flatness_threshold": 0.35,
        "overlap_similarity_threshold": 0.75,
        "cluster_distance_threshold": 18.0,
        "min_speech_ms": 120,
        "min_silence_ms": 90,
        "worker_threads": 4,
        "snapshot_interval": 2,
    }
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, sort_keys=True, indent=2)

    # Commands reference ranges well inside the generated audio.
    commands = {
        "commands": [
            {
                "type": "trim",
                "file": "sample_001.wav",
                "start_ms": 300,
                "end_ms": 600,
            },
            {
                "type": "replace",
                "file": "sample_002.wav",
                "start_ms": 400,
                "end_ms": 900,
                "replace_seed": 4242,
                "frequency_hz": 170.0,
            },
            {"type": "merge_clusters", "cluster_ids": ["SPK-000", "SPK-001"]},
            {
                "type": "split_cluster",
                "source_cluster_id": "SPK-000",
                "member_count": 1,
            },
            {"type": "set_threshold", "cluster_distance_threshold": 22.0},
            {
                "type": "replace",
                "file": "sample_001.wav",
                "start_ms": 90000,
                "end_ms": 99000,
                "injection_id": "INJ-004",
            },
        ]
    }
    with open(os.path.join(out, "commands.json"), "w", encoding="utf-8") as fh:
        json.dump(commands, fh, sort_keys=True, indent=2)

    injections = {
        "injections": [
            {
                "id": "INJ-001",
                "fault": "corrupt_wav",
                "file": "sample_003.wav",
                "stage": "io",
            },
            {
                "id": "INJ-002",
                "fault": "non_finite",
                "file": "sample_002.wav",
                "frame_index": 200,
                "stage": "feature",
            },
            {
                "id": "INJ-003",
                "fault": "zero_threshold",
                "stage": "clustering",
            },
            {
                "id": "INJ-004",
                "fault": "replace_out_of_range",
                "stage": "command",
            },
        ]
    }
    with open(
        os.path.join(out, "injections.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(injections, fh, sort_keys=True, indent=2)
    print("generated dataset at", out)


if __name__ == "__main__":
    main()
