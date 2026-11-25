"""Local deterministic command-word corpus synthesis.

No recording, network resource, or process-global RNG is used.  Every WAV is a
16 kHz, mono, signed 16-bit PCM file generated from an integer seed plus a
stable textual variant key.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fingerprint import derive_int
from .wavio import write_pcm16_wav

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class WordProfile:
    command_word: str
    syllables: int
    f1: float
    f2: float
    f3: float
    fricative: float
    confusion_with: str | None


WORD_PROFILES = (
    WordProfile("LING", 1, 310.0, 2210.0, 2860.0, 0.10, "LIN"),
    WordProfile("LIN", 1, 330.0, 2110.0, 2810.0, 0.07, "LING"),
    WordProfile("MA", 1, 745.0, 1110.0, 2450.0, 0.025, "NA"),
    WordProfile("NA", 1, 725.0, 1210.0, 2480.0, 0.035, "MA"),
    WordProfile("SASA", 2, 525.0, 1610.0, 2580.0, 0.52, "SHASHA"),
    WordProfile("SHASHA", 2, 505.0, 1510.0, 2520.0, 0.57, "SASA"),
)


def _profile_voice(profile: WordProfile, seed: int, variant_key: str, *, negative: bool) -> np.ndarray:
    rng = np.random.default_rng(derive_int(seed, f"voice|{profile.command_word}|{variant_key}|{negative}"))
    syllable_duration = 0.30 + rng.uniform(-0.018, 0.018)
    duration = profile.syllables * syllable_duration + 0.08
    count = int(round(duration * SAMPLE_RATE))
    samples = np.zeros(count, dtype=np.float64)
    for syllable in range(profile.syllables):
        start = int((0.04 + syllable * syllable_duration) * SAMPLE_RATE)
        length = int(syllable_duration * SAMPLE_RATE)
        stop = min(count, start + length)
        local_count = stop - start
        if local_count <= 0:
            continue
        local_t = np.arange(local_count, dtype=np.float64) / SAMPLE_RATE
        attack = max(1, int(0.030 * SAMPLE_RATE))
        release = max(1, int(0.060 * SAMPLE_RATE))
        envelope = np.ones(local_count, dtype=np.float64)
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
        envelope[-release:] *= np.linspace(1.0, 0.0, release)
        wobble = 1.0 + 0.006 * np.sin(2.0 * np.pi * (2.4 + syllable) * local_t)
        f1 = profile.f1 * wobble + rng.uniform(-5.0, 5.0)
        f2 = profile.f2 * wobble + rng.uniform(-14.0, 14.0)
        f3 = profile.f3 + rng.uniform(-22.0, 22.0)
        vowel = (
            0.55 * np.sin(2.0 * np.pi * f1 * local_t)
            + 0.34 * np.sin(2.0 * np.pi * f2 * local_t + 0.23)
            + 0.16 * np.sin(2.0 * np.pi * f3 * local_t + 0.51)
            + 0.08 * np.sin(2.0 * np.pi * (2.0 * f1) * local_t + 0.11)
        )
        vowel *= envelope
        consonant_length = max(1, int((0.045 + profile.fricative * 0.07) * SAMPLE_RATE))
        consonant_stop = min(local_count, consonant_length)
        noise = rng.normal(0.0, 1.0, consonant_stop)
        if consonant_stop > 4:
            noise = np.diff(noise, prepend=noise[0])
        noise *= np.linspace(0.08, 1.0, consonant_stop) ** 0.5
        vowel[:consonant_stop] += profile.fricative * 0.42 * noise
        gain = 0.33 + 0.07 * rng.uniform(-1.0, 1.0)
        samples[start:stop] += gain * vowel
    if negative:
        noise = rng.normal(0.0, 0.10, count)
        envelope = np.sin(np.linspace(0.0, np.pi, count)) ** 2
        samples = 0.40 * samples + noise * envelope
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0.92:
        samples *= 0.92 / peak
    return np.clip(np.rint(samples * 32767.0), -32768, 32767).astype(np.int16)


def write_sample_config(path: Path, *, seed: int, version: str = "demo-v1") -> dict[str, Any]:
    words = []
    for profile in WORD_PROFILES:
        templates = [
            {
                "template_id": f"T{index:02d}",
                "path": f"templates/{profile.command_word}/T{index:02d}.wav",
            }
            for index in range(1, 4)
        ]
        words.append({"command_word": profile.command_word, "templates": templates})
    config = {
        "schema_version": 1,
        "version": version,
        "seed": seed,
        "paths": {"templates": "templates", "clips": "clips", "labels": "labels.jsonl"},
        "audio": {"sample_rate": 16000, "channels": 1, "bit_depth": 16},
        "features": {
            "frame_length": 256,
            "frame_shift": 128,
            "dimension": 12,
            "cepstral_coefficients": 4,
        },
        "projection": {"seed": seed, "seed_version": 1},
        "quantization": {
            "profile": "q12",
            "fractional_bits": 8,
            "allowed_profiles": ["q8", "q12", "q16"],
            "allowed_fractional_bits": [4, 6, 8, 10],
            "reference_profile": "q16",
        },
        "thresholds": {
            "reject_permille": 900,
            "confusion_candidate_permille": 850,
            "cluster_merge_permille": 940,
        },
        "registration": {
            "template_min": 2,
            "max_workers": 4,
            "comparison_budget": 500000,
            "overlap_candidates": 3,
            "representative_rule": "composite",
        },
        "annotations": {
            "confusion_pairs": [["LING", "LIN"], ["MA", "NA"], ["SASA", "SHASHA"]]
        },
        "words": words,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config


def generate_demo(out: Path, *, seed: int = 20261126, positive_clips: int = 4, negative_clips: int = 4) -> dict[str, Any]:
    """Generate a complete local demo corpus and its registration config."""

    if out.exists():
        for child in out.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    config = write_sample_config(out / "config.json", seed=seed)
    template_paths: dict[tuple[str, str], Path] = {}
    for profile in WORD_PROFILES:
        for template_index in range(1, 4):
            template_id = f"T{template_index:02d}"
            path = out / "templates" / profile.command_word / f"{template_id}.wav"
            samples = _profile_voice(profile, seed, f"template-{template_index}", negative=False)
            write_pcm16_wav(path, samples)
            template_paths[(profile.command_word, template_id)] = path

    labels: list[dict[str, Any]] = []
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for profile in WORD_PROFILES:
        for clip_index in range(1, positive_clips + 1):
            clip_id = f"pos_{profile.command_word.lower()}_{clip_index:02d}"
            path = clips_dir / f"{clip_id}.wav"
            samples = _profile_voice(profile, seed, f"clip-{clip_index}", negative=False)
            write_pcm16_wav(path, samples)
            labels.append(
                {
                    "clip_id": clip_id,
                    "path": f"clips/{path.name}",
                    "true_command": profile.command_word,
                    "confusion_with": profile.confusion_with,
                    "kind": "positive",
                    "seed": seed,
                    "variant_index": clip_index,
                }
            )
    for clip_index in range(1, negative_clips + 1):
        clip_id = f"neg_{clip_index:02d}"
        path = clips_dir / f"{clip_id}.wav"
        rng = np.random.default_rng(derive_int(seed, f"negative-{clip_index}"))
        duration = int(0.62 * SAMPLE_RATE)
        noise = rng.normal(0.0, 0.055, duration)
        envelope = np.sin(np.linspace(0.0, np.pi, duration)) ** 2
        samples = np.clip(np.rint(noise * envelope * 32767.0), -32768, 32767).astype(np.int16)
        write_pcm16_wav(path, samples)
        labels.append(
            {
                "clip_id": clip_id,
                "path": f"clips/{path.name}",
                "true_command": "NEGATIVE",
                "confusion_with": None,
                "kind": "negative",
                "seed": seed,
                "variant_index": clip_index,
            }
        )
    with (out / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "template_count": len(template_paths),
        "clip_count": len(labels),
        "config": str((out / "config.json").relative_to(out)),
        "labels": "labels.jsonl",
    }
    (out / "corpus_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def synthesize_word_audio(
    out: Path,
    profile: WordProfile,
    *,
    seed: int,
    variant_key: str,
    negative: bool = False,
) -> Path:
    """Helper used by tests and edit scripts to create one deterministic WAV."""

    out.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16_wav(out, _profile_voice(profile, seed, variant_key, negative=negative))
    return out
