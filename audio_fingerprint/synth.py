from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import fingerprint
from .wavio import write_wav_from_levels


SYNTH_VERSION = "det-melody-percussion-v1"
FIXED_MASTER_SEED = 0xAF9408
PENTATONIC_HZ = (261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25)
SYNTH_CONFIG = {
    "synth_version": SYNTH_VERSION,
    "note_interval_samples": 1536,
    "hit_interval_samples": 768,
    "pentatonic_hz": PENTATONIC_HZ,
    "master_seed": FIXED_MASTER_SEED,
}


def derive_rng_seed(clip_seed: int, material_fingerprint: str | None = None) -> int:
    if material_fingerprint is None:
        material_fingerprint = fingerprint(SYNTH_CONFIG, "synthcfg_")
    material = int(hashlib.sha256(material_fingerprint.encode()).hexdigest()[:16], 16)
    sequence = np.random.SeedSequence([FIXED_MASTER_SEED, int(clip_seed) & 0xFFFFFFFF, material & 0xFFFFFFFF])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def synthesize_levels(seed: int, sample_count: int = 12288, sample_rate: int = 11025) -> np.ndarray:
    """Synthesize deterministic melody notes and percussion transients.

    Returned values are integer amplitude levels on an exact power-of-two
    grid (-128..127). Encoding those same levels at 8/16/24/32 bits decodes to
    exactly the same float samples.
    """

    generation_fingerprint = fingerprint(
        {**SYNTH_CONFIG, "requested_sample_count": sample_count, "sample_rate": sample_rate},
        "gencfg_",
    )
    rng_seed = derive_rng_seed(seed, generation_fingerprint)
    py_random = random.Random(rng_seed)
    noise_rng = np.random.default_rng(np.random.SeedSequence(rng_seed ^ 0x9E3779B97F4A7C15))

    output = np.zeros(sample_count, dtype=np.float64)
    note_interval = 1536
    note_length = note_interval
    attack = max(1, int(0.008 * sample_rate))

    for note_index in range(8):
        start = note_index * note_interval
        if start >= sample_count:
            break
        frequency = PENTATONIC_HZ[py_random.randrange(len(PENTATONIC_HZ))]
        if note_index % 4 == 0:
            frequency /= 2.0
        length = min(note_length, sample_count - start)
        time_axis = np.arange(length, dtype=np.float64) / sample_rate
        envelope = np.exp(-3.2 * time_axis / max(note_length / sample_rate, 1e-6))
        attack_samples = min(attack, length)
        envelope[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)
        tone = (
            0.62 * np.sin(2.0 * np.pi * frequency * time_axis)
            + 0.24 * np.sin(2.0 * np.pi * frequency * 2.0 * time_axis)
            + 0.10 * np.sin(2.0 * np.pi * frequency * 3.01 * time_axis)
        )
        output[start : start + length] += 0.34 * tone * envelope

    hit_interval = 768
    for hit_index, start in enumerate(range(0, sample_count, hit_interval)):
        length = min(int(0.055 * sample_rate), sample_count - start)
        if length <= 1:
            continue
        time_axis = np.arange(length, dtype=np.float64) / sample_rate
        noise = noise_rng.standard_normal(length)
        decay = np.exp(-55.0 * time_axis)
        click = np.zeros(length, dtype=np.float64)
        click[: min(16, length)] += np.linspace(0.9, 0.1, min(16, length))
        accent = 1.15 if hit_index % 4 == 0 else 0.82
        output[start : start + length] += accent * 0.24 * noise * decay + 0.12 * click

    output += 0.035 * np.sin(2.0 * np.pi * 130.81 * np.arange(sample_count) / sample_rate) * np.exp(
        -0.8 * np.arange(sample_count) / sample_rate
    )
    output = np.tanh(output * 1.05)
    levels = np.rint(output * 127.0).astype(np.int16)
    return np.clip(levels, -128, 127)


def levels_content_fingerprint(levels: np.ndarray, seed: int) -> str:
    return fingerprint(
        {
            "synth_version": SYNTH_VERSION,
            "seed": int(seed),
            "sample_count": int(levels.shape[0]),
            "levels_sha256": hashlib.sha256(np.ascontiguousarray(levels).tobytes()).hexdigest(),
        },
        "content_",
    )


def render_source_wav(
    path: str | Path,
    levels: np.ndarray,
    source_sample_rate: int,
    bit_depth: int,
    internal_sample_rate: int = 11025,
) -> dict[str, Any]:
    if source_sample_rate % internal_sample_rate != 0:
        raise ValueError("deterministic generator requires an integer source/internal rate ratio")
    factor = source_sample_rate // internal_sample_rate
    source_levels = np.repeat(np.asarray(levels, dtype=np.int16), factor)
    return write_wav_from_levels(path, source_levels, source_sample_rate, bit_depth)


def generate_clip_file(spec: dict[str, Any], data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    clip_id = spec["clip_id"]
    relative_path = spec.get("path") or f"clips/{clip_id}.wav"
    path = data_dir / relative_path
    sample_count = int(spec.get("internal_sample_count", 12288))
    internal_rate = int(spec.get("internal_sample_rate", 11025))
    levels = synthesize_levels(int(spec["seed"]), sample_count, internal_rate)
    start_sample = int(spec.get("start_sample", 0))
    excerpt_count = spec.get("excerpt_sample_count")
    if excerpt_count is not None:
        end_sample = min(sample_count, start_sample + int(excerpt_count))
        rendered_levels = levels[start_sample:end_sample]
    else:
        rendered_levels = levels
    info = render_source_wav(
        path,
        rendered_levels,
        int(spec["source_sample_rate"]),
        int(spec["bit_depth"]),
        internal_rate,
    )
    result = dict(spec)
    result["path"] = relative_path
    result["internal_sample_count"] = sample_count
    result["rendered_sample_count"] = int(rendered_levels.shape[0])
    result["content_fingerprint"] = levels_content_fingerprint(rendered_levels, int(spec["seed"]))
    result["source_sample_count"] = info["sample_count"]
    return result
