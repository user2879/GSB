"""Deterministic synthesis of WAV audio and per-shard AED result packages.

Everything is seeded. Re-running the generator produces byte-identical WAV and
shard JSON files. The generator deliberately introduces probability
discontinuities at shard boundaries so the reconstruction stages have real
cut events to repair.
"""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import CANONICAL_CLASSES

CLASS_TONES_HZ = {
    "speech": (220.0, 330.0),
    "music": (440.0, 550.0),
    "noise": (110.0, 165.0),
    "alarm": (880.0, 990.0),
}


def _rng(seed: int, *parts: int) -> np.random.Generator:
    digest = seed
    for part in parts:
        digest = (digest * 1_000_003) ^ (part + 0x9E3779B1)
    digest &= 0xFFFFFFFF
    return np.random.default_rng(digest)


def _envelope(center_ms: int, start_ms: int, end_ms: int, fade_ms: int = 60) -> float:
    if center_ms < start_ms or center_ms >= end_ms:
        return 0.0
    left = min(fade_ms, max(1, (end_ms - start_ms) // 2))
    right = left
    if center_ms < start_ms + left:
        return 0.5 - 0.5 * math.cos(math.pi * (center_ms - start_ms) / left)
    if center_ms >= end_ms - right:
        return 0.5 - 0.5 * math.cos(math.pi * (end_ms - center_ms) / right)
    return 1.0


def synthesize_wav(
    path: Path,
    *,
    sample_rate: int,
    duration_ms: int,
    events: list[dict[str, Any]],
    seed: int,
) -> None:
    total = sample_rate * duration_ms // 1000
    rng = _rng(seed, 17)
    samples = rng.normal(0.0, 0.015, size=total)
    timeline = np.arange(total) / sample_rate
    for idx, event in enumerate(events):
        start_sample = event["start_ms"] * sample_rate // 1000
        end_sample = event["end_ms"] * sample_rate // 1000
        length = max(0, end_sample - start_sample)
        if length == 0:
            continue
        local_t = np.arange(length) / sample_rate
        f0, f1 = CLASS_TONES_HZ[event["class"]]
        phase = 2 * math.pi * (f0 * local_t + (f1 - f0) * local_t**2 / (2 * local_t[-1] if length > 1 else 1))
        tone = 0.28 * np.sin(phase) + 0.12 * np.sin(2 * phase)
        env = np.ones(length)
        fade = min(length, int(0.03 * sample_rate))
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, fade))
        env[:fade] = ramp
        env[-fade:] = ramp[::-1]
        samples[start_sample:end_sample] += tone * env
    peak = float(np.max(np.abs(samples)))
    if peak > 0.98:
        samples = samples * (0.98 / peak)
    pcm = np.round(samples * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def global_baseline_probabilities(
    *,
    frame_count: int,
    frame_shift_ms: int,
    events: list[dict[str, Any]],
    seed: int,
) -> np.ndarray:
    """Per-frame canonical-order probabilities before shard perturbation."""

    rng = _rng(seed, 29)
    rows = []
    for frame_index in range(frame_count):
        center_ms = frame_index * frame_shift_ms + frame_shift_ms // 2
        active = None
        for event in events:
            level = _envelope(center_ms, event["start_ms"], event["end_ms"])
            if level > 0.0:
                active = (event["class"], level)
                break
        base = np.full(4, 0.06, dtype=np.float64)
        base += rng.uniform(0.0, 0.03, size=4)
        if active is not None:
            label, level = active
            class_index = CANONICAL_CLASSES.index(label)
            base[class_index] = 0.72 * level + rng.uniform(0.0, 0.05)
        rows.append(base)
    matrix = np.vstack(rows)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix


def _perturb(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministic per-package perturbation that creates boundary jumps."""

    offsets = rng.uniform(-0.12, 0.12, size=matrix.shape[1])
    scale = rng.uniform(0.9, 1.1, size=matrix.shape[1])
    out = matrix * scale + offsets
    out = np.clip(out, 0.01, 0.99)
    return out / out.sum(axis=1, keepdims=True)


def _reorder_for_codebook(
    matrix: np.ndarray, codebook: dict[str, str]
) -> tuple[list[str], np.ndarray]:
    codes = sorted(codebook, key=lambda code: int(code))
    usable = [code for code in codes if codebook[code] in CANONICAL_CLASSES]
    columns = [CANONICAL_CLASSES.index(codebook[code]) for code in usable]
    return usable, matrix[:, columns]


def _write_package(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def _resolve(workspace: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else workspace / candidate


def generate(workspace: Path, config_path: Path) -> dict[str, int]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest_path = _resolve(workspace, config["paths"]["manifest"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    truth_path = _resolve(workspace, config["paths"]["truth"])
    with truth_path.open("r", encoding="utf-8") as handle:
        truth = yaml.safe_load(handle)

    seed = int(manifest["seed"])
    audio_cfg = config["audio"]
    shift = audio_cfg["frame_shift_ms"]
    total_frames = audio_cfg["total_duration_ms"] // shift + 1
    baseline = global_baseline_probabilities(
        frame_count=total_frames,
        frame_shift_ms=shift,
        events=truth["events"],
        seed=seed,
    )

    synthesize_wav(
        _resolve(workspace, config["paths"]["wav"]),
        sample_rate=audio_cfg["sample_rate_hz"],
        duration_ms=audio_cfg["total_duration_ms"],
        events=truth["events"],
        seed=seed,
    )

    shard_dir = _resolve(workspace, config["paths"]["shard_dir"])
    written = 0
    codebooks = config["classes"]["codebooks"]
    grids = config["classes"]["grids"]

    all_specs = list(manifest["shards"]) + list(manifest.get("invalid_shards", []))
    for ordinal, spec in enumerate(all_specs):
        shard_id = spec["shard_id"]
        header = dict(spec["header"])
        grid_version = header["frame_grid_version"]
        codebook_version = header["codebook_version"]
        native_shift = grids[grid_version]["frame_shift_ms"]
        start_ms, end_ms = int(spec["start_ms"]), int(spec["end_ms"])
        native_count = (end_ms - start_ms) // native_shift + 1
        native_centers = start_ms + np.arange(native_count) * native_shift
        source_positions = native_centers / shift
        lower = np.floor(source_positions).astype(int)
        upper = np.ceil(source_positions).astype(int)
        lower = np.clip(lower, 0, total_frames - 1)
        upper = np.clip(upper, 0, total_frames - 1)
        weight = (source_positions - lower)[:, None]
        native_probs = baseline[lower] * (1 - weight) + baseline[upper] * weight
        rng = _rng(seed, ordinal + 1, len(shard_id))
        native_probs = _perturb(native_probs, rng)

        codes, ordered = _reorder_for_codebook(
            native_probs, codebooks[codebook_version]
        )
        if header.get("inject_unmapped_code"):
            extra_code = header["inject_unmapped_code"]
            extra = np.full((ordered.shape[0], 1), 0.02)
            chosen = rng.choice(ordered.shape[0], size=min(12, ordered.shape[0]), replace=False)
            extra[chosen, 0] = rng.uniform(0.35, 0.8, size=chosen.size)
            codes = codes + [extra_code]
            ordered = np.hstack([ordered, extra])

        if spec.get("inject_probability_violation"):
            ordered[0, 0] = 1.7
        lengths_match = (ordered.shape[0] == ((end_ms - start_ms) // native_shift + 1))
        if spec.get("inject_length_mismatch") and lengths_match:
            ordered = ordered[:-1, :]

        payload = {
            "shard_id": shard_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "header": {
                "frame_grid_version": grid_version,
                "codebook_version": codebook_version,
                "frame_length_ms": grids[grid_version]["frame_length_ms"],
                "frame_shift_ms": native_shift,
            },
            "class_codes": codes,
            "frame_probabilities": [[round(float(v), 6) for v in row] for row in ordered],
        }
        _write_package(shard_dir / f"{shard_id}.json", payload)
        written += 1

    return {"shards_written": written, "global_frames": total_frames}
