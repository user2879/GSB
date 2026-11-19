from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from .audio import wav_bytes
from .canonical import atomic_write_bytes, atomic_write_json, canonical_bytes, sha256_bytes, sha256_file
from .config import SAMPLE_RATE


GENERATOR_VERSION = 1


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 20261119
    sample_rate: int = SAMPLE_RATE
    seconds_per_track: float = 12.0
    track_count: int = 7

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError("generator currently requires 44100 Hz")
        if not 8.0 <= self.seconds_per_track <= 30.0:
            raise ValueError("seconds_per_track must be 8..30")
        if not 7 <= self.track_count <= 16:
            raise ValueError("track_count must be 7..16")


SCALE_HZ = (110.0, 123.471, 146.832, 164.814, 196.0, 220.0, 246.941, 293.665)


def _envelope(length: int, attack: int, release: int) -> np.ndarray:
    result = np.ones(length, dtype=np.float64)
    attack = max(1, min(attack, length // 2))
    release = max(1, min(release, length // 2))
    result[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
    result[-release:] = np.linspace(1.0, 0.0, release, endpoint=True)
    return result


def _synth_note(freq: float, seconds: float, timbre: int, sample_rate: int) -> np.ndarray:
    count = max(1, int(round(seconds * sample_rate)))
    time = np.arange(count, dtype=np.float64) / sample_rate
    if timbre == 0:
        partials = ((1.0, 1.0), (2.0, 0.28), (3.0, 0.12))
        phase = np.sin(2.0 * np.pi * freq * time)
    elif timbre == 1:
        partials = ((1.0, 0.72), (2.0, 0.48), (3.0, 0.30), (4.0, 0.18), (5.0, 0.1))
        phase = np.zeros_like(time)
    elif timbre == 2:
        partials = ((1.0, 1.0), (2.0, 0.06), (3.0, 0.02))
        phase = np.sin(2.0 * np.pi * freq * time)
    else:
        partials = ((1.0, 0.68), (1.5, 0.2), (2.0, 0.3), (3.0, 0.15))
        phase = np.sin(2.0 * np.pi * freq * time + 0.13)
    wave = np.zeros_like(time)
    for harmonic, amplitude in partials:
        wave += amplitude * np.sin(2.0 * np.pi * freq * harmonic * time + harmonic * 0.07)
    if timbre == 1:
        wave += phase * 0.02
    wave *= _envelope(count, max(1, count // 40), max(1, count // 8))
    return wave


def _build_background(track_index: int, config: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    sample_rate = config.sample_rate
    total = int(round(config.seconds_per_track * sample_rate))
    output = np.zeros(total, dtype=np.float64)
    section_seconds = 1.5
    section_count = int(math.ceil(config.seconds_per_track / section_seconds))
    for section in range(section_count):
        timbre = (track_index + section) % 4
        section_start = int(round(section * section_seconds * sample_rate))
        section_end = min(total, int(round((section + 1) * section_seconds * sample_rate)))
        if section_start >= total:
            break
        if section in {2, 7}:
            continue
        note_seconds = 0.25 if section % 2 == 0 else 0.375
        cursor = section_start
        note_index = 0
        while cursor < section_end:
            length = min(int(round(note_seconds * sample_rate)), section_end - cursor)
            if length <= 0:
                break
            scale_index = (track_index * 2 + section + note_index * (1 + track_index % 2)) % len(SCALE_HZ)
            octave = 0.5 if track_index >= 5 else 1.0
            note = _synth_note(SCALE_HZ[scale_index] * octave, length / sample_rate, timbre, sample_rate)
            if len(note) < length:
                note = np.pad(note, (0, length - len(note)))
            fade = min(96, length // 6)
            if fade > 1:
                note[-fade:] *= np.linspace(1.0, 0.35, fade)
            output[cursor : cursor + length] += 0.085 * note[:length]
            cursor += length
            note_index += 1

    beat_seconds = 0.5 if track_index % 2 == 0 else 0.375
    beat = 0.0
    while beat < config.seconds_per_track - 0.05:
        start = int(round(beat * sample_rate))
        length = int(round(0.11 * sample_rate))
        if start < total:
            amount = min(length, total - start)
            local_t = np.arange(amount) / sample_rate
            kick = np.sin(2.0 * np.pi * (74.0 - 28.0 * local_t) * local_t)
            kick *= np.exp(-local_t * 26.0)
            output[start : start + amount] += 0.11 * kick
        hat_start = start + int(round(0.25 * sample_rate))
        if hat_start < total:
            amount = min(int(round(0.035 * sample_rate)), total - hat_start)
            noise = rng.standard_normal(amount)
            noise = signal.lfilter([1.0, -0.85], [1.0], noise)
            noise *= np.exp(-np.arange(amount) / max(1.0, amount * 0.22))
            output[hat_start : hat_start + amount] += 0.017 * noise
        beat += beat_seconds

    live_noise = rng.standard_normal(total) * 0.0028
    live_noise = signal.lfilter([0.08, 0.12, 0.16, 0.12, 0.08], [1.0], live_noise)
    envelope = 0.75 + 0.25 * np.sin(2.0 * np.pi * (0.11 + track_index * 0.003) * np.arange(total) / sample_rate)
    output += live_noise * envelope

    silence_starts = (2.8, 9.2)
    for seconds in silence_starts:
        start = int(round(seconds * sample_rate))
        end = min(total, start + int(round(0.42 * sample_rate)))
        if start < total:
            output[start:end] *= np.linspace(1.0, 0.02, end - start)
    return output


def _synth_motif(seed: int, kind_index: int, sample_rate: int, pitch_ratio: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed + 10_000 + kind_index)
    base_seconds = 1.6
    count = int(round(base_seconds * sample_rate))
    output = np.zeros(count, dtype=np.float64)
    note_starts = (0.0, 0.2, 0.43, 0.69, 0.94, 1.18, 1.39)
    degrees = (0, 3, 5, 7, 5, 2, 4)
    for note_index, (start_seconds, degree) in enumerate(zip(note_starts, degrees)):
        start = int(round(start_seconds * sample_rate))
        length = min(count - start, int(round(0.31 * sample_rate)))
        if length <= 0:
            continue
        freq = SCALE_HZ[degree % len(SCALE_HZ)] * pitch_ratio
        timbre = (kind_index + note_index) % 4
        note = _synth_note(freq, length / sample_rate, timbre, sample_rate)
        output[start : start + length] += 0.34 * note[:length]
        click_len = min(int(round(0.018 * sample_rate)), count - start)
        if click_len > 0:
            click = rng.standard_normal(click_len) * np.exp(-np.arange(click_len) / max(1.0, click_len / 5.0))
            output[start : start + click_len] += 0.018 * click
    return output


def _fit_slice(signal_data: np.ndarray, start: int, length: int, total: int) -> tuple[int, int, np.ndarray]:
    start = max(0, min(start, total))
    end = max(start, min(start + length, total))
    return start, end, signal_data[: end - start]


def _quantize_to_pcm(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -0.98, 0.98)
    return np.rint(clipped * 32767.0).astype("<i2")


def generate_materials(root: str | Path, config: GeneratorConfig | None = None) -> dict[str, Any]:
    config = config or GeneratorConfig()
    output_root = Path(root)
    if output_root.exists():
        shutil.rmtree(output_root)
    tracks_dir = output_root / "tracks"
    queries_dir = output_root / "queries"
    tracks_dir.mkdir(parents=True)
    queries_dir.mkdir(parents=True)
    sample_rate = config.sample_rate
    total = int(round(config.seconds_per_track * sample_rate))

    track_ids = [f"track_{index + 1:02d}" for index in range(config.track_count)]
    float_tracks: list[np.ndarray] = []
    rngs: list[np.random.Generator] = []
    for index in range(config.track_count):
        rng = np.random.default_rng(config.seed + index * 9973)
        rngs.append(rng)
        float_tracks.append(_build_background(index, config, rng))

    variants: list[dict[str, Any]] = []
    variant_specs = (
        {
            "variant_id": "v_exact_cross",
            "kind": "exact_repeat",
            "source_track": "track_01",
            "source_start": 1.2,
            "destination_track": "track_02",
            "destination_start": 7.4,
            "pitch_ratio": 1.0,
            "speed_ratio": 1.0,
        },
        {
            "variant_id": "v_pitch_up",
            "kind": "pitch_shift",
            "source_track": "track_03",
            "source_start": 3.0,
            "destination_track": "track_04",
            "destination_start": 7.0,
            "pitch_ratio": 2.0 ** (3.0 / 12.0),
            "speed_ratio": 1.0,
        },
        {
            "variant_id": "v_time_fast",
            "kind": "time_stretch",
            "source_track": "track_05",
            "source_start": 2.0,
            "destination_track": "track_06",
            "destination_start": 8.2,
            "pitch_ratio": 2.0 ** (7.0 / 12.0),
            "speed_ratio": 1.005,
        },
        {
            "variant_id": "v_exact_same",
            "kind": "exact_repeat",
            "source_track": "track_01",
            "source_start": 4.0,
            "destination_track": "track_01",
            "destination_start": 9.7,
            "pitch_ratio": 1.0,
            "speed_ratio": 1.0,
        },
        {
            "variant_id": "v_pitch_down_cross",
            "kind": "pitch_shift",
            "source_track": "track_07",
            "source_start": 3.8,
            "destination_track": "track_06",
            "destination_start": 3.9,
            "pitch_ratio": 2.0 ** (-2.0 / 12.0),
            "speed_ratio": 1.0,
        },
    )

    for variant_index, spec in enumerate(variant_specs):
        source_index = track_ids.index(spec["source_track"])
        destination_index = track_ids.index(spec["destination_track"])
        source_rate = config.seed + source_index * 9973
        motif = _synth_motif(source_rate, variant_index, sample_rate, spec["pitch_ratio"])
        source_start = int(round(spec["source_start"] * sample_rate))
        source_end = min(total, source_start + len(motif))
        add_length = source_end - source_start
        float_tracks[source_index][source_start:source_end] += motif[:add_length]

        destination_len = int(round(add_length / spec["speed_ratio"]))
        destination_start = int(round(spec["destination_start"] * sample_rate))
        destination_end = min(total, destination_start + destination_len)
        destination_len = destination_end - destination_start
        if destination_len != add_length:
            transformed = signal.resample(motif[:add_length], destination_len)
        else:
            transformed = motif[:add_length]
        float_tracks[destination_index][destination_start:destination_end] += 0.92 * transformed[:destination_len]
        variants.append(
            {
                "variant_id": spec["variant_id"],
                "kind": spec["kind"],
                "source_track": spec["source_track"],
                "source_start_sample": source_start,
                "source_end_sample": source_end,
                "destination_track": spec["destination_track"],
                "destination_start_sample": destination_start,
                "destination_end_sample": destination_end,
                "pitch_ratio": round(float(spec["pitch_ratio"]), 9),
                "speed_ratio": round(float(spec["speed_ratio"]), 9),
            }
        )

    track_records: list[dict[str, Any]] = []
    pcm_tracks: list[np.ndarray] = []
    for index, track_id in enumerate(track_ids):
        pcm = _quantize_to_pcm(float_tracks[index])
        if len(pcm) != total:
            raise AssertionError("generated sample count drifted")
        pcm_tracks.append(pcm)
        data = wav_bytes(pcm, sample_rate)
        path = tracks_dir / f"{track_id}.wav"
        atomic_write_bytes(path, data)
        silence_intervals = []
        for seconds in (2.8, 9.2):
            start = int(round(seconds * sample_rate))
            end = min(total, start + int(round(0.42 * sample_rate)))
            silence_intervals.append([start, end])
        track_records.append(
            {
                "track_id": track_id,
                "file": f"tracks/{track_id}.wav",
                "wav_sha256": sha256_bytes(data),
                "duration_samples": int(len(pcm)),
                "duration_seconds": round(len(pcm) / sample_rate, 6),
                "sample_rate": sample_rate,
                "channels": 1,
                "bit_depth": 16,
                "silence_intervals_samples": silence_intervals,
                "timbre_sections": [
                    {
                        "start_sample": int(round(section * 1.5 * sample_rate)),
                        "end_sample": min(total, int(round((section + 1) * 1.5 * sample_rate))),
                        "timbre": (index + section) % 4,
                    }
                    for section in range(int(math.ceil(config.seconds_per_track / 1.5)))
                    if section not in {2, 7}
                ],
            }
        )

    for variant in variants:
        if not (
            0 <= variant["source_start_sample"] < variant["source_end_sample"] <= total
            and 0 <= variant["destination_start_sample"] < variant["destination_end_sample"] <= total
        ):
            raise AssertionError("ground-truth interval falls outside audio")

    query_variants = ("v_exact_cross", "v_pitch_up", "v_time_fast", "v_exact_same", "v_pitch_down_cross")
    queries: list[dict[str, Any]] = []
    for query_index, variant_id in enumerate(query_variants):
        variant = next(item for item in variants if item["variant_id"] == variant_id)
        destination_index = track_ids.index(variant["destination_track"])
        start = variant["destination_start_sample"]
        end = variant["destination_end_sample"]
        query_samples = pcm_tracks[destination_index][start:end].copy()
        query_id = f"q_{query_index + 1:02d}_{variant_id[2:]}"
        path = queries_dir / f"{query_id}.wav"
        data = wav_bytes(query_samples, sample_rate)
        atomic_write_bytes(path, data)
        queries.append(
            {
                "query_id": query_id,
                "file": f"queries/{query_id}.wav",
                "wav_sha256": sha256_bytes(data),
                "duration_samples": int(len(query_samples)),
                "duration_seconds": round(len(query_samples) / sample_rate, 6),
                "variant_id": variant_id,
                "variant_kind": variant["kind"],
                "expected_track_id": variant["source_track"],
                "expected_start_sample": variant["source_start_sample"],
                "expected_end_sample": variant["source_end_sample"],
            }
        )

    manifest = {
        "schema": 1,
        "generator_version": GENERATOR_VERSION,
        "generator_config": asdict(config),
        "sample_rate": sample_rate,
        "channels": 1,
        "bit_depth": 16,
        "sample_range_convention": "half_open_[start,end)",
        "tracks": track_records,
        "variants": variants,
        "queries": queries,
    }
    manifest_path = output_root / "truth_manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    (output_root / "truth_manifest.sha256").write_text(manifest_hash + "\n", encoding="ascii")
    (output_root / "generator_config.json").write_bytes(canonical_bytes(asdict(config)) + b"\n")
    return {
        "root": str(output_root),
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "tracks": {record["track_id"]: str(output_root / record["file"]) for record in track_records},
        "queries": {record["query_id"]: str(output_root / record["file"]) for record in queries},
    }
