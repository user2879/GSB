"""Deterministic four-channel RIR synthesis and truth metadata."""

from __future__ import annotations

import itertools
import math
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .canonical import canonical_hash, json_safe, sha256_file
from .config import Config


GENERATOR_VERSION = "deterministic-foa-image-source/1"
PLASTIC = 1.32471795724474602596
R3_ALPHAS = (1.0 / PLASTIC, 1.0 / (PLASTIC**2), 1.0 / (PLASTIC**3))


def _fractional(value: float) -> float:
    return value - math.floor(value)


def _r3_unit(index: int) -> np.ndarray:
    point = np.asarray(
        [_fractional(0.5 + (index + 1) * alpha) for alpha in R3_ALPHAS], dtype=np.float64
    )
    vector = 2.0 * point - 1.0
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm


def _source_seed(base_seed: int, index: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), 0x534F5552, int(index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def source_catalogue(config: Config, count: int) -> list[dict[str, Any]]:
    """Nested deterministic source catalogue.

    The first eight sources in the 96-source tier are exactly the eight
    sources in the small tier.  This makes cross-tier comparisons a comparison
    of added observations rather than an unrelated room.
    """

    room = config.room()
    dimensions = np.asarray(room["dimensions_m"], dtype=np.float64)
    listener = np.asarray(room["listener_m"], dtype=np.float64)
    radius_min, radius_max = (float(v) for v in room["source_radius_m"])
    margin = float(room["source_margin_m"])
    sources: list[dict[str, Any]] = []
    for index in range(int(count)):
        direction = _r3_unit(index)
        limits: list[float] = []
        for axis in range(3):
            if direction[axis] > 1.0e-12:
                limits.append((dimensions[axis] - margin - listener[axis]) / direction[axis])
            elif direction[axis] < -1.0e-12:
                limits.append((margin - listener[axis]) / direction[axis])
        max_radius = min(limits) if limits else radius_max
        allowed_max = min(radius_max, max_radius)
        if allowed_max <= radius_min:
            allowed_max = max(radius_min, 0.5 * max_radius)
        fraction = _fractional(0.5 + (index + 1) * 0.7548776662466927)
        radius = radius_min + (allowed_max - radius_min) * fraction
        position = listener + radius * direction
        position = np.clip(position, margin, dimensions - margin)
        sources.append(
            {
                "index": index,
                "position_m": position.tolist(),
                "radial_direction": direction.tolist(),
                "radius_m": float(np.linalg.norm(position - listener)),
                "seed": _source_seed(config.seed, index),
            }
        )
    return sources


def _reflection_sequences(max_order: int) -> list[tuple[int, ...]]:
    sequences: list[tuple[int, ...]] = [tuple()]
    for order in range(1, int(max_order) + 1):
        for candidate in itertools.product(range(6), repeat=order):
            if any(candidate[i] == candidate[i + 1] for i in range(len(candidate) - 1)):
                continue
            sequences.append(tuple(int(v) for v in candidate))
    return sequences


def _reflect(point: np.ndarray, wall: int, dimensions: np.ndarray) -> np.ndarray:
    reflected = np.asarray(point, dtype=np.float64).copy()
    axis = wall // 2
    if wall % 2 == 0:
        reflected[axis] = -reflected[axis]
    else:
        reflected[axis] = 2.0 * dimensions[axis] - reflected[axis]
    return reflected


def _image_source(
    source_position: np.ndarray, wall_sequence: Iterable[int], dimensions: np.ndarray
) -> np.ndarray:
    image = np.asarray(source_position, dtype=np.float64)
    for wall in wall_sequence:
        image = _reflect(image, int(wall), dimensions)
    return image


def _fmt_sequence(sequence: tuple[int, ...]) -> str:
    return "".join(str(int(v)) for v in sequence) or "direct"


def _channel_first_arrivals(paths: list[dict[str, Any]], channels: int) -> list[int]:
    first_arrivals: list[int] = []
    for channel in range(channels):
        indexes: list[int] = []
        for path in paths:
            if channel == 0:
                component = 1.0 / math.sqrt(2.0)
            else:
                component = float(path["direction"][channel - 1])
            if abs(component) > 1.0e-12:
                indexes.append(int(path["arrival_samples"]))
        first_arrivals.append(min(indexes) if indexes else -1)
    return first_arrivals


def _paths_for_source(
    source_position: np.ndarray,
    room: dict[str, Any],
    sequences: list[tuple[int, ...]],
    rir_samples: int,
) -> list[dict[str, Any]]:
    dimensions = np.asarray(room["dimensions_m"], dtype=np.float64)
    listener = np.asarray(room["listener_m"], dtype=np.float64)
    speed = float(room["speed_of_sound_m_s"])
    sample_rate = int(round(float(room["_sample_rate"])))
    absorption = np.asarray(room["wall_absorption"], dtype=np.float64)
    paths: list[dict[str, Any]] = []
    for sequence in sequences:
        image = _image_source(source_position, sequence, dimensions)
        offset = image - listener
        distance = float(np.linalg.norm(offset))
        if distance <= 1.0e-9:
            continue
        arrival_sample = int(round(distance / speed * sample_rate))
        if arrival_sample < 0 or arrival_sample >= rir_samples:
            continue
        reflection_energy = 1.0
        for wall in sequence:
            reflection_energy *= 1.0 - float(absorption[wall])
        energy = reflection_energy / (distance * distance)
        direction = offset / distance
        paths.append(
            {
                "wall_sequence": [int(v) for v in sequence],
                "wall_label": _fmt_sequence(sequence),
                "order": len(sequence),
                "image_source_m": image.tolist(),
                "direction": direction.tolist(),
                "distance_m": distance,
                "arrival_samples": arrival_sample,
                "arrival_seconds": arrival_sample / float(sample_rate),
                "reflection_energy": reflection_energy,
                "energy": energy,
                "rendered_energy": energy,
            }
        )
    paths.sort(
        key=lambda item: (
            item["arrival_samples"],
            item["direction"][0],
            item["direction"][1],
            item["direction"][2],
            item["wall_label"],
        )
    )
    for index, path in enumerate(paths):
        path["path_index"] = index
    return paths


def _render_source(
    paths: list[dict[str, Any]],
    audio: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    frames = int(audio["rir_samples"])
    channels = int(audio["channels"])
    sample_rate = int(audio["sample_rate"])
    signal = np.zeros((channels, frames), dtype=np.float64)
    for path in paths:
        sample = int(path["arrival_samples"])
        direction = np.asarray(path["direction"], dtype=np.float64)
        amplitude = math.sqrt(float(path["energy"]))
        if int(path["order"]) % 2 == 1:
            amplitude = -amplitude
        signal[0, sample] += amplitude / math.sqrt(2.0)
        signal[1:, sample] += amplitude * direction

    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak <= 0.0:
        raise RuntimeError("synthetic source has zero peak")
    target_peak = 10.0 ** (float(audio["peak_dbfs"]) / 20.0)
    scale = target_peak / peak
    signal *= scale

    noise_rms = 10.0 ** (float(audio["noise_floor_dbfs"]) / 20.0)
    signal += rng.normal(0.0, noise_rms, size=signal.shape)
    signal = np.clip(signal, -1.0, 1.0)

    pcm = np.rint(signal.T * 32767.0).astype("<i2", copy=False)
    return pcm, scale


def _write_wav(path: Path, pcm_frames: np.ndarray, audio: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(audio["channels"]))
        handle.setsampwidth(int(audio["bit_depth"]) // 8)
        handle.setframerate(int(audio["sample_rate"]))
        handle.writeframes(pcm_frames.tobytes(order="C"))


def generate_tier(config: Config, tier: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Generate a complete tier and return its truth manifest."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = config.audio()
    room = config.room()
    room_with_rate = dict(room)
    room_with_rate["_sample_rate"] = audio["sample_rate"]
    sequences = _reflection_sequences(int(room["reflection_order"]))
    source_count = int(tier["source_positions"])
    sources = source_catalogue(config, source_count)
    manifest_sources: list[dict[str, Any]] = []
    sample_paths: list[Path] = []

    for source in sources:
        paths = _paths_for_source(
            np.asarray(source["position_m"], dtype=np.float64),
            room_with_rate,
            sequences,
            int(audio["rir_samples"]),
        )
        if not paths:
            raise RuntimeError(f"source {source['index']} has no audible paths")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(source["seed"])))
        pcm, scale = _render_source(paths, audio, rng)
        for path in paths:
            path["rendered_energy"] = float(path["energy"]) * scale * scale

        channel_first_arrivals = _channel_first_arrivals(paths, int(audio["channels"]))

        wav_name = f"source_{source['index']:03d}.wav"
        wav_path = out_dir / wav_name
        _write_wav(wav_path, pcm, audio)
        sample_paths.append(wav_path)
        manifest_sources.append(
            {
                "index": int(source["index"]),
                "seed": int(source["seed"]),
                "position_m": source["position_m"],
                "radial_direction": source["radial_direction"],
                "radius_m": source["radius_m"],
                "normalization_scale": scale,
                "first_arrival_samples": channel_first_arrivals,
                "first_arrival_seconds": [
                    value / float(audio["sample_rate"]) if value >= 0 else None
                    for value in channel_first_arrivals
                ],
                "wav": {
                    "path": wav_name,
                    "sha256": sha256_file(wav_path),
                    "sample_rate": int(audio["sample_rate"]),
                    "channels": int(audio["channels"]),
                    "bit_depth": int(audio["bit_depth"]),
                    "frames": int(audio["rir_samples"]),
                },
                "paths": paths,
            }
        )

    manifest: dict[str, Any] = {
        "schema": "spatial-diagnostic-rir-manifest/1",
        "generator_version": GENERATOR_VERSION,
        "config_hash": config.config_hash,
        "tier": {
            "name": tier["name"],
            "source_positions": source_count,
            "sphere_points": int(tier["sphere_points"]),
        },
        "seed": config.seed,
        "audio": audio,
        "room": room,
        "truth_policy": (
            "path directions, arrival samples, and rendered path energies are emitted by "
            "the geometric image-source generator before reconstruction; they are never "
            "inferred from reconstruction output"
        ),
        "path_sequence_count": len(sequences),
        "path_count": sum(len(source["paths"]) for source in manifest_sources),
        "sources": manifest_sources,
    }
    manifest["content_hash"] = canonical_hash(manifest)
    manifest_path = out_dir / "manifest.json"
    from .canonical import atomic_write_json

    atomic_write_json(manifest_path, json_safe(manifest))
    return {
        "manifest": json_safe(manifest),
        "manifest_path": manifest_path,
        "sample_paths": sorted(sample_paths),
        "step_hash": canonical_hash(manifest),
        "wav_hash_set": canonical_hash(
            {path.name: sha256_file(path) for path in sorted(sample_paths)}
        ),
    }
