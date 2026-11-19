"""Deterministic 48 kHz / 16-bit / four-channel room impulse responses."""

from __future__ import annotations

import io
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import CHANNEL_SPECS, SAMPLE_RATE, WALL_IDS, channel_spec
from .errors import E_PARAM_RANGE, fail
from .hashing import atomic_write_bytes, sha256_bytes


SPEED_OF_SOUND = 343.0


@dataclass(frozen=True)
class PathArrival:
    path_id: str
    kind: str
    wall: str | None
    distance_m: float
    delay_samples: int
    delay_seconds: float
    amplitude: float
    arrival_direction: tuple[float, float, float]
    reflection_point: tuple[float, float, float] | None


@dataclass(frozen=True)
class ChannelRir:
    samples: np.ndarray
    metadata: dict[str, Any]
    wav_bytes: bytes


def pcm16_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    if samples.ndim != 2:
        fail(E_PARAM_RANGE, "PCM sample array must be two-dimensional")
    if samples.dtype != np.int16:
        fail(E_PARAM_RANGE, "PCM sample array must be int16", dtype=str(samples.dtype))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(int(samples.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes(order="C"))
    return buffer.getvalue()


def read_pcm16_wav_bytes(data: bytes) -> tuple[np.ndarray, int, int]:
    buffer = io.BytesIO(data)
    with wave.open(buffer, "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if sample_width != 2:
        fail(E_PARAM_RANGE, "WAV is not 16-bit PCM", sample_width=sample_width)
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).copy()
    return samples, sample_rate, channels


def check_wav_roundtrip(path: Path, frame_count: int = 2048) -> dict[str, Any]:
    rng = np.random.default_rng(424242)
    probe = rng.integers(-12000, 12000, size=(frame_count, 4), dtype=np.int16)
    payload = pcm16_wav_bytes(probe)
    atomic_write_bytes(path, payload)
    decoded, sample_rate, channels = read_pcm16_wav_bytes(path.read_bytes())
    exact = bool(np.array_equal(decoded, probe))
    if not exact or sample_rate != SAMPLE_RATE or channels != 4:
        fail(
            E_PARAM_RANGE,
            "48 kHz / 16-bit / 4-channel WAV roundtrip failed",
            exact=exact,
            sample_rate=sample_rate,
            channels=channels,
        )
    return {
        "path": str(path),
        "sample_rate": sample_rate,
        "bit_depth": 16,
        "channels": channels,
        "frames": frame_count,
        "exact": exact,
        "sha256": sha256_bytes(payload),
    }


def _unit(vector: np.ndarray) -> tuple[float, float, float]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return (0.0, 0.0, 1.0)
    unit = vector / norm
    return (float(unit[0]), float(unit[1]), float(unit[2]))


def _forward(spec_azimuth: float, spec_elevation: float) -> np.ndarray:
    azimuth = math.radians(spec_azimuth)
    elevation = math.radians(spec_elevation)
    return np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = left / max(float(np.linalg.norm(left)), 1e-12)
    right_unit = right / max(float(np.linalg.norm(right)), 1e-12)
    cosine = float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _wall_geometry(wall_id: str, source: np.ndarray, room: dict[str, float]) -> tuple[np.ndarray, float, str, tuple[int, int]]:
    width = float(room["width"])
    depth = float(room["depth"])
    height = float(room["height"])
    sx, sy, sz = (float(part) for part in source)
    if wall_id == "xmin":
        return np.array([-sx, sy, sz]), 0.0, "x", (0, 0)
    if wall_id == "xmax":
        return np.array([2.0 * width - sx, sy, sz]), width, "x", (0, 1)
    if wall_id == "ymin":
        return np.array([sx, -sy, sz]), 0.0, "y", (1, 0)
    if wall_id == "ymax":
        return np.array([sx, 2.0 * depth - sy, sz]), depth, "y", (1, 1)
    if wall_id == "floor":
        return np.array([sx, sy, -sz]), 0.0, "z", (2, 0)
    if wall_id == "ceiling":
        return np.array([sx, sy, 2.0 * height - sz]), height, "z", (2, 1)
    fail(E_PARAM_RANGE, "unknown wall", wall_id=wall_id)


def _reflection_point(image_source: np.ndarray, listener: np.ndarray, axis_index: int, plane: float) -> np.ndarray:
    delta = listener - image_source
    denominator = float(delta[axis_index])
    if abs(denominator) <= 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    fraction = (plane - float(image_source[axis_index])) / denominator
    return image_source + fraction * delta


def _point_on_wall(point: np.ndarray, wall_id: str, room: dict[str, float]) -> bool:
    margin = 1e-9
    if wall_id in {"xmin", "xmax"}:
        return margin <= point[1] <= float(room["depth"]) + margin and margin <= point[2] <= float(room["height"]) + margin
    if wall_id in {"ymin", "ymax"}:
        return margin <= point[0] <= float(room["width"]) + margin and margin <= point[2] <= float(room["height"]) + margin
    return margin <= point[0] <= float(room["width"]) + margin and margin <= point[1] <= float(room["depth"]) + margin


def _collect_paths(scene: dict[str, Any], channel_id: int) -> list[PathArrival]:
    spec = channel_spec(channel_id)
    listener = np.asarray(scene["listener"], dtype=np.float64)
    source = np.asarray(scene["sources"][spec.source_id], dtype=np.float64)
    room = scene["room"]
    wall_alphas = scene["walls"]
    forward = _forward(spec.azimuth_deg, spec.elevation_deg)
    paths: list[PathArrival] = []

    direct_vector = source - listener
    direct_distance = max(float(np.linalg.norm(direct_vector)), 1e-6)
    direct_delay = direct_distance / SPEED_OF_SOUND
    direct_sample = int(round(direct_delay * SAMPLE_RATE))
    paths.append(
        PathArrival(
            path_id=f"c{channel_id}:direct:{spec.source_id}",
            kind="direct",
            wall=None,
            distance_m=direct_distance,
            delay_samples=direct_sample,
            delay_seconds=direct_delay,
            amplitude=0.90 / (1.0 + direct_distance),
            arrival_direction=_unit(direct_vector),
            reflection_point=None,
        )
    )

    for wall_id in WALL_IDS:
        image_source, plane, _axis, _face = _wall_geometry(wall_id, source, room)
        point = _reflection_point(image_source, listener, 0 if wall_id.startswith("x") else 1 if wall_id.startswith("y") else 2, plane)
        if not np.all(np.isfinite(point)) or not _point_on_wall(point, wall_id, room):
            continue
        arrival_vector = point - listener
        if _angle_deg(arrival_vector, forward) > spec.aperture_deg:
            continue
        distance = max(float(np.linalg.norm(listener - image_source)), 1e-6)
        delay = distance / SPEED_OF_SOUND
        alpha = float(wall_alphas[wall_id])
        reflection_gain = math.sqrt(max(0.01, 1.0 - alpha))
        paths.append(
            PathArrival(
                path_id=f"c{channel_id}:wall:{wall_id}",
                kind="reflection",
                wall=wall_id,
                distance_m=distance,
                delay_samples=int(round(delay * SAMPLE_RATE)),
                delay_seconds=delay,
                amplitude=(0.62 / (1.0 + distance)) * reflection_gain,
                arrival_direction=_unit(arrival_vector),
                reflection_point=(float(point[0]), float(point[1]), float(point[2])),
            )
        )
    paths.sort(key=lambda path: (path.delay_samples, path.path_id))
    return paths


def synthesize_channel(scene: dict[str, Any], channel_id: int) -> ChannelRir:
    if channel_id < 0 or channel_id >= len(CHANNEL_SPECS):
        fail(E_PARAM_RANGE, "channel id is outside the four-channel set", channel_id=channel_id)
    spec = channel_spec(channel_id)
    frame_count = int(round(float(scene["duration_seconds"]) * SAMPLE_RATE))
    samples = np.zeros(frame_count, dtype=np.float64)
    paths = _collect_paths(scene, channel_id)
    for path in paths:
        index = path.delay_samples
        if 0 <= index < frame_count:
            samples[index] += path.amplitude

    seed = int(scene["noise_seed"]) * 1009 + channel_id * 9176 + 17
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(frame_count)
    noise_envelope = np.exp(-np.arange(frame_count, dtype=np.float64) / (SAMPLE_RATE * 0.075))
    direct_index = paths[0].delay_samples if paths else 0
    noise[:direct_index] = 0.0
    samples += noise * noise_envelope * 7.5e-5

    peak = float(np.max(np.abs(samples))) if frame_count else 0.0
    scale = min(1.0, 0.94 / peak) if peak > 0.0 else 1.0
    pcm = np.clip(np.rint(samples * scale * 32767.0), -32768, 32767).astype(np.int16)
    wav_bytes = pcm16_wav_bytes(pcm[:, np.newaxis])

    path_records = []
    for path in paths:
        index = min(max(path.delay_samples, 0), frame_count - 1)
        sample_amplitude = abs(float(pcm[index])) / 32768.0
        record = asdict(path)
        record["sample_interval"] = [index, index + 1]
        record["quantized_amplitude"] = sample_amplitude
        record["energy"] = sample_amplitude * sample_amplitude * math.exp(-2.0 * path.delay_seconds / 0.28)
        path_records.append(record)
    peak_indices = np.argsort(np.abs(pcm), kind="stable")[-8:][::-1]
    peaks = [
        {
            "sample": int(index),
            "time_seconds": float(index) / SAMPLE_RATE,
            "amplitude": float(pcm[index]) / 32768.0,
        }
        for index in peak_indices
    ]
    metadata = {
        "channel_id": channel_id,
        "source_id": spec.source_id,
        "sample_rate": SAMPLE_RATE,
        "bit_depth": 16,
        "channels": 1,
        "frame_count": frame_count,
        "noise_seed": int(scene["noise_seed"]),
        "directional_forward": [spec.azimuth_deg, spec.elevation_deg],
        "directional_aperture_deg": spec.aperture_deg,
        "support_cap_deg": spec.support_deg,
        "paths": path_records,
        "walls_with_paths": sorted({path.wall for path in paths if path.wall is not None}),
        "first_arrival_sample": int(paths[0].delay_samples) if paths else None,
        "first_arrival_seconds": float(paths[0].delay_seconds) if paths else None,
        "peak_sample": int(np.argmax(np.abs(pcm))) if frame_count else None,
        "peaks": peaks,
        "wav_sha256": sha256_bytes(wav_bytes),
    }
    return ChannelRir(samples=pcm, metadata=metadata, wav_bytes=wav_bytes)


def rir_param_keys(metadata: dict[str, Any]) -> list[str]:
    keys = [f"source.{metadata['source_id']}.position", "noise.seed"]
    keys.extend(f"wall.{wall_id}.alpha" for wall_id in metadata["walls_with_paths"])
    return sorted(set(keys))


def synth_all_channels(scene: dict[str, Any]) -> dict[int, ChannelRir]:
    return {channel_id: synthesize_channel(scene, channel_id) for channel_id in range(len(CHANNEL_SPECS))}
