"""Deterministic per-template acoustic extraction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import RegistrationConfig
from .fingerprint import rounded_vector, sha256_json, stable_float
from .wavio import AudioData

def raw_feature_count(config: RegistrationConfig) -> int:
    return 2 * (11 + int(config.features["cepstral_coefficients"]))


def _dct_matrix(size: int, coefficients: int) -> np.ndarray:
    index = np.arange(size, dtype=np.float64)
    order = np.arange(coefficients, dtype=np.float64)[:, None]
    matrix = np.cos((math.pi / size) * (index[None, :] + 0.5) * order)
    matrix[0, :] *= 1.0 / math.sqrt(2.0)
    return matrix * math.sqrt(2.0 / size)


def projection_matrix(config: RegistrationConfig) -> np.ndarray:
    dimension = int(config.features["dimension"])
    seed = int(config.projection["seed"])
    version = int(config.projection["seed_version"])
    config_fingerprint = config.feature_fingerprint
    count = raw_feature_count(config)
    rows = []
    for output_index in range(dimension):
        row = np.asarray(
            [
                stable_float(seed, "projection", config_fingerprint, version, output_index, input_index)
                for input_index in range(count)
            ],
            dtype=np.float64,
        )
        row -= float(np.mean(row))
        norm = float(np.linalg.norm(row))
        rows.append(row / norm if norm > 0 else row)
    return np.asarray(rows, dtype=np.float64)


def projection_fingerprint(config: RegistrationConfig) -> str:
    return sha256_json(
        {
            "seed": config.projection["seed"],
            "seed_version": config.projection["seed_version"],
            "dimension": config.features["dimension"],
            "raw_feature_count": raw_feature_count(config),
        }
    )


def _segment_stats(frames: list[dict[str, Any]], coefficient_count: int) -> list[float]:
    if not frames:
        return [0.0] * (11 + coefficient_count)
    energies = np.asarray([frame["energy"] for frame in frames], dtype=np.float64)
    crossings = np.asarray([frame["zcr"] for frame in frames], dtype=np.float64)
    cepstra = np.asarray([frame["cepstrum"][:coefficient_count] for frame in frames], dtype=np.float64)
    bands = np.asarray([frame["bands"] for frame in frames], dtype=np.float64)
    formants = np.asarray([frame["formants"] for frame in frames], dtype=np.float64)
    return [
        float(np.mean(energies)) * 5.0,
        float(np.std(energies, ddof=0)) * 5.0,
        float(np.mean(crossings)) * 5.0,
        float(np.std(crossings, ddof=0)) * 5.0,
        *[float(value) / 20.0 for value in np.std(cepstra, axis=0, ddof=0)],
        *[(float(value) - 0.25) * 6.0 for value in np.mean(bands, axis=0)],
        *[float(value) for value in np.mean(formants, axis=0)],
    ]


def extract_acoustic_template(
    audio: AudioData,
    config: RegistrationConfig,
    *,
    command_word: str,
    template_id: str,
    relative_path: str,
) -> dict[str, Any]:
    samples = audio.samples.astype(np.float64) / 32768.0
    frame_length = int(config.features["frame_length"])
    frame_shift = int(config.features["frame_shift"])
    coefficient_count = int(config.features["cepstral_coefficients"])
    if samples.size < frame_length:
        frame_starts = [0]
        padded = np.zeros(frame_length, dtype=np.float64)
        padded[: samples.size] = samples
        frame_buffers = [padded]
    else:
        frame_starts = list(range(0, samples.size - frame_length + 1, frame_shift))
        frame_buffers = [samples[start : start + frame_length] for start in frame_starts]

    window = np.hanning(frame_length).astype(np.float64)
    spectrum_length = frame_length // 2 + 1
    dct = _dct_matrix(spectrum_length, coefficient_count)
    frames: list[dict[str, Any]] = []
    for frame_index, (start, frame) in enumerate(zip(frame_starts, frame_buffers, strict=True)):
        energy = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        signs = np.signbit(frame)
        crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
        zcr = float(crossings / max(1, frame_length - 1))
        spectrum = np.abs(np.fft.rfft(frame * window, n=frame_length))
        power = np.square(spectrum)
        total_power = float(np.sum(power))
        freq_step = audio.sample_rate / frame_length
        band_edges = [(0, 500), (500, 1000), (1000, 2000), (2000, audio.sample_rate // 2 + 1)]
        bands = []
        for low_hz, high_hz in band_edges:
            low = max(0, int(low_hz / freq_step))
            high = min(power.size, max(low + 1, int(high_hz / freq_step)))
            bands.append(float(np.sum(power[low:high]) / total_power) if total_power > 0 else 0.0)
        formants = []
        for low_hz, high_hz in ((250, 900), (900, 2400), (2400, 3800)):
            low = max(0, int(low_hz / freq_step))
            high = min(power.size, max(low + 1, int(high_hz / freq_step)))
            local = power[low:high]
            peak = low + int(np.argmax(local)) if local.size else low
            formants.append(float(peak * freq_step) / 4000.0)
        log_spectrum = np.log(spectrum + 1.0e-9)
        cepstrum = np.asarray(dct @ log_spectrum, dtype=np.float64)
        frames.append(
            {
                "frame_index": frame_index,
                "sample_start": start,
                "energy": energy,
                "zcr": zcr,
                "cepstrum": [float(value) for value in cepstrum],
                "bands": bands,
                "formants": formants,
            }
        )

    energies = np.asarray([frame["energy"] for frame in frames], dtype=np.float64)
    median_energy = float(np.median(energies)) if energies.size else 0.0
    vowel_frames: list[dict[str, Any]] = []
    fricative_frames: list[dict[str, Any]] = []
    for frame in sorted(frames, key=lambda item: item["frame_index"]):
        is_fricative = frame["zcr"] >= 0.24 and (
            frame["energy"] <= median_energy * 1.5 or frame["zcr"] >= 0.38
        )
        (fricative_frames if is_fricative else vowel_frames).append(frame)

    raw_features = _segment_stats(vowel_frames, coefficient_count) + _segment_stats(fricative_frames, coefficient_count)
    projected = projection_matrix(config) @ np.asarray(raw_features, dtype=np.float64)
    projected = projected - float(np.mean(projected, dtype=np.float64))
    projected = projected * 6.0
    vector = rounded_vector(projected, 12)
    total_frames = len(vowel_frames) + len(fricative_frames)
    projection_version = int(config.projection["seed_version"])
    vector_fingerprint = sha256_json({"vector": vector, "projection_version": projection_version})
    return {
        "template_key": f"{command_word}:{template_id}",
        "command_word": command_word,
        "template_id": template_id,
        "path": relative_path,
        "content_sha256": audio.content_sha256,
        "sample_count": audio.sample_count,
        "frame_length": frame_length,
        "frame_shift": frame_shift,
        "frame_count": total_frames,
        "effective_frames": {
            "vowel": len(vowel_frames),
            "fricative": len(fricative_frames),
            "total": total_frames,
        },
        "segment_statistics": {
            "vowel": _segment_stats(vowel_frames, coefficient_count),
            "fricative": _segment_stats(fricative_frames, coefficient_count),
        },
        "projection_seed_version": projection_version,
        "projection_fingerprint": projection_fingerprint(config),
        "vector": vector,
        "dimensions": [
            {
                "index": index,
                "value": value,
                "projection_seed_version": projection_version,
                "valid_frames": total_frames,
            }
            for index, value in enumerate(vector)
        ],
        "vector_fingerprint": vector_fingerprint,
    }
