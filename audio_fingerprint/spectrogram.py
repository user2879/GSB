from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import fingerprint
from .config import spectral_fingerprint


FRAME_PRODUCT = "spec_frame"


def frame_product_id(clip_id: str, frame_index: int) -> str:
    return f"frame:{clip_id}:{frame_index:06d}"


def peak_product_id(clip_id: str, frame_index: int) -> str:
    return f"peaks:{clip_id}:{frame_index:06d}"


def frame_count(sample_count: int, config: dict[str, Any]) -> int:
    frame_size = int(config["frame_size"])
    hop = int(config["hop_size"])
    if sample_count < frame_size:
        return 0
    return (sample_count - frame_size) // hop + 1


def frame_cost_points(config: dict[str, Any]) -> int:
    return int(config["frame_size"])


def window_array(config: dict[str, Any]) -> np.ndarray:
    size = int(config["frame_size"])
    if config.get("window") == "ones":
        return np.ones(size, dtype=np.float64)
    return np.hanning(size).astype(np.float64)


def band_edges(config: dict[str, Any]) -> np.ndarray:
    return np.geomspace(
        float(config["band_min_hz"]),
        float(config["band_max_hz"]),
        int(config["band_count"]) + 1,
    )


def frame_cache_key(
    clip: dict[str, Any],
    normalization: dict[str, Any],
    config: dict[str, Any],
    frame_index: int,
    start_sample: int,
) -> str:
    return fingerprint(
        {
            "product_type": FRAME_PRODUCT,
            "product_id": frame_product_id(clip["clip_id"], frame_index),
            "audio_content_fingerprint": clip.get("content_fingerprint"),
            "normalized_content_fingerprint": normalization["normalized_content_fingerprint"],
            "format_fingerprint": clip["format_fingerprint"],
            "config_fingerprint": spectral_fingerprint(config),
            "quant_gear": config["spectrum_quant_gear"],
            "quant_shift": config["spectrum_quant_shift"],
            "frame_index": frame_index,
            "start_sample": start_sample,
        },
        "fkey_",
    )


def compute_frame(
    clip: dict[str, Any],
    normalization: dict[str, Any],
    samples: np.ndarray,
    frame_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    frame_size = int(config["frame_size"])
    hop = int(config["hop_size"])
    start = frame_index * hop
    segment = samples[start : start + frame_size]
    if segment.shape[0] != frame_size:
        raise IndexError(f"frame {frame_index} is outside available samples")

    window = window_array(config)
    spectrum = np.fft.rfft(segment * window)
    magnitude = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / int(config["internal_sample_rate"]))
    edges = band_edges(config)
    bands: list[int] = []
    residuals: list[int] = []
    scale = 1 << int(config["fixed_fraction_bits"])
    shift = int(config["spectrum_quant_shift"])

    for band_index in range(int(config["band_count"])):
        lower = edges[band_index]
        upper = edges[band_index + 1]
        if band_index + 1 == int(config["band_count"]):
            mask = (frequencies >= lower) & (frequencies <= upper)
        else:
            mask = (frequencies >= lower) & (frequencies < upper)
        amplitude = float(np.sum(magnitude[mask])) if np.any(mask) else 0.0
        full_fixed = int(round(amplitude * scale))
        quantized = full_fixed >> shift
        bands.append(int(quantized))
        residuals.append(int(full_fixed - (quantized << shift)))

    energy = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
    energy_full_fixed = int(round(energy * scale))
    energy_q = energy_full_fixed >> shift
    energy_residual = energy_full_fixed - (energy_q << shift)
    cache_key = frame_cache_key(clip, normalization, config, frame_index, start)

    return {
        "product_type": FRAME_PRODUCT,
        "product_id": frame_product_id(clip["clip_id"], frame_index),
        "cache_key": cache_key,
        "clip_id": clip["clip_id"],
        "track_id": clip["track_id"],
        "version": int(clip["version"]),
        "role": clip.get("role", "library"),
        "frame_index": frame_index,
        "start_sample": start,
        "sample_count": frame_size,
        "points_cost": frame_size,
        "energy_q": int(energy_q),
        "energy_residual_q": int(energy_residual),
        "bands_q": bands,
        "band_residuals_q": residuals,
        "upstream_refs": [
            {"type": "audio_content", "key": clip.get("content_fingerprint")},
            {"type": "normalized_content", "key": normalization["normalized_content_fingerprint"]},
            {"type": "format", "key": clip["format_fingerprint"]},
            {"type": "spectral_config", "key": spectral_fingerprint(config)},
        ],
    }


def frames_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = ["frame_index", "start_sample", "energy_q", "bands_q"]
    return all(left.get(field) == right.get(field) for field in fields)
