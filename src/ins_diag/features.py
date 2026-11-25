from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import Band, EffectiveConfig
from .errors import BusinessError, WindowProcessingError
from .util import fingerprint
from .windows import WindowRef, window_set_fingerprint


def _feature_names(bands: tuple[Band, ...]) -> list[str]:
    return ["rms", "kurtosis"] + [
        f"band_energy_{index}_{band.low_hz:g}_{band.high_hz:g}"
        for index, band in enumerate(bands)
    ]


def extract_window_features(window: WindowRef, config: EffectiveConfig) -> dict[str, Any]:
    values = window.samples
    present = window.present
    present_positions = np.flatnonzero(present)
    if present_positions.size == 0:
        raise WindowProcessingError(
            "window contains no present samples",
            path=f"windows[{window.window_index}].present",
            window_id=window.window_id,
            code="E_WINDOW_ALL_MISSING",
        )
    for position in present_positions:
        value = float(values[int(position)])
        if not math.isfinite(value):
            raise WindowProcessingError(
                "non-finite sample in present position",
                path=f"windows[{window.window_index}].samples[{window.start + int(position)}]",
                window_id=window.window_id,
                code="E_NONFINITE_SAMPLE",
            ) from None

    seen_values = [float(values[int(position)]) for position in present_positions]
    seen_mean = math.fsum(seen_values) / len(seen_values)
    filled = np.asarray(values, dtype=np.float64).copy()
    missing_positions = np.flatnonzero(~present)
    for position in missing_positions:
        filled[int(position)] = seen_mean

    sample_count = window.sample_count
    rms = math.sqrt(math.fsum(float(value) * float(value) for value in filled) / sample_count)
    mean = math.fsum(float(value) for value in filled) / sample_count
    second = 0.0
    fourth = 0.0
    for value in filled:
        centered = float(value) - mean
        squared = centered * centered
        second += squared
        fourth += squared * squared
    second /= sample_count
    fourth /= sample_count
    kurtosis = 0.0 if second == 0.0 else fourth / (second * second) - 3.0

    spectrum = np.fft.rfft(np.asarray(filled, dtype=np.float64))
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / window.sample_rate_hz)
    band_energies: list[float] = []
    for band in config.bands:
        terms: list[float] = []
        for bin_index in range(spectrum.size):
            frequency = float(frequencies[bin_index])
            if band.low_hz <= frequency < band.high_hz:
                magnitude = abs(complex(spectrum[bin_index]))
                terms.append(magnitude * magnitude / sample_count)
        band_energies.append(math.fsum(terms))

    feature_values = [rms, kurtosis] + band_energies
    fill_source = {
        "rule": "none" if missing_positions.size == 0 else "window_seen_mean",
        "missing_count": int(missing_positions.size),
        "seen_count": int(present_positions.size),
        "missing_indices": [int(position) for position in missing_positions],
        "missing_sample_sources": [
            {"offset": int(position), "source": "window_seen_mean"}
            for position in missing_positions
        ],
        "seen_mean": seen_mean,
    }
    return {
        "window_index": window.window_index,
        "window_id": window.window_id,
        "content_fingerprint": window.content_fingerprint(),
        "feature_names": _feature_names(config.bands),
        "values": feature_values,
        "fill_source": fill_source,
    }


def failure_to_error(exc: WindowProcessingError, window: WindowRef) -> BusinessError:
    return BusinessError(
        code=exc.code,
        path=exc.path,
        message=exc.message,
        stage="window",
        window_id=window.window_id,
    )


def compute_standardization(
    raw_features: list[dict[str, Any]],
    config: EffectiveConfig,
) -> dict[str, Any]:
    ordered = sorted(raw_features, key=lambda item: item["window_index"])
    if not ordered:
        raise WindowProcessingError(
            "no valid windows are available for standardization",
            path="windows",
        )
    dimension = len(ordered[0]["values"])
    means: list[float] = []
    dispersions: list[float] = []
    zero_dispersion_dimensions: list[int] = []
    for dimension_index in range(dimension):
        values = [float(item["values"][dimension_index]) for item in ordered]
        mean = math.fsum(values) / len(values)
        variance_terms = [(value - mean) * (value - mean) for value in values]
        variance = math.fsum(variance_terms) / len(values)
        dispersion = math.sqrt(variance)
        if dispersion == 0.0:
            zero_dispersion_dimensions.append(dimension_index)
            dispersion = 1.0
        means.append(mean)
        dispersions.append(dispersion)
    window_ids = [item["window_id"] for item in ordered]
    payload = {
        "window_set_fingerprint": window_set_fingerprint_ids(
            ordered, config
        ),
        "window_count": len(ordered),
        "window_ids": window_ids,
        "dimension": dimension,
        "means": means,
        "dispersions": dispersions,
        "zero_dispersion_dimensions": zero_dispersion_dimensions,
        "source_window_indices": [item["window_index"] for item in ordered],
    }
    payload["standardization_fingerprint"] = fingerprint(
        {
            "window_set_fingerprint": payload["window_set_fingerprint"],
            "window_ids": window_ids,
            "means": means,
            "dispersions": dispersions,
            "parameter_fingerprint": config.standardization_fingerprint(),
        }
    )
    return payload


def window_set_fingerprint_ids(
    ordered_raw_features: list[dict[str, Any]], config: EffectiveConfig
) -> str:
    return fingerprint(
        {
            "parameter_fingerprint": config.feature_parameter_fingerprint(),
            "windows": [
                {
                    "window_index": item["window_index"],
                    "window_id": item["window_id"],
                    "content_fingerprint": item["content_fingerprint"],
                }
                for item in ordered_raw_features
            ],
        }
    )


def standardize_observations(
    raw_features: list[dict[str, Any]],
    stats: dict[str, Any],
) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    means = stats["means"]
    dispersions = stats["dispersions"]
    for item in sorted(raw_features, key=lambda value: value["window_index"]):
        vector = []
        for dimension_index, value in enumerate(item["values"]):
            vector.append((float(value) - means[dimension_index]) / dispersions[dimension_index])
        result[int(item["window_index"])] = vector
    return result
