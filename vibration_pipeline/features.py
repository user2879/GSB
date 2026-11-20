from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import E_BAND_INVALID, E_NONFINITE_SAMPLE, PipelineError
from .util import hash_object, stable_float


TIME_FEATURE_NAMES = ("rms", "kurtosis", "peak")
FEATURE_FORMULA_VERSION = "time-band-features-v1"


@dataclass(frozen=True)
class FeatureBundle:
    time_key: str
    band_key: str
    time_values: tuple[float, ...]
    band_values: tuple[float, ...]
    vector: tuple[float, ...]
    feature_names: tuple[str, ...]
    content_fingerprint: str
    preprocessing_fingerprint: str

    def summary(self, *, mode: str) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if self.time_values:
            summary.update({name: value for name, value in zip(TIME_FEATURE_NAMES, self.time_values, strict=True)})
        if self.band_values:
            summary["band_energies"] = list(self.band_values)
        summary["mode"] = mode
        summary["vector_order"] = list(self.feature_names)
        return summary


def validate_finite(samples: np.ndarray) -> None:
    array = np.asarray(samples, dtype=np.float64)
    if array.size == 0:
        raise PipelineError(E_NONFINITE_SAMPLE, "segment contains no samples")
    if not np.all(np.isfinite(array)):
        bad = int(np.flatnonzero(~np.isfinite(array))[0])
        raise PipelineError(E_NONFINITE_SAMPLE, f"non-finite sample value at local index {bad}")


def validate_band_edges(band_edges_hz: tuple[float, ...], sample_rate_hz: float) -> None:
    if len(band_edges_hz) < 2:
        raise PipelineError(E_BAND_INVALID, "at least two band boundaries are required")
    if any(not math.isfinite(float(edge)) for edge in band_edges_hz):
        raise PipelineError(E_BAND_INVALID, "band boundaries must be finite")
    if float(band_edges_hz[0]) < 0:
        raise PipelineError(E_BAND_INVALID, "first band boundary must be >= 0")
    if any(float(left) >= float(right) for left, right in zip(band_edges_hz, band_edges_hz[1:])):
        raise PipelineError(E_BAND_INVALID, "band boundaries must be strictly increasing")
    nyquist = float(sample_rate_hz) / 2.0
    if float(band_edges_hz[-1]) > nyquist + 1e-12:
        raise PipelineError(E_BAND_INVALID, f"band boundary exceeds Nyquist ({nyquist})")


def time_feature_key(waveform_fp: str, preprocess_fp: str, segment_length: int) -> str:
    return hash_object(
        {
            "formula": FEATURE_FORMULA_VERSION,
            "waveform_fingerprint": waveform_fp,
            "preprocess_fingerprint": preprocess_fp,
            "segment_length": segment_length,
            "feature_kind": "time",
        },
        namespace="feature-cache-key-v1",
    )


def band_feature_key(
    waveform_fp: str,
    preprocess_fp: str,
    segment_length: int,
    sample_rate_hz: float,
    band_edges_hz: tuple[float, ...],
) -> str:
    return hash_object(
        {
            "formula": FEATURE_FORMULA_VERSION,
            "waveform_fingerprint": waveform_fp,
            "preprocess_fingerprint": preprocess_fp,
            "segment_length": segment_length,
            "sample_rate_hz": sample_rate_hz,
            "band_edges_hz": list(band_edges_hz),
            "feature_kind": "band",
        },
        namespace="feature-cache-key-v1",
    )


def extract_time_features(samples: np.ndarray) -> tuple[float, ...]:
    validate_finite(samples)
    array = np.asarray(samples, dtype=np.float64)
    mean = float(np.mean(array))
    centered = array - mean
    second = float(np.mean(centered * centered))
    fourth = float(np.mean(centered * centered * centered * centered))
    rms = math.sqrt(float(np.mean(array * array)))
    kurtosis = fourth / (second * second) if second > 0.0 else 0.0
    peak = float(np.max(np.abs(array)))
    return tuple(stable_float(value) for value in (rms, kurtosis, peak))


def extract_band_energies(samples: np.ndarray, sample_rate_hz: float, band_edges_hz: tuple[float, ...]) -> tuple[float, ...]:
    validate_finite(samples)
    validate_band_edges(band_edges_hz, sample_rate_hz)
    array = np.asarray(samples, dtype=np.float64)
    n = int(array.size)
    spectrum = np.fft.rfft(array)
    frequencies = np.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz))
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag) / float(n * n)
    energies: list[float] = []
    last_band = len(band_edges_hz) - 2
    for band_index in range(len(band_edges_hz) - 1):
        lower = float(band_edges_hz[band_index])
        upper = float(band_edges_hz[band_index + 1])
        if band_index == last_band:
            mask = (frequencies >= lower) & (frequencies <= upper)
        else:
            mask = (frequencies >= lower) & (frequencies < upper)
        energies.append(stable_float(float(np.sum(power[mask], dtype=np.float64))))
    return tuple(energies)


def extract_features(
    samples: np.ndarray,
    *,
    segment_length: int,
    sample_rate_hz: float,
    band_edges_hz: tuple[float, ...],
    need_time: bool,
    need_band: bool,
    waveform_fingerprint: str,
    preprocessing_fingerprint: str,
    mode: str,
) -> FeatureBundle:
    validate_finite(samples)
    t_key = time_feature_key(waveform_fingerprint, preprocessing_fingerprint, segment_length)
    b_key = band_feature_key(
        waveform_fingerprint,
        preprocessing_fingerprint,
        segment_length,
        sample_rate_hz,
        band_edges_hz,
    )
    time_values = extract_time_features(samples) if need_time else ()
    band_values = extract_band_energies(samples, sample_rate_hz, band_edges_hz) if need_band else ()
    return compose_feature_bundle(
        time_values=time_values,
        band_values=band_values,
        time_key_value=t_key,
        band_key_value=b_key,
        segment_length=segment_length,
        band_edges_hz=band_edges_hz,
        waveform_fingerprint=waveform_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        mode=mode,
    )


def compose_feature_bundle(
    *,
    time_values: tuple[float, ...],
    band_values: tuple[float, ...],
    time_key_value: str,
    band_key_value: str,
    segment_length: int,
    band_edges_hz: tuple[float, ...],
    waveform_fingerprint: str,
    preprocessing_fingerprint: str,
    mode: str,
) -> FeatureBundle:
    need_time = bool(time_values)
    need_band = bool(band_values)
    names: list[str] = []
    vector: list[float] = []
    if need_time:
        names.extend(TIME_FEATURE_NAMES)
        vector.extend(time_values)
    if need_band:
        names.extend(f"band_energy_{index}" for index in range(len(band_values)))
        vector.extend(band_values)
    rounded_vector = tuple(stable_float(value) for value in vector)
    content_fp = hash_object(
        {
            "formula": FEATURE_FORMULA_VERSION,
            "waveform_fingerprint": waveform_fingerprint,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "segment_length": segment_length,
            "band_edges_hz": list(band_edges_hz),
            "mode": mode,
            "feature_names": names,
            "feature_values": list(rounded_vector),
        },
        namespace="feature-content-v1",
    )
    return FeatureBundle(
        time_key=time_key_value,
        band_key=band_key_value,
        time_values=time_values,
        band_values=band_values,
        vector=rounded_vector,
        feature_names=tuple(names),
        content_fingerprint=content_fp,
        preprocessing_fingerprint=preprocessing_fingerprint,
    )


def feature_values_by_name(bundle: FeatureBundle) -> dict[str, float]:
    return {name: value for name, value in zip(bundle.feature_names, bundle.vector, strict=True)}
