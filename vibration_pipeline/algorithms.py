from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import AlgorithmSpec, PipelineConfig
from .data import DeviceData, Segment
from .errors import E_ALGO_PARAM_INVALID, E_FEATURE_DEPENDENCY, PipelineError
from .features import FeatureBundle
from .util import hash_object, stable_float


ALGORITHM_FORMULA_VERSION = "segment-algorithms-v1"


@dataclass(frozen=True)
class AlgorithmContext:
    name: str
    params: dict[str, Any]
    params_hash: str
    feature_dependencies: tuple[str, ...]
    baseline_fingerprint: str
    segment_context_fingerprints: dict[int, str]
    context_data: dict[str, Any]
    dependencies_by_segment: dict[int, tuple[tuple[str, int], ...]]

    def referenced_segments(self, segment_index: int) -> tuple[tuple[str, int], ...]:
        return self.dependencies_by_segment.get(segment_index, ())


@dataclass(frozen=True)
class RawScore:
    algorithm: str
    raw_score: float
    cache_key: str
    params_hash: str
    referenced_features: tuple[str, ...]
    referenced_segments: tuple[tuple[str, int], ...]
    details: dict[str, Any]


def algorithm_feature_dependencies(name: str, mode: str) -> tuple[str, ...]:
    if name == "crest_factor":
        return ("rms", "peak")
    if name == "spectral_residual":
        return ("band_energy_0",) if mode == "combined" else ("band_energies",)
    if mode == "time_domain":
        return ("rms", "kurtosis", "peak")
    if mode == "band_energy":
        return ("band_energies",)
    return ("rms", "kurtosis", "peak", "band_energies")


def _feature_content_keys(bundle: FeatureBundle) -> dict[str, str]:
    return {
        "time_key": bundle.time_key,
        "band_key": bundle.band_key,
        "content_fingerprint": bundle.content_fingerprint,
    }


def _build_robust_context(
    device: DeviceData,
    bundles: dict[int, FeatureBundle],
    spec: AlgorithmSpec,
    mode: str,
) -> AlgorithmContext:
    available = sorted(index for index in bundles)
    baseline_fp = hash_object(
        {
            "device": device.device_id,
            "segments": [
                {
                    "segment_index": index,
                    "content_fingerprint": bundles[index].content_fingerprint,
                    "preprocess_fingerprint": bundles[index].preprocessing_fingerprint,
                }
                for index in available
            ],
            "params_hash": spec.params_hash,
            "formula": "robust-mahalanobis-baseline-v1",
        },
        namespace="algorithm-context-v1",
    )
    dependencies = {index: tuple((device.device_id, other) for other in available) for index in available}
    segment_contexts = {index: baseline_fp for index in available}
    return AlgorithmContext(
        name=spec.name,
        params=dict(spec.params),
        params_hash=spec.params_hash,
        feature_dependencies=algorithm_feature_dependencies(spec.name, mode),
        baseline_fingerprint=baseline_fp,
        segment_context_fingerprints=segment_contexts,
        context_data={"feature_matrix": [list(bundles[index].vector) for index in available], "indices": available},
        dependencies_by_segment=dependencies,
    )


def _build_spectral_context(
    device: DeviceData,
    bundles: dict[int, FeatureBundle],
    spec: AlgorithmSpec,
) -> AlgorithmContext:
    available = sorted(index for index in bundles if bundles[index].band_values)
    baseline_fp = hash_object(
        {
            "device": device.device_id,
            "segments": [
                {
                    "segment_index": index,
                    "band_key": bundles[index].band_key,
                    "band_values": list(bundles[index].band_values),
                }
                for index in available
            ],
            "params_hash": spec.params_hash,
            "formula": "spectral-baseline-v1",
        },
        namespace="algorithm-context-v1",
    )
    matrix = np.asarray([bundles[index].band_values for index in available], dtype=np.float64)
    method = spec.params.get("baseline_method", "median")
    if matrix.size:
        baseline = np.median(matrix, axis=0) if method == "median" else np.mean(matrix, axis=0)
        baseline_values = [stable_float(value) for value in baseline.tolist()]
    else:
        baseline_values = []
    dependencies = {index: tuple((device.device_id, other) for other in available) for index in available}
    segment_contexts = {index: baseline_fp for index in available}
    return AlgorithmContext(
        name=spec.name,
        params=dict(spec.params),
        params_hash=spec.params_hash,
        feature_dependencies=("band_energies",),
        baseline_fingerprint=baseline_fp,
        segment_context_fingerprints=segment_contexts,
        context_data={"baseline_values": baseline_values},
        dependencies_by_segment=dependencies,
    )


def _build_crest_context(
    device: DeviceData,
    bundles: dict[int, FeatureBundle],
    spec: AlgorithmSpec,
) -> AlgorithmContext:
    window_length = int(spec.params.get("window_length", 3))
    available = sorted(bundles)
    segment_crest: dict[int, float] = {}
    for index in available:
        vector = bundles[index].vector
        if len(vector) < 3:
            continue
        rms = float(vector[0])
        peak = float(vector[2])
        segment_crest[index] = peak / rms if rms > 0 else 0.0
    window_by_segment: dict[int, tuple[int, ...]] = {}
    for position, index in enumerate(available):
        candidates = [candidate for candidate in available[max(0, position - window_length + 1) : position + 1] if candidate in segment_crest]
        window_by_segment[index] = tuple(candidates)
    baseline_fp = hash_object(
        {
            "device": device.device_id,
            "windows": {
                str(index): [
                    {"segment_index": other, "time_key": bundles[other].time_key, "crest": stable_float(segment_crest[other])}
                    for other in window
                ]
                for index, window in sorted(window_by_segment.items())
            },
            "params_hash": spec.params_hash,
            "formula": "crest-window-v1",
        },
        namespace="algorithm-context-v1",
    )
    dependencies = {index: tuple((device.device_id, other) for other in window) for index, window in window_by_segment.items()}
    segment_contexts = {
        index: hash_object(
            {
                "device": device.device_id,
                "segment_index": index,
                "window": [
                    {"segment_index": other, "time_key": bundles[other].time_key, "crest": stable_float(segment_crest[other])}
                    for other in window
                ],
                "params_hash": spec.params_hash,
                "formula": "crest-window-v1",
            },
            namespace="algorithm-segment-context-v1",
        )
        for index, window in window_by_segment.items()
    }
    return AlgorithmContext(
        name=spec.name,
        params=dict(spec.params),
        params_hash=spec.params_hash,
        feature_dependencies=("rms", "peak"),
        baseline_fingerprint=baseline_fp,
        segment_context_fingerprints=segment_contexts,
        context_data={"window_by_segment": window_by_segment, "segment_crest": segment_crest},
        dependencies_by_segment=dependencies,
    )


def build_algorithm_contexts(
    device: DeviceData,
    bundles: dict[int, FeatureBundle],
    config: PipelineConfig,
) -> dict[str, AlgorithmContext]:
    contexts: dict[str, AlgorithmContext] = {}
    for name in config.enabled_algorithms:
        spec = config.algorithms[name]
        if name == "robust_mahalanobis":
            contexts[name] = _build_robust_context(device, bundles, spec, config.features.mode)
        elif name == "spectral_residual":
            contexts[name] = _build_spectral_context(device, bundles, spec)
        else:
            contexts[name] = _build_crest_context(device, bundles, spec)
    return contexts


def raw_score_cache_key(
    *,
    segment: Segment,
    bundle: FeatureBundle,
    context: AlgorithmContext,
) -> str:
    if context.name == "crest_factor":
        feature_binding = {"time_key": bundle.time_key, "feature_names": ["rms", "peak"]}
    elif context.name == "spectral_residual":
        feature_binding = {"band_key": bundle.band_key, "content_fingerprint": bundle.content_fingerprint}
    else:
        feature_binding = _feature_content_keys(bundle)
    return hash_object(
        {
            "formula": ALGORITHM_FORMULA_VERSION,
            "algorithm": context.name,
            "device_id": segment.device_id,
            "segment_index": segment.segment_index,
            "waveform_fingerprint": segment.waveform_fingerprint,
            "preprocess_fingerprint": segment.preprocess_fingerprint,
            "segment_length": segment.segment_length,
            "params_hash": context.params_hash,
            "feature_binding": feature_binding,
            "context_fingerprint": context.segment_context_fingerprints.get(
                segment.segment_index, context.baseline_fingerprint
            ),
        },
        namespace="algorithm-raw-score-key-v1",
    )


def _require_band_features(bundle: FeatureBundle) -> None:
    if not bundle.band_values:
        raise PipelineError(E_FEATURE_DEPENDENCY, "spectral_residual requires band-energy features")


def _require_time_features(bundle: FeatureBundle) -> tuple[float, float]:
    if len(bundle.vector) < 3 or bundle.feature_names[:3] != ("rms", "kurtosis", "peak"):
        raise PipelineError(E_FEATURE_DEPENDENCY, "crest_factor requires RMS and peak time-domain features")
    return float(bundle.vector[0]), float(bundle.vector[2])


def score_segment(
    *,
    segment: Segment,
    bundle: FeatureBundle,
    context: AlgorithmContext,
    injected_window_length: int | None = None,
) -> RawScore:
    cache_key = raw_score_cache_key(segment=segment, bundle=bundle, context=context)
    dependencies = context.referenced_segments(segment.segment_index)
    segment_context_fingerprint = context.segment_context_fingerprints.get(
        segment.segment_index, context.baseline_fingerprint
    )
    base_details: dict[str, Any] = {
        "feature_names": list(context.feature_dependencies),
        "referenced_segments": [{"device_id": device_id, "segment_index": index} for device_id, index in dependencies],
        "baseline_fingerprint": segment_context_fingerprint,
        "segment_context_fingerprint": segment_context_fingerprint,
    }
    if context.name == "robust_mahalanobis":
        matrix = np.asarray(context.context_data.get("feature_matrix", []), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(bundle.vector) or matrix.shape[0] == 0:
            raise PipelineError(E_FEATURE_DEPENDENCY, "robust covariance baseline is unavailable")
        median = np.median(matrix, axis=0)
        deviations = np.abs(matrix - median)
        mad = np.median(deviations, axis=0)
        floor = float(context.params.get("scale_floor", 1e-9))
        scale = 1.4826 * mad
        reference = np.maximum(np.abs(median), 1.0)
        scale = np.maximum(scale, floor * reference)
        normalized = (matrix - median) / scale
        target = (np.asarray(bundle.vector, dtype=np.float64) - median) / scale
        if matrix.shape[0] > 1:
            covariance = np.cov(normalized, rowvar=False, ddof=1)
            covariance = np.atleast_2d(covariance)
            # A rank-deficient robust covariance is expected when there are fewer
            # segments than features. A relative pseudo-inverse cut-off avoids
            # amplifying numerical null-space eigenvalues into enormous scores.
            inverse = np.linalg.pinv(covariance, rcond=1e-10, hermitian=True)
            quadratic = float(target @ inverse @ target)
        else:
            quadratic = 0.0
        robust_scale = float(context.params.get("robust_scale", 1.4826))
        raw = math.sqrt(max(quadratic, 0.0)) * robust_scale
        base_details.update(
            {
                "median": [stable_float(value) for value in median.tolist()],
                "robust_scale_vector": [stable_float(value) for value in scale.tolist()],
                "quadratic_form": stable_float(quadratic),
                "robust_scale": robust_scale,
            }
        )
    elif context.name == "spectral_residual":
        _require_band_features(bundle)
        baseline = np.asarray(context.context_data.get("baseline_values", []), dtype=np.float64)
        energies = np.asarray(bundle.band_values, dtype=np.float64)
        if baseline.shape != energies.shape or baseline.size == 0:
            raise PipelineError(E_FEATURE_DEPENDENCY, "spectral baseline has a different band shape")
        epsilon = float(context.params.get("epsilon", 1e-12))
        ratios = energies / np.maximum(baseline, epsilon)
        raw = float(np.mean(ratios, dtype=np.float64))
        base_details.update(
            {
                "baseline_values": [stable_float(value) for value in baseline.tolist()],
                "ratios": [stable_float(value) for value in ratios.tolist()],
            }
        )
    elif context.name == "crest_factor":
        rms, peak = _require_time_features(bundle)
        window_length = int(context.params.get("window_length", 3)) if injected_window_length is None else int(injected_window_length)
        if window_length <= 0:
            raise PipelineError(E_ALGO_PARAM_INVALID, "crest_factor window_length must be a positive integer")
        windows = context.context_data.get("window_by_segment", {})
        window = tuple(windows.get(segment.segment_index, ()))[-window_length:]
        crest_values = context.context_data.get("segment_crest", {})
        if not window:
            crest = peak / rms if rms > 0 else 0.0
            raw = crest
            window = (segment.segment_index,)
            selected = [crest]
        else:
            selected = [float(crest_values[index]) for index in window]
            raw = float(np.mean(np.asarray(selected, dtype=np.float64)))
        base_details.update(
            {
                "window_length": window_length,
                "window_segment_indices": list(window),
                "window_crest_values": [stable_float(value) for value in selected],
                "segment_crest": stable_float(peak / rms if rms > 0 else 0.0),
            }
        )
    else:
        raise PipelineError(E_ALGO_PARAM_INVALID, f"unsupported algorithm {context.name}")
    raw = stable_float(raw)
    base_details["raw_score"] = raw
    return RawScore(
        algorithm=context.name,
        raw_score=raw,
        cache_key=cache_key,
        params_hash=context.params_hash,
        referenced_features=context.feature_dependencies,
        referenced_segments=dependencies,
        details=base_details,
    )


def algorithm_dependency_report(contexts: dict[str, AlgorithmContext], device_id: str) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in sorted(contexts):
        context = contexts[name]
        all_segments = sorted(
            {
                (referenced_device, referenced_index)
                for references in context.dependencies_by_segment.values()
                for referenced_device, referenced_index in references
            }
        )
        report[name] = {
            "feature_set": list(context.feature_dependencies),
            "segment_set": [
                {"device_id": referenced_device, "segment_index": referenced_index}
                for referenced_device, referenced_index in all_segments
                if referenced_device == device_id
            ],
            "params_hash": context.params_hash,
            "baseline_fingerprint": context.baseline_fingerprint,
        }
    return report
