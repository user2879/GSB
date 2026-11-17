"""Task-independent shared encoder forward pass and representation cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical import ALGORITHM_VERSION, fingerprint
from .data import EncoderPackage, FeatureSample
from .errors import SharedForwardError


def representation_key(sample: FeatureSample, encoder: EncoderPackage) -> str:
    """Content-addressed key for one frozen shared representation."""

    return fingerprint(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "sample_id": sample.sample_id,
            "feature_fingerprint": sample.feature_fingerprint(),
            "encoder_fingerprint": encoder.fingerprint(),
        },
        domain="shared-representation",
    )


def forward_feature(sample: FeatureSample, encoder: EncoderPackage) -> np.ndarray:
    """Apply deterministic normalization, projection, and bias."""

    vector = sample.as_array()
    if vector.shape[0] != encoder.input_dim:
        raise SharedForwardError(
            f"sample {sample.sample_id!r} feature dimension does not match encoder input"
        )
    standardized = (vector - encoder.mean) / encoder.std
    representation = np.dot(standardized, encoder.weights.T) + encoder.bias
    if not np.isfinite(representation).all():
        raise SharedForwardError(
            f"shared forward produced a non-finite representation for {sample.sample_id!r}"
        )
    return np.ascontiguousarray(representation, dtype=np.float64)


@dataclass(frozen=True)
class RepresentationBuild:
    current: Mapping[str, Mapping[str, Any]]
    cache: Mapping[str, Mapping[str, Any]]
    reused_samples: tuple[str, ...]
    recomputed_samples: tuple[str, ...]


def build_representations(
    samples: Mapping[str, FeatureSample],
    encoder: EncoderPackage,
    referenced_ids: Sequence[str],
    previous_cache: Mapping[str, Mapping[str, Any]],
    *,
    force_full: bool = False,
) -> RepresentationBuild:
    """Reuse exact content cache entries; forward only missing sample keys."""

    cache: dict[str, Mapping[str, Any]] = {
        str(key): dict(value) for key, value in previous_cache.items()
    }
    current: dict[str, Mapping[str, Any]] = {}
    reused: list[str] = []
    recomputed: list[str] = []
    for sample_id in sorted(set(referenced_ids)):
        if sample_id not in samples:
            raise SharedForwardError(f"sample {sample_id!r} is unavailable for shared forward")
        sample = samples[sample_id]
        key = representation_key(sample, encoder)
        cached = None if force_full else cache.get(key)
        if cached is not None:
            vector = cached.get("vector")
            if (
                isinstance(vector, list)
                and len(vector) == encoder.output_dim
                and all(isinstance(value, (int, float)) for value in vector)
            ):
                entry = dict(cached)
                reused.append(sample_id)
            else:
                cached = None
        if cached is None:
            vector_array = forward_feature(sample, encoder)
            entry = {
                "sample_id": sample_id,
                "key": key,
                "feature_fingerprint": sample.feature_fingerprint(),
                "encoder_fingerprint": encoder.fingerprint(),
                "vector": vector_array.tolist(),
                "artifact_version": None,
            }
            cache[key] = entry
            recomputed.append(sample_id)
        current[sample_id] = entry
    return RepresentationBuild(
        current=current,
        cache=cache,
        reused_samples=tuple(reused),
        recomputed_samples=tuple(recomputed),
    )
