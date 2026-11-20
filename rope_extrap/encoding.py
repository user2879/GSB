from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .canonical import sha256_json
from .errors import E_COMPUTE_NONFINITE, BusinessError
from .models import (
    PRECISION_F32,
    PRUNING_ENERGY,
    PRUNING_FREQUENCY,
    PRUNING_NONE,
    SCHEME_LINEAR,
    SCHEME_NTK,
    AttentionSample,
    EncodedSample,
    PositionEncodingConfig,
)


def scheme_parameter_payload(config: PositionEncodingConfig) -> dict[str, object]:
    return {
        "scheme": config.scheme,
        "base": config.base,
        "scale_factor": config.scale_factor,
        "dimension": config.dimension,
        "precision": config.precision,
        "dimension_chunk_size": config.dimension_chunk_size,
        "pruning_policy": config.pruning_policy,
        "pruning_tolerance": config.pruning_tolerance,
    }


def scheme_parameter_fingerprint(config: PositionEncodingConfig) -> str:
    return sha256_json(scheme_parameter_payload(config))


def effective_base_and_position_scale(config: PositionEncodingConfig) -> tuple[float, float]:
    if config.scheme == SCHEME_LINEAR:
        return config.base, config.scale_factor
    if config.scheme == SCHEME_NTK:
        if config.dimension > 2:
            exponent = config.dimension / (config.dimension - 2)
        else:
            # The usual NTK expression has a removable singularity at d=2.
            exponent = 1.0
        return config.base * (config.scale_factor**exponent), 1.0
    return config.base, 1.0


def _precision_quantize(values: np.ndarray, precision: str) -> np.ndarray:
    if precision == PRECISION_F32:
        return np.asarray(values, dtype=np.float32).astype(np.float64)
    return np.asarray(values, dtype=np.float64)


def _chunks(config: PositionEncodingConfig) -> list[tuple[int, int]]:
    return [
        (start, start + config.dimension_chunk_size)
        for start in range(0, config.dimension, config.dimension_chunk_size)
    ]


def select_kept_chunks(config: PositionEncodingConfig, sample: AttentionSample) -> tuple[int, ...]:
    chunks = _chunks(config)
    if config.pruning_policy == PRUNING_NONE:
        return tuple(range(len(chunks)))
    if config.pruning_policy == PRUNING_FREQUENCY:
        # Low-frequency pairs are first.  The tolerance is the fraction of
        # chunks that may be discarded; one chunk is always retained.
        retain_fraction = 1.0 - min(max(config.pruning_tolerance, 0.0), 0.999999)
        retain_count = max(1, int(math.ceil(len(chunks) * retain_fraction)))
        return tuple(range(retain_count))
    if config.pruning_policy != PRUNING_ENERGY:
        raise ValueError(f"unsupported pruning policy: {config.pruning_policy}")

    query = np.asarray(sample.query_features, dtype=np.float64)
    keys = np.asarray(sample.key_features, dtype=np.float64)
    energies: list[float] = []
    for start, end in chunks:
        key_energy = float(np.mean(np.sum(keys[:, start:end] ** 2, axis=1)))
        query_energy = float(np.sum(query[start:end] ** 2))
        energies.append(query_energy + key_energy)
    maximum = max(energies) if energies else 0.0
    threshold = max(config.pruning_tolerance, 0.0) * maximum
    kept = tuple(index for index, energy in enumerate(energies) if energy >= threshold)
    if not kept:
        kept = (int(np.argmax(np.asarray(energies, dtype=np.float64))),)
    return kept


def _flatten_dimensions(chunks: Sequence[tuple[int, int]], kept_chunks: Iterable[int]) -> tuple[int, ...]:
    dimensions: list[int] = []
    for chunk_index in sorted(kept_chunks):
        start, end = chunks[chunk_index]
        dimensions.extend(range(start, end))
    return tuple(dimensions)


def _rotate_one(
    vector: np.ndarray,
    position: int,
    *,
    inverse_frequencies: np.ndarray,
    position_scale: float,
    kept_dimensions: set[int],
    precision: str,
) -> np.ndarray:
    scaled_position = float(position) / position_scale
    angles = scaled_position * inverse_frequencies
    cosines = np.cos(angles)
    sines = np.sin(angles)
    rotated = np.array(vector, dtype=np.float64, copy=True)
    for pair_index in range(inverse_frequencies.size):
        left = 2 * pair_index
        right = left + 1
        x0 = float(vector[left])
        x1 = float(vector[right])
        rotated[left] = x0 * cosines[pair_index] - x1 * sines[pair_index]
        rotated[right] = x0 * sines[pair_index] + x1 * cosines[pair_index]
    for dimension in range(rotated.size):
        if dimension not in kept_dimensions:
            rotated[dimension] = 0.0
    return _precision_quantize(rotated, precision)


def _ensure_finite(values: np.ndarray, *, sample_id: str, stage: str) -> None:
    if not np.all(np.isfinite(values)):
        raise BusinessError(
            E_COMPUTE_NONFINITE,
            "position encoding produced a non-finite value",
            stage=stage,
            sample_id=sample_id,
        )


def encode_sample(
    config: PositionEncodingConfig,
    sample: AttentionSample,
    *,
    positions: tuple[int, Sequence[int]] | None = None,
    kept_chunks: tuple[int, ...] | None = None,
) -> EncodedSample:
    """Encode query/key features with one of the three rotation schemes.

    ``positions`` exists for the error-tracing stage.  It lets the tracer
    encode the same feature rows at clamped, training-range positions while
    retaining exactly the same dimension mask as the measured encoding.
    """

    if positions is None:
        query_position = sample.query_position
        key_positions = sample.key_positions
    else:
        query_position = int(positions[0])
        key_positions = tuple(int(value) for value in positions[1])

    chunks = _chunks(config)
    selected_chunks = kept_chunks if kept_chunks is not None else select_kept_chunks(config, sample)
    kept_dimensions = _flatten_dimensions(chunks, selected_chunks)
    kept_set = set(kept_dimensions)
    effective_base, position_scale = effective_base_and_position_scale(config)
    pair_count = config.dimension // 2
    pair_indices = np.arange(pair_count, dtype=np.float64)
    inverse_frequencies = 1.0 / np.power(effective_base, (2.0 * pair_indices) / config.dimension)

    raw_query = _precision_quantize(np.asarray(sample.query_features, dtype=np.float64), config.precision)
    raw_keys = _precision_quantize(np.asarray(sample.key_features, dtype=np.float64), config.precision)
    query_vector = _rotate_one(
        raw_query,
        query_position,
        inverse_frequencies=inverse_frequencies,
        position_scale=position_scale,
        kept_dimensions=kept_set,
        precision=config.precision,
    )
    key_vectors = tuple(
        _rotate_one(
            raw_keys[index],
            key_position,
            inverse_frequencies=inverse_frequencies,
            position_scale=position_scale,
            kept_dimensions=kept_set,
            precision=config.precision,
        )
        for index, key_position in enumerate(key_positions)
    )
    for vector in (query_vector, *key_vectors):
        _ensure_finite(vector, sample_id=sample.sample_id, stage="position_encoding")

    pruned_dimensions = tuple(dimension for dimension in range(config.dimension) if dimension not in kept_set)
    fingerprint = sha256_json(
        {
            "scheme": config.scheme,
            "precision": config.precision,
            "query_position": query_position,
            "key_positions": list(key_positions),
            "query_vector": [float(value) for value in query_vector],
            "key_vectors": [[float(value) for value in row] for row in key_vectors],
            "kept_dimensions": list(kept_dimensions),
            "effective_base": effective_base,
            "position_scale": position_scale,
        }
    )
    return EncodedSample(
        query_position=query_position,
        query_vector=tuple(float(value) for value in query_vector),
        key_positions=key_positions,
        key_vectors=tuple(tuple(float(value) for value in row) for row in key_vectors),
        kept_chunks=tuple(selected_chunks),
        kept_dimensions=kept_dimensions,
        pruned_dimensions=pruned_dimensions,
        effective_base=effective_base,
        position_scale=position_scale,
        scheme=config.scheme,
        position_fingerprint=fingerprint,
    )


def position_vector_summary(encoded: EncodedSample) -> dict[str, object]:
    query = np.asarray(encoded.query_vector, dtype=np.float64)
    keys = np.asarray(encoded.key_vectors, dtype=np.float64)
    return {
        "scheme": encoded.scheme,
        "query_norm": float(np.linalg.norm(query)),
        "query_first_4": [float(value) for value in query[:4]],
        "key_norms": [float(np.linalg.norm(row)) for row in keys],
        "kept_chunks": list(encoded.kept_chunks),
        "kept_dimensions": list(encoded.kept_dimensions),
        "pruned_dimensions": list(encoded.pruned_dimensions),
        "effective_base": encoded.effective_base,
        "position_scale": encoded.position_scale,
        "position_fingerprint": encoded.position_fingerprint,
    }

