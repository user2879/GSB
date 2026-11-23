from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .models import ALL_SCHEMES, AttentionSample, PositionConfig


@dataclass(frozen=True)
class EncodedSequence:
    scheme: str
    query_position: int
    key_positions: tuple[int, ...]
    query_vector: np.ndarray
    key_vectors: tuple[np.ndarray, ...]
    query_effective_position: float
    key_effective_positions: tuple[float, ...]
    frequency_fingerprint: str

    @property
    def all_vectors(self) -> tuple[np.ndarray, ...]:
        return (self.query_vector,) + self.key_vectors


def scheme_parameters(config: PositionConfig, scheme: str) -> dict[str, Any]:
    if scheme not in ALL_SCHEMES:
        raise ValueError(f"unknown scheme: {scheme}")
    return {
        "scheme": scheme,
        "base_frequency": config.base_frequency,
        "training_context_length": config.training_context_length,
        "extrapolation_target_length": config.extrapolation_target_length,
        "dimension": config.dimension,
        "scaling_factor": config.scaling_factor,
    }


def _frequency_parameters(config: PositionConfig, scheme: str) -> tuple[np.ndarray, float]:
    dimension = config.dimension
    pair_indices = np.arange(0, dimension, 2, dtype=np.float64)
    exponent = pair_indices / float(dimension)

    if scheme == "rope_standard":
        base = config.base_frequency
        position_scale = 1.0
    elif scheme == "rope_linear":
        base = config.base_frequency
        position_scale = 1.0 / config.scaling_factor
    elif scheme == "rope_ntk":
        ntk_exponent = dimension / (dimension - 2.0) if dimension > 2 else 1.0
        base = config.base_frequency * (config.scaling_factor ** ntk_exponent)
        position_scale = 1.0
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    inv_frequency = 1.0 / np.power(float(base), exponent, dtype=np.float64)
    return inv_frequency, float(position_scale)


def _frequency_fingerprint(inv_frequency: np.ndarray, position_scale: float, scheme: str) -> str:
    digest = hashlib.sha256()
    digest.update(scheme.encode("ascii"))
    digest.update(np.asarray(inv_frequency, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray([position_scale], dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _rotate_vectors(vectors: Sequence[np.ndarray], positions: Sequence[float], inv_frequency: np.ndarray) -> tuple[np.ndarray, ...]:
    if not vectors:
        return ()
    stacked = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors], axis=0)
    if stacked.ndim != 2 or stacked.shape[1] != inv_frequency.size * 2:
        raise ValueError("vector dimension does not match frequency table")
    pos = np.asarray(positions, dtype=np.float64)
    angles = pos[:, None] * inv_frequency[None, :]
    cosines = np.cos(angles)
    sines = np.sin(angles)
    even = stacked[:, 0::2]
    odd = stacked[:, 1::2]
    rotated_even = even * cosines - odd * sines
    rotated_odd = even * sines + odd * cosines
    rotated = np.empty_like(stacked)
    rotated[:, 0::2] = rotated_even
    rotated[:, 1::2] = rotated_odd
    return tuple(np.ascontiguousarray(rotated[index], dtype=np.float64) for index in range(rotated.shape[0]))


def encode_sequence(
    config: PositionConfig,
    sample: AttentionSample,
    scheme: str,
    *,
    query_position: int | None = None,
    key_positions: Iterable[int] | None = None,
) -> EncodedSequence:
    if scheme not in ALL_SCHEMES:
        raise ValueError(f"unknown scheme: {scheme}")
    q_pos = sample.query_position if query_position is None else int(query_position)
    k_positions = tuple(int(item) for item in (sample.key_positions if key_positions is None else key_positions))
    if len(k_positions) != len(sample.key_vectors):
        raise ValueError("key position/vector count mismatch")

    inv_frequency, position_scale = _frequency_parameters(config, scheme)
    effective_positions = [position_scale * float(position) for position in (q_pos,) + k_positions]
    vectors = (sample.query_vector,) + sample.key_vectors
    rotated = _rotate_vectors(vectors, effective_positions, inv_frequency)
    return EncodedSequence(
        scheme=scheme,
        query_position=q_pos,
        key_positions=k_positions,
        query_vector=rotated[0],
        key_vectors=rotated[1:],
        query_effective_position=effective_positions[0],
        key_effective_positions=tuple(effective_positions[1:]),
        frequency_fingerprint=_frequency_fingerprint(inv_frequency, position_scale, scheme),
    )


def position_digest(encoded: EncodedSequence, config: PositionConfig) -> str:
    digest = hashlib.sha256()
    digest.update(b"posdiag-position-v1\0")
    digest.update(encoded.scheme.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(config.dimension).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(encoded.query_position).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in encoded.key_positions).encode("ascii"))
    digest.update(b"\0")
    for vector in encoded.all_vectors:
        digest.update(np.asarray(vector, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def vector_arrays_from_payload(payload: dict[str, Any]) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    query = np.asarray(payload["query_vector"], dtype=np.float64)
    keys = tuple(np.asarray(vector, dtype=np.float64) for vector in payload["key_vectors"])
    return query, keys


def encoding_cache_entry(config: PositionConfig, sample: AttentionSample, scheme: str, encoded: EncodedSequence) -> dict[str, Any]:
    query = np.asarray(encoded.query_vector, dtype=np.float64)
    key_vectors = [np.asarray(vector, dtype=np.float64) for vector in encoded.key_vectors]
    return {
        "stage": "encoding",
        "sample_id": sample.sample_id,
        "scheme": scheme,
        "query_position": encoded.query_position,
        "key_positions": list(encoded.key_positions),
        "query_vector": [float(value) for value in query.tolist()],
        "key_vectors": [[float(value) for value in vector.tolist()] for vector in key_vectors],
        "query_effective_position": float(encoded.query_effective_position),
        "key_effective_positions": [float(value) for value in encoded.key_effective_positions],
        "frequency_fingerprint": encoded.frequency_fingerprint,
        "position_digest": position_digest(encoded, config),
        "query_norm": float(np.linalg.norm(query)),
        "key_norms": [float(np.linalg.norm(vector)) for vector in key_vectors],
        "rotation_pair_count": config.dimension // 2,
    }


def finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite numeric result")
    return value
