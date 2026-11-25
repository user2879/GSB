"""Deterministic integer-only speaker scoring forward pass."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .canonical import canonical_bytes, canonical_sha256, sha256_hex
from .spec import WorkerSpec


_WORKER_SALT = 0x53504B44


@dataclass(frozen=True)
class ParameterSet:
    worker_id: str
    parameter_version: int
    seed: int
    model_dim: int
    hidden_dim: int
    speaker_count: int
    quantization_scheme: str
    fractional_bits: int
    weight_input: np.ndarray
    weight_output: np.ndarray
    digest: str
    snapshot: dict[str, Any]

    @property
    def shift(self) -> int:
        return self.fractional_bits


@dataclass(frozen=True)
class InferenceResult:
    attribution: str
    scores: tuple[int, ...]
    confidence_q12: int
    margin: int
    tie_break_used: bool


def _worker_identity(worker_id: str) -> int:
    return int.from_bytes(hashlib.sha256(worker_id.encode("utf-8")).digest()[:8], "little")


def build_parameters(worker: WorkerSpec, speaker_count: int) -> ParameterSet:
    fractional_bits = worker.quantization.fractional_bits
    weight_bound = 1 << min(fractional_bits, 15)
    seed_sequence = np.random.SeedSequence(
        [
            worker.seed,
            _worker_identity(worker.worker_id),
            worker.parameter_version,
            worker.model_dim,
            worker.hidden_dim,
            speaker_count,
            fractional_bits,
            int.from_bytes(hashlib.sha256(worker.quantization.scheme.encode("utf-8")).digest()[:4], "little"),
            _WORKER_SALT,
        ]
    )
    generator = np.random.Generator(np.random.PCG64(seed_sequence))
    weight_input = generator.integers(
        -weight_bound,
        weight_bound,
        size=(worker.model_dim, worker.hidden_dim),
        dtype=np.int64,
        endpoint=False,
    )
    weight_output = generator.integers(
        -weight_bound,
        weight_bound,
        size=(worker.hidden_dim, speaker_count),
        dtype=np.int64,
        endpoint=False,
    )
    metadata = {
        "worker_id": worker.worker_id,
        "parameter_version": worker.parameter_version,
        "seed": worker.seed,
        "model_dim": worker.model_dim,
        "hidden_dim": worker.hidden_dim,
        "speaker_count": speaker_count,
        "scheme": worker.quantization.scheme,
        "fractional_bits": fractional_bits,
        "truncation": "arithmetic_shift_floor",
    }
    digest = sha256_hex(
        canonical_bytes(metadata)
        + weight_input.astype("<i8", copy=False).tobytes(order="C")
        + weight_output.astype("<i8", copy=False).tobytes(order="C")
    )
    snapshot = {
        **metadata,
        "digest": digest,
        "weight_input_sha256": sha256_hex(weight_input.astype("<i8", copy=False).tobytes(order="C")),
        "weight_output_sha256": sha256_hex(weight_output.astype("<i8", copy=False).tobytes(order="C")),
    }
    return ParameterSet(
        worker_id=worker.worker_id,
        parameter_version=worker.parameter_version,
        seed=worker.seed,
        model_dim=worker.model_dim,
        hidden_dim=worker.hidden_dim,
        speaker_count=speaker_count,
        quantization_scheme=worker.quantization.scheme,
        fractional_bits=fractional_bits,
        weight_input=weight_input,
        weight_output=weight_output,
        digest=digest,
        snapshot=snapshot,
    )


class ParameterCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int, int, str, int], ParameterSet] = {}

    def get(self, worker: WorkerSpec, speaker_count: int) -> ParameterSet:
        key = (
            worker.worker_id,
            worker.parameter_version,
            worker.seed,
            worker.model_dim,
            worker.quantization.scheme,
            worker.quantization.fractional_bits,
        )
        parameters = self._cache.get(key)
        if parameters is None:
            parameters = build_parameters(worker, speaker_count)
            self._cache[key] = parameters
        return parameters


def integer_forward(parameters: ParameterSet, features: Sequence[int]) -> InferenceResult:
    if len(features) != parameters.model_dim:
        raise ValueError(f"expected {parameters.model_dim} features, got {len(features)}")
    vector = np.asarray(features, dtype=np.int64)
    hidden_raw = (vector @ parameters.weight_input) >> parameters.shift
    activation_scale = 1 << 12
    hidden = (activation_scale * hidden_raw) // (activation_scale + np.abs(hidden_raw))
    scores_raw = (hidden @ parameters.weight_output) >> parameters.shift
    scores = tuple(int(score) for score in scores_raw)
    best_index = 0
    for index in range(1, len(scores)):
        if scores[index] > scores[best_index]:
            best_index = index
    competition = [score for index, score in enumerate(scores) if index != best_index]
    second = max(competition) if competition else scores[best_index]
    margin = abs(scores[best_index] - second)
    tie_candidates = [index for index, score in enumerate(scores) if score == scores[best_index]]
    tie_break_used = len(tie_candidates) > 1
    confidence = int((margin * activation_scale) // (activation_scale + abs(scores[best_index])))
    confidence = max(0, min(activation_scale, confidence))
    return InferenceResult(
        attribution=f"spk_{best_index + 1}",
        scores=scores,
        confidence_q12=confidence,
        margin=margin,
        tie_break_used=tie_break_used,
    )


def parameter_summary_digest(parameters: ParameterSet) -> str:
    return canonical_sha256(parameters.snapshot)

