from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import InferenceConfig, sha256_file
from .errors import (
    E_LABEL_UNKNOWN,
    E_PARAM_SHAPE_MISMATCH,
    WorkbenchError,
)
from .fixedpoint import fixed_scale, quantization_step, quantize_number
from .types import canonical_tag, required_tags


@dataclass(frozen=True)
class ParameterPackage:
    model_version: str
    labels: tuple[str, ...]
    feature_dim: int
    hash_buckets: int
    hash_salt: str
    char_embeddings: dict[str, tuple[int, ...]]
    position_embeddings: dict[int, tuple[int, ...]]
    vocab_embeddings: dict[int, tuple[int, ...]]
    vocab: frozenset[str]
    emission_weights: tuple[tuple[tuple[int, ...], ...], ...]
    transition_weights: tuple[tuple[int, ...], ...]
    start_weights: tuple[int, ...]
    fixed_decimals: int
    quantization_level: int
    raw_hash: str

    @property
    def label_count(self) -> int:
        return len(self.labels)

    @property
    def scale(self) -> int:
        return fixed_scale(self.fixed_decimals)

    @property
    def quant_step(self) -> int:
        return quantization_step(self.quantization_level)

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    def summary(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "model_hash": self.raw_hash,
            "labels": list(self.labels),
            "label_count": self.label_count,
            "feature_dim": self.feature_dim,
            "hash_buckets": self.hash_buckets,
            "fixed_decimals": self.fixed_decimals,
            "quantization_level": self.quantization_level,
            "quantization_step": self.quant_step,
            "fixed_scale": self.scale,
            "start_weight_shape": [self.label_count],
            "transition_weight_shape": [self.label_count, self.label_count],
            "emission_weight_shape": [self.label_count, self.hash_buckets, self.feature_dim],
            "char_embedding_count": len(self.char_embeddings),
            "vocab_size": len(self.vocab),
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", f"cannot read parameter package: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", f"invalid parameter JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", "parameter package root must be an object")
    return value


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", f"{key} must be a positive integer")
    return value


def _matrix(value: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", f"{name} is not a numeric matrix") from exc
    if array.size and not np.isfinite(array).all():
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", f"{name} contains non-finite values")
    return array


def _quantize_vector(values: np.ndarray, cfg: InferenceConfig, name: str) -> tuple[int, ...]:
    if values.ndim != 1:
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", f"{name} must be a vector")
    try:
        return tuple(
            quantize_number(float(value), cfg.fixed_decimals, cfg.quantization_level)
            for value in values.tolist()
        )
    except (TypeError, ValueError) as exc:
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", f"{name} cannot be quantized: {exc}") from exc


def load_parameter_package(path: Path, cfg: InferenceConfig) -> ParameterPackage:
    payload = _load_json(path)
    raw_hash = sha256_file(path)
    model_version = payload.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", "model_version must be a non-empty string")
    model_version = model_version.strip()

    raw_labels = payload.get("labels")
    if not isinstance(raw_labels, list) or not raw_labels:
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", "labels must be a non-empty list")
    labels: list[str] = []
    for raw_label in raw_labels:
        try:
            labels.append(canonical_tag(str(raw_label)))
        except ValueError as exc:
            raise WorkbenchError(E_LABEL_UNKNOWN, "load", str(exc)) from exc
    if len(set(labels)) != len(labels):
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", "parameter labels must be unique")
    missing_labels = sorted(set(required_tags()) - set(labels))
    if missing_labels:
        raise WorkbenchError(
            E_PARAM_SHAPE_MISMATCH,
            "load",
            "parameter labels do not cover required BIESO/type enum: " + ", ".join(missing_labels),
        )

    feature_dim = _positive_int(payload, "feature_dim")
    hash_buckets = _positive_int(payload, "hash_buckets")
    label_count = len(labels)
    scale = fixed_scale(cfg.fixed_decimals)
    step = quantization_step(cfg.quantization_level)

    char_payload = payload.get("char_embeddings")
    if not isinstance(char_payload, dict) or not char_payload:
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", "char_embeddings must be a non-empty object")
    char_embeddings: dict[str, tuple[int, ...]] = {}
    for char, raw_vector in char_payload.items():
        vector = _matrix(raw_vector, f"char_embeddings[{char!r}]")
        if vector.shape != (feature_dim,):
            raise WorkbenchError(
                E_PARAM_SHAPE_MISMATCH,
                "load",
                f"char embedding for {char!r} has shape {vector.shape}, expected ({feature_dim},)",
            )
        char_embeddings[str(char)] = _quantize_vector(vector, cfg, f"char_embeddings[{char!r}]")

    position_payload = payload.get("position_embeddings")
    if not isinstance(position_payload, dict):
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", "position_embeddings must be an object")
    position_embeddings: dict[int, tuple[int, ...]] = {}
    for raw_key, raw_vector in position_payload.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", f"bad position embedding key: {raw_key!r}") from exc
        vector = _matrix(raw_vector, f"position_embeddings[{key}]")
        if vector.shape != (feature_dim,):
            raise WorkbenchError(
                E_PARAM_SHAPE_MISMATCH,
                "load",
                f"position embedding {key} has shape {vector.shape}, expected ({feature_dim},)",
            )
        position_embeddings[key] = _quantize_vector(vector, cfg, f"position_embeddings[{key}]")
    if set(position_embeddings) != set(range(4)):
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", "position_embeddings must cover buckets 0..3")

    vocab_payload = payload.get("vocab", [])
    if not isinstance(vocab_payload, list) or any(not isinstance(item, str) for item in vocab_payload):
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", "vocab must be a list of strings")
    vocab = frozenset(vocab_payload)
    max_vocab_length = max((len(item) for item in vocab), default=1)

    vocab_payload_weights = payload.get("vocab_embeddings")
    if not isinstance(vocab_payload_weights, dict):
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", "vocab_embeddings must be an object")
    vocab_embeddings: dict[int, tuple[int, ...]] = {}
    for raw_key, raw_vector in vocab_payload_weights.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", f"bad vocab embedding key: {raw_key!r}") from exc
        vector = _matrix(raw_vector, f"vocab_embeddings[{key}]")
        if vector.shape != (feature_dim,):
            raise WorkbenchError(
                E_PARAM_SHAPE_MISMATCH,
                "load",
                f"vocab embedding {key} has shape {vector.shape}, expected ({feature_dim},)",
            )
        vocab_embeddings[key] = _quantize_vector(vector, cfg, f"vocab_embeddings[{key}]")
    missing_vocab_lengths = sorted(set(range(max_vocab_length + 1)) - set(vocab_embeddings))
    if missing_vocab_lengths:
        raise WorkbenchError(
            E_PARAM_SHAPE_MISMATCH,
            "load",
            "vocab_embeddings missing hit lengths: " + ", ".join(str(item) for item in missing_vocab_lengths),
        )

    emission_array = _matrix(payload.get("emission_weights"), "emission_weights")
    expected_emission_shape = (label_count, hash_buckets, feature_dim)
    if emission_array.shape != expected_emission_shape:
        raise WorkbenchError(
            E_PARAM_SHAPE_MISMATCH,
            "load",
            f"emission_weights shape {emission_array.shape}, expected {expected_emission_shape}",
        )
    emission_weights: list[tuple[tuple[int, ...], ...]] = []
    for label_index in range(label_count):
        bucket_rows: list[tuple[int, ...]] = []
        for bucket_index in range(hash_buckets):
            vector = emission_array[label_index, bucket_index]
            bucket_rows.append(_quantize_vector(vector, cfg, f"emission_weights[{label_index}][{bucket_index}]"))
        emission_weights.append(tuple(bucket_rows))

    transition_array = _matrix(payload.get("transition_weights"), "transition_weights")
    expected_transition_shape = (label_count, label_count)
    if transition_array.shape != expected_transition_shape:
        raise WorkbenchError(
            E_PARAM_SHAPE_MISMATCH,
            "load",
            f"transition_weights shape {transition_array.shape}, expected {expected_transition_shape}",
        )
    transition_weights = tuple(
        _quantize_vector(transition_array[row], cfg, f"transition_weights[{row}]")
        for row in range(label_count)
    )

    start_payload = payload.get("start_weights")
    if start_payload is None:
        start_array = np.zeros((label_count,), dtype=np.float64)
    else:
        start_array = _matrix(start_payload, "start_weights")
    if start_array.shape != (label_count,):
        raise WorkbenchError(
            E_PARAM_SHAPE_MISMATCH,
            "load",
            f"start_weights shape {start_array.shape}, expected ({label_count},)",
        )
    start_weights = _quantize_vector(start_array, cfg, "start_weights")

    for name, values in (
        ("char_embeddings", [value for vector in char_embeddings.values() for value in vector]),
        ("position_embeddings", [value for vector in position_embeddings.values() for value in vector]),
        ("vocab_embeddings", [value for vector in vocab_embeddings.values() for value in vector]),
        ("emission_weights", [value for label in emission_weights for bucket in label for value in bucket]),
        ("transition_weights", [value for row in transition_weights for value in row]),
        ("start_weights", list(start_weights)),
    ):
        if step > 1 and any(int(value) % step != 0 for value in values):
            raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "load", f"{name} violates quantization step")

    hash_salt = str(payload.get("hash_salt", f"{model_version}:fixed-salt"))
    return ParameterPackage(
        model_version=model_version,
        labels=tuple(labels),
        feature_dim=feature_dim,
        hash_buckets=hash_buckets,
        hash_salt=hash_salt,
        char_embeddings=char_embeddings,
        position_embeddings=position_embeddings,
        vocab_embeddings=vocab_embeddings,
        vocab=vocab,
        emission_weights=tuple(emission_weights),
        transition_weights=transition_weights,
        start_weights=start_weights,
        fixed_decimals=cfg.fixed_decimals,
        quantization_level=cfg.quantization_level,
        raw_hash=raw_hash,
    )


def validate_runtime_bucket_shape(params: ParameterPackage) -> None:
    if len(params.emission_weights) != len(params.labels):
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "emission", "emission label dimension mismatch")
    for label_index, buckets in enumerate(params.emission_weights):
        if len(buckets) != params.hash_buckets:
            raise WorkbenchError(
                E_PARAM_SHAPE_MISMATCH,
                "emission",
                f"emission bucket dimension mismatch at label {label_index}",
            )
        for bucket_index, vector in enumerate(buckets):
            if len(vector) != params.feature_dim:
                raise WorkbenchError(
                    E_PARAM_SHAPE_MISMATCH,
                    "emission",
                    f"emission feature dimension mismatch at label {label_index}, bucket {bucket_index}",
                )


def replace_hash_buckets(params: ParameterPackage, hash_buckets: int) -> ParameterPackage:
    from dataclasses import replace

    return replace(params, hash_buckets=int(hash_buckets))


def all_quantized_values(params: ParameterPackage) -> Iterable[int]:
    for vector in params.char_embeddings.values():
        yield from vector
    for vector in params.position_embeddings.values():
        yield from vector
    for vector in params.vocab_embeddings.values():
        yield from vector
    for label in params.emission_weights:
        for bucket in label:
            yield from bucket
    for row in params.transition_weights:
        yield from row
    yield from params.start_weights
