"""Dot-product attention scoring and train/extrapolation stratification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .encoding import SampleEncoding
from .errors import E_NUMERIC_OVERFLOW, WorkbenchError
from .samples import Sample
from .util import clean_float


@dataclass
class AttentionResult:
    scheme: str
    sample_id: str
    query_position: int
    key_positions: np.ndarray
    relative_distances: np.ndarray
    layers: tuple[str, ...]
    boundary_flags: tuple[bool, ...]
    scores: np.ndarray
    weights: np.ndarray
    reference_scores: np.ndarray
    reference_weights: np.ndarray
    reference_encoding: SampleEncoding

    def to_payload(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "sample_id": self.sample_id,
            "query_position": self.query_position,
            "key_positions": self.key_positions.tolist(),
            "relative_distances": self.relative_distances.tolist(),
            "layers": list(self.layers),
            "boundary_flags": list(self.boundary_flags),
            "scores": self.scores.tolist(),
            "weights": self.weights.tolist(),
            "reference_scores": self.reference_scores.tolist(),
            "reference_weights": self.reference_weights.tolist(),
            "reference_encoding": self.reference_encoding.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AttentionResult":
        return cls(
            scheme=str(payload["scheme"]),
            sample_id=str(payload["sample_id"]),
            query_position=int(payload["query_position"]),
            key_positions=np.asarray(payload["key_positions"], dtype=np.int64),
            relative_distances=np.asarray(payload["relative_distances"], dtype=np.int64),
            layers=tuple(str(item) for item in payload["layers"]),
            boundary_flags=tuple(bool(item) for item in payload["boundary_flags"]),
            scores=np.asarray(payload["scores"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            reference_scores=np.asarray(payload["reference_scores"], dtype=np.float64),
            reference_weights=np.asarray(payload["reference_weights"], dtype=np.float64),
            reference_encoding=SampleEncoding.from_payload(payload["reference_encoding"]),
        )


def dot_scores(query_encoded: np.ndarray, key_encoded: np.ndarray, dimension: int) -> np.ndarray:
    query = np.asarray(query_encoded, dtype=np.float64)
    keys = np.asarray(key_encoded, dtype=np.float64)
    if query.shape != (dimension,) or keys.ndim != 2 or keys.shape[1] != dimension:
        raise WorkbenchError(
            E_NUMERIC_OVERFLOW,
            "encoded feature shape does not match the configured dimension",
            stage="scoring",
            details={"query_shape": list(query.shape), "key_shape": list(keys.shape)},
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scores = np.einsum("i,ki->k", query, keys, optimize=False) / np.sqrt(float(dimension))
    if not np.all(np.isfinite(scores)):
        raise WorkbenchError(
            E_NUMERIC_OVERFLOW,
            "dot-product attention score became non-finite",
            stage="scoring",
        )
    return scores


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise WorkbenchError(
            E_NUMERIC_OVERFLOW,
            "softmax requires a non-empty one-dimensional score vector",
            stage="scoring",
        )
    maximum = float(np.max(values))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exponentials = np.exp(values - maximum)
        total = float(np.sum(exponentials))
        weights = exponentials / total
    if not np.all(np.isfinite(weights)) or total <= 0.0:
        raise WorkbenchError(
            E_NUMERIC_OVERFLOW,
            "normalized attention weights became non-finite",
            stage="scoring",
        )
    return weights


def relative_distances(query_position: int, key_positions: Iterable[int]) -> np.ndarray:
    return np.abs(np.asarray(list(key_positions), dtype=np.int64) - int(query_position))


def classify_distances(
    distances: np.ndarray, train_context_length: int
) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    layers: list[str] = []
    boundary: list[bool] = []
    for distance in np.asarray(distances, dtype=np.int64).tolist():
        is_boundary = int(distance) == int(train_context_length)
        layers.append("in_train" if int(distance) <= int(train_context_length) else "out_of_train")
        boundary.append(is_boundary)
    return tuple(layers), tuple(boundary)


def compute_attention(
    sample: Sample,
    scheme: str,
    encoding: SampleEncoding,
    reference: SampleEncoding,
    dimension: int,
    train_context_length: int,
) -> AttentionResult:
    scores = dot_scores(encoding.query_encoded, encoding.key_encoded, dimension)
    weights = stable_softmax(scores)
    reference_scores = dot_scores(reference.query_encoded, reference.key_encoded, dimension)
    reference_weights = stable_softmax(reference_scores)
    key_positions = np.asarray(sample.key_positions, dtype=np.int64)
    distances = relative_distances(sample.query_position, key_positions)
    layers, boundary_flags = classify_distances(distances, train_context_length)
    return AttentionResult(
        scheme=scheme,
        sample_id=sample.sample_id,
        query_position=sample.query_position,
        key_positions=key_positions,
        relative_distances=distances,
        layers=layers,
        boundary_flags=boundary_flags,
        scores=scores,
        weights=weights,
        reference_scores=reference_scores,
        reference_weights=reference_weights,
        reference_encoding=reference,
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    quantile_levels = {
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
    }
    return {
        name: clean_float(np.quantile(values, level, method="linear"))
        for name, level in quantile_levels.items()
    }


def _stats_row(values: list[float], sample_ids: list[str]) -> dict[str, object]:
    if not values:
        return {
            "sample_count": 0,
            "score_count": 0,
            "mean": 0.0,
            "variance": 0.0,
            "min": None,
            "max": None,
            "quantiles": {
                "p25": None,
                "p50": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "p99": None,
            },
            "sample_ids": [],
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(set(sample_ids)),
        "score_count": int(array.size),
        "mean": clean_float(np.mean(array)),
        "variance": clean_float(np.var(array, ddof=0)),
        "min": clean_float(np.min(array)),
        "max": clean_float(np.max(array)),
        "quantiles": _quantiles(array),
        "sample_ids": sorted(set(sample_ids)),
    }


def aggregate_stratified_statistics(
    results: Iterable[AttentionResult], train_context_length: int
) -> dict[str, object]:
    groups: dict[tuple[str, int], dict[str, list[object]]] = {}
    layer_groups: dict[str, dict[str, list[object]]] = {
        "in_train": {"values": [], "sample_ids": []},
        "out_of_train": {"values": [], "sample_ids": []},
    }
    boundary_samples: list[dict[str, object]] = []
    seen_boundary: set[tuple[str, int]] = set()

    for result in results:
        for index, score in enumerate(result.scores.tolist()):
            distance = int(result.relative_distances[index])
            layer = result.layers[index]
            sample_id = result.sample_id
            group = groups.setdefault((layer, distance), {"values": [], "sample_ids": []})
            group["values"].append(float(score))
            group["sample_ids"].append(sample_id)
            layer_groups[layer]["values"].append(float(score))
            layer_groups[layer]["sample_ids"].append(sample_id)
            if result.boundary_flags[index]:
                identity = (sample_id, index)
                if identity not in seen_boundary:
                    seen_boundary.add(identity)
                    boundary_samples.append(
                        {
                            "scheme": result.scheme,
                            "sample_id": sample_id,
                            "query_position": int(result.query_position),
                            "key_index": index,
                            "key_position": int(result.key_positions[index]),
                            "relative_distance": distance,
                            "layer": "in_train",
                            "boundary_marked": True,
                        }
                    )

    rows: list[dict[str, object]] = []
    for (layer, distance), group in sorted(
        groups.items(), key=lambda item: (0 if item[0][0] == "in_train" else 1, item[0][1])
    ):
        row: dict[str, object] = {
            "layer": layer,
            "relative_distance": distance,
            "boundary_marked": bool(distance == int(train_context_length)),
        }
        row.update(_stats_row(group["values"], group["sample_ids"]))  # type: ignore[arg-type]
        rows.append(row)

    totals: list[dict[str, object]] = []
    for layer in ("in_train", "out_of_train"):
        group = layer_groups[layer]
        row = {
            "layer": layer,
            "relative_distance": None,
            "boundary_marked": False,
            "distance_count": sum(1 for key in groups if key[0] == layer),
        }
        row.update(_stats_row(group["values"], group["sample_ids"]))  # type: ignore[arg-type]
        totals.append(row)

    boundary_samples.sort(
        key=lambda item: (str(item["scheme"]), str(item["sample_id"]), int(item["key_index"]))
    )
    return {
        "boundary_rule": "relative_distance <= train_context_length is in_train; equality is boundary_marked",
        "by_distance": rows,
        "layer_totals": totals,
        "boundary_samples": boundary_samples,
    }
