from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .errors import DiagnosticError, ErrorCode


@dataclass(frozen=True)
class ChunkBoundary:
    chunk_index: int
    dimension_start: int
    dimension_end: int
    dimension_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "dimension_start": self.dimension_start,
            "dimension_end": self.dimension_end,
            "dimension_indices": list(self.dimension_indices),
        }


@dataclass(frozen=True)
class ScoreResult:
    scores: np.ndarray
    weights: np.ndarray
    dimensions: tuple[int, ...]
    chunks: tuple[ChunkBoundary, ...]
    max_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": [float(value) for value in self.scores.tolist()],
            "weights": [float(value) for value in self.weights.tolist()],
            "dimensions": list(self.dimensions),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "max_score": self.max_score,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ScoreResult":
        chunks = tuple(
            ChunkBoundary(
                chunk_index=int(item["chunk_index"]),
                dimension_start=int(item["dimension_start"]),
                dimension_end=int(item["dimension_end"]),
                dimension_indices=tuple(int(value) for value in item["dimension_indices"]),
            )
            for item in payload["chunks"]
        )
        return cls(
            scores=np.asarray(payload["scores"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            dimensions=tuple(int(value) for value in payload["dimensions"]),
            chunks=chunks,
            max_score=float(payload["max_score"]),
        )


def build_chunks(dimensions: Sequence[int], chunk_size: int) -> tuple[ChunkBoundary, ...]:
    if chunk_size <= 0:
        raise DiagnosticError(
            ErrorCode.CONFIG_INVALID, "分块大小必须为正整数", stage="scoring"
        )
    dimension_list = [int(value) for value in dimensions]
    chunks: list[ChunkBoundary] = []
    for chunk_index, start in enumerate(range(0, len(dimension_list), chunk_size)):
        indices = tuple(dimension_list[start : start + chunk_size])
        chunks.append(
            ChunkBoundary(
                chunk_index=chunk_index,
                dimension_start=start,
                dimension_end=start + len(indices),
                dimension_indices=indices,
            )
        )
    return tuple(chunks)


def chunked_dot_scores(
    query_rotated: np.ndarray,
    key_rotated: np.ndarray,
    dimensions: Sequence[int],
    chunk_size: int,
) -> tuple[np.ndarray, tuple[ChunkBoundary, ...]]:
    query = np.asarray(query_rotated, dtype=np.float64)
    keys = np.asarray(key_rotated, dtype=np.float64)
    if query.ndim != 1 or keys.ndim != 2 or keys.shape[0] == 0:
        raise DiagnosticError(
            ErrorCode.INTERNAL_ERROR,
            "注意力特征形状非法",
            stage="scoring",
            details={"query_shape": list(query.shape), "key_shape": list(keys.shape)},
        )
    if query.shape[0] != keys.shape[1]:
        raise DiagnosticError(
            ErrorCode.INTERNAL_ERROR,
            "查询与键特征维度不一致",
            stage="scoring",
            details={"query_dimension": query.shape[0], "key_dimension": keys.shape[1]},
        )
    if not dimensions:
        raise DiagnosticError(
            ErrorCode.CONFIG_INVALID, "参与打分的维度集合不得为空", stage="scoring"
        )
    if any(index < 0 or index >= query.shape[0] for index in dimensions):
        raise DiagnosticError(
            ErrorCode.CONFIG_INVALID,
            "参与打分的维度越界",
            stage="scoring",
            details={"dimension": query.shape[0], "indices": list(dimensions)},
        )

    chunks = build_chunks(dimensions, chunk_size)
    scores = np.zeros(keys.shape[0], dtype=np.float64)
    for chunk in chunks:
        indices = np.asarray(chunk.dimension_indices, dtype=np.int64)
        scores += keys[:, indices] @ query[indices]
    if not np.all(np.isfinite(scores)):
        raise DiagnosticError(
            ErrorCode.INTERNAL_ERROR,
            "注意力点积产生非有限数",
            stage="scoring",
        )
    return scores.astype(np.float64, copy=False), chunks


def stable_attention(scores: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise DiagnosticError(
            ErrorCode.INTERNAL_ERROR, "注意力打分数组不得为空", stage="scoring"
        )
    if not np.all(np.isfinite(values)):
        raise DiagnosticError(
            ErrorCode.INTERNAL_ERROR, "稳定归一化前存在非有限打分", stage="scoring"
        )
    maximum = float(np.max(values))
    shifted = values - maximum
    exponentials = np.exp(shifted).astype(np.float64)
    denominator = float(np.sum(exponentials, dtype=np.float64))
    if not math_is_finite_positive(denominator):
        raise DiagnosticError(
            ErrorCode.INTERNAL_ERROR,
            "稳定归一化分母无效",
            stage="scoring",
            details={"denominator": denominator},
        )
    weights = (exponentials / denominator).astype(np.float64)
    return weights, maximum


def math_is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


def score_attention(
    query_rotated: np.ndarray,
    key_rotated: np.ndarray,
    dimensions: Sequence[int],
    chunk_size: int,
) -> ScoreResult:
    scores, chunks = chunked_dot_scores(query_rotated, key_rotated, dimensions, chunk_size)
    weights, maximum = stable_attention(scores)
    return ScoreResult(
        scores=scores,
        weights=weights,
        dimensions=tuple(int(value) for value in dimensions),
        chunks=chunks,
        max_score=maximum,
    )

