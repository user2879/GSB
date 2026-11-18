"""Fixed-point speaker embedding projection plus float and higher-width references."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .aggregation import Aggregation
from .config import PipelineConfig
from .fixedpoint import (
    FixedPoint,
    calibration_scales,
    deterministic_rng,
    integer_l2_normalize,
    quantize_symmetric,
)
from .util import dround_array, fingerprint, sha256_text


@dataclass
class EmbeddingResult:
    sample_id: str
    width: int
    levels: int
    embedding_int: np.ndarray      # Qn integer embedding
    embedding_float: np.ndarray    # dequantized deterministic-rounded float view
    feature_codes: np.ndarray      # post-quantization feature codes
    feature_q_ints: np.ndarray    # Qn feature integers
    raw_accumulator: np.ndarray    # integer matmul accumulator (before rescale)
    calibration_scales: np.ndarray # per-dimension deterministic calibration
    pre_norm_int: np.ndarray       # projected embedding before integer L2 normalize
    integer_norm: int
    projection_seed: int
    embedding_fp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "width": self.width,
            "levels": self.levels,
            "embedding_int": self.embedding_int.tolist(),
            "embedding_float": self.embedding_float.tolist(),
            "feature_codes": self.feature_codes.tolist(),
            "feature_q_ints": self.feature_q_ints.tolist(),
            "raw_accumulator": self.raw_accumulator.tolist(),
            "calibration_scales": self.calibration_scales.tolist(),
            "pre_norm_int": self.pre_norm_int.tolist(),
            "integer_norm": int(self.integer_norm),
            "projection_seed": self.projection_seed,
            "embedding_fp": self.embedding_fp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmbeddingResult":
        return cls(
            sample_id=payload["sample_id"],
            width=int(payload["width"]),
            levels=int(payload["levels"]),
            embedding_int=np.array(payload["embedding_int"], dtype=np.int64),
            embedding_float=np.array(payload["embedding_float"], dtype=np.float64),
            feature_codes=np.array(payload["feature_codes"], dtype=np.int64),
            feature_q_ints=np.array(payload["feature_q_ints"], dtype=np.int64),
            raw_accumulator=np.array(payload["raw_accumulator"], dtype=np.int64),
            calibration_scales=np.array(payload["calibration_scales"], dtype=np.float64),
            pre_norm_int=np.array(payload["pre_norm_int"], dtype=np.int64),
            integer_norm=int(payload["integer_norm"]),
            projection_seed=int(payload["projection_seed"]),
            embedding_fp=payload["embedding_fp"],
        )


def projection_seed(config: PipelineConfig, feature_dim: int) -> int:
    from .util import stable_seed

    return stable_seed(
        f"projection|{config.scope_hash('embedding')}|{config.scope_hash('quant')}"
        f"|dim={feature_dim}|out={config.embedding_dim}"
    )


def build_projection_matrix(config: PipelineConfig, feature_dim: int) -> np.ndarray:
    """Deterministic matrix derived from config fingerprints; never uses runtime randomness."""
    seed = projection_seed(config, feature_dim)
    rng = deterministic_rng(f"projection-matrix|{seed}")
    matrix = rng.standard_normal(size=(config.embedding_dim, feature_dim))
    # Stable normalization keeps integer accumulation bounded.
    norms = np.sqrt(np.sum(matrix * matrix, axis=1, dtype=np.float64))
    return matrix / norms[:, None]


def fixed_embedding(agg: Aggregation, config: PipelineConfig, width: int,
                    projection_matrix: np.ndarray | None = None) -> EmbeddingResult:
    fp = FixedPoint(width)
    stats = agg.stats_vector
    scales = calibration_scales(agg.feature_names, config.scope_hash("quant"))
    normalized = np.clip(stats / scales, -1.0, 1.0).astype(np.float64)
    quant = quantize_symmetric(normalized, config.quant_levels, fp)
    matrix = (
        projection_matrix
        if projection_matrix is not None
        else build_projection_matrix(config, stats.shape[0])
    )
    matrix_q = fp.quantize(matrix)
    raw = matrix_q @ quant["q_ints"]  # int64 integer matmul, deterministic under BLAS for ints
    projected_qn = dround_array(raw.astype(np.float64) / fp.scale)
    normalized_emb = integer_l2_normalize(projected_qn, fp)
    embedding_qn = normalized_emb["embedding_int"]
    embedding_float = normalized_emb["embedding_float"]
    embedding_fp = sha256_text(
        fingerprint(
            {
                "sample_id": agg.sample_id,
                "aggregation_fp": agg.aggregation_fp,
                "width": width,
                "levels": config.quant_levels,
                "embedding_int": embedding_qn,
                "pre_norm_int": projected_qn,
                "projection_seed": projection_seed(config, stats.shape[0]),
            }
        )
    )
    return EmbeddingResult(
        sample_id=agg.sample_id,
        width=width,
        levels=config.quant_levels,
        embedding_int=embedding_qn,
        embedding_float=embedding_float,
        feature_codes=quant["codes"],
        feature_q_ints=quant["q_ints"],
        raw_accumulator=raw,
        calibration_scales=scales,
        pre_norm_int=projected_qn,
        integer_norm=normalized_emb["norm"],
        projection_seed=projection_seed(config, stats.shape[0]),
        embedding_fp=embedding_fp,
    )


def float_reference_embedding(agg: Aggregation, config: PipelineConfig,
                              projection_matrix: np.ndarray) -> dict[str, Any]:
    """Pure float64 path: no quantization, no fixed point; result must differ from fixed path."""
    stats = agg.stats_vector
    scales = calibration_scales(agg.feature_names, config.scope_hash("quant"))
    normalized = np.clip(stats / scales, -1.0, 1.0).astype(np.float64)
    projected = projection_matrix @ normalized
    norm = float(np.sqrt(np.dot(projected, projected)))
    embedding = projected / norm if norm > 0 else projected
    ref_fp = sha256_text(
        fingerprint(
            {
                "sample_id": agg.sample_id,
                "aggregation_fp": agg.aggregation_fp,
                "path": "float_reference",
                "embedding": embedding,
            }
        )
    )
    return {
        "sample_id": agg.sample_id,
        "embedding_float": embedding.astype(np.float64),
        "embedding_fp": ref_fp,
    }
