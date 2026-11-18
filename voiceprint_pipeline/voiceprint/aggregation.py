"""Speaker-level (per-utterance) deterministic statistic aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import PipelineConfig
from .errors import InsufficientFramesError
from .features import FrameFeatures
from .util import fingerprint, sha256_text


@dataclass
class Aggregation:
    sample_id: str
    valid_frame_count: int
    total_frame_count: int
    valid_frame_indices: list[int]
    stats_vector: np.ndarray
    feature_names: list[str]
    partial_summary: dict[str, Any]
    aggregation_fp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "valid_frame_count": self.valid_frame_count,
            "total_frame_count": self.total_frame_count,
            "valid_frame_indices": self.valid_frame_indices,
            "stats_vector": self.stats_vector.tolist(),
            "feature_names": self.feature_names,
            "partial_summary": self.partial_summary,
            "aggregation_fp": self.aggregation_fp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Aggregation":
        return cls(
            sample_id=payload["sample_id"],
            valid_frame_count=int(payload["valid_frame_count"]),
            total_frame_count=int(payload["total_frame_count"]),
            valid_frame_indices=[int(x) for x in payload["valid_frame_indices"]],
            stats_vector=np.array(payload["stats_vector"], dtype=np.float64),
            feature_names=list(payload["feature_names"]),
            partial_summary=dict(payload["partial_summary"]),
            aggregation_fp=payload["aggregation_fp"],
        )


def aggregate(frames: FrameFeatures, config: PipelineConfig) -> Aggregation:
    total = frames.n_frames
    gate = config.silence_energy_db
    valid_indices = [i for i in range(total) if float(frames.energy_db[i]) >= gate]
    valid_count = len(valid_indices)

    partial = {
        "total_frames": total,
        "frames_above_gate": valid_count,
        "min_valid_frames": config.min_valid_frames,
        "mean_energy_db": float(np.mean(frames.energy_db, dtype=np.float64)) if total else 0.0,
        "gate_energy_db": gate,
    }
    if valid_count < config.min_valid_frames:
        raise InsufficientFramesError(
            f"valid frames {valid_count} < minimum {config.min_valid_frames}",
            **partial,
        )

    index = np.array(valid_indices, dtype=np.int64)
    f0 = frames.f0[index]
    cep = frames.cepstrum[index]
    energy = frames.energy_db[index]

    blocks: list[np.ndarray] = []
    names: list[str] = []
    blocks.append(_stats_block("f0", f0.reshape(-1, 1), config, names))
    blocks.append(_stats_block("cep", cep, config, names))
    blocks.append(_stats_block("energy_db", energy.reshape(-1, 1), config, names))
    stats_vector = np.concatenate(blocks, axis=0).astype(np.float64)

    aggregation_fp = sha256_text(
        fingerprint(
            {
                "sample_id": frames.sample_id,
                "feature_fp": frames.feature_fp,
                "valid_frame_indices": valid_indices,
                "stats_vector": stats_vector,
            }
        )
    )
    partial["valid_frame_indices"] = valid_indices
    return Aggregation(
        sample_id=frames.sample_id,
        valid_frame_count=valid_count,
        total_frame_count=total,
        valid_frame_indices=valid_indices,
        stats_vector=stats_vector,
        feature_names=names,
        partial_summary=partial,
        aggregation_fp=aggregation_fp,
    )


def _stats_block(prefix: str, matrix: np.ndarray, config: PipelineConfig,
                 names: list[str]) -> np.ndarray:
    """For each column emit mean, population variance, and configured quantiles, in order."""
    columns = [matrix[:, j] for j in range(matrix.shape[1])]
    rows: list[np.ndarray] = []
    method = config.quantile_interpolation
    points = config.quantile_points
    for col_idx, column in enumerate(columns):
        base = prefix if matrix.shape[1] == 1 else f"{prefix}{col_idx}"
        rows.append(np.array([np.mean(column, dtype=np.float64)], dtype=np.float64))
        names.append(f"{base}.mean")
        rows.append(np.array([np.var(column, dtype=np.float64)], dtype=np.float64))
        names.append(f"{base}.var")
        for q in points:
            rows.append(np.array([np.quantile(column, q, method=method)], dtype=np.float64))
            names.append(f"{base}.q{q}")
    return np.concatenate(rows, axis=0)
