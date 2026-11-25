"""Internal data models shared by every stage."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WeightLayer:
    name: str
    shape: tuple[int, ...]
    weights: np.ndarray
    content_version: int
    group_granularity: int

    def size(self) -> int:
        return int(self.weights.size)

    def group_count(self) -> int:
        return self.size() // self.group_granularity

    def to_record(self) -> dict[str, Any]:
        flat = self.weights.reshape(-1)
        return {
            "layer": self.name,
            "shape": list(self.shape),
            "weights": [float(x) for x in flat.tolist()],
            "content_version": self.content_version,
            "group_granularity": self.group_granularity,
        }


@dataclass(frozen=True)
class CalibrationSet:
    inputs: np.ndarray          # (n_samples, in_dim) float64
    reference_outputs: np.ndarray  # (n_samples, out_dim) float64
    seed: int

    @property
    def sample_count(self) -> int:
        return int(self.inputs.shape[0])


# Field precision labels record the *source* precision of every receipt field.
PRECISION_INT32 = "int32"
PRECISION_INT64 = "int64"
PRECISION_FLOAT64 = "float64"
PRECISION_FLOAT32 = "float32"
PRECISION_UTF8 = "utf8"


@dataclass
class UnifiedReceipt:
    """Canonical internal representation for q1/q2/q3 receipts."""

    layer: str
    coverage_permille: int
    source_version: str
    missing_fields: list[str] = field(default_factory=list)
    precision_labels: dict[str, str] = field(default_factory=dict)
    param_version: int | None = None
    time_ms: int | None = None
    group_scale_digest: str | None = None
    duplicate_refs: list[str] = field(default_factory=list)
    weight_snapshot: str | None = None
    sidecar: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedReceipt":
        return cls(**data)
