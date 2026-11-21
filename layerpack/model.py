"""In-memory input and result data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import Failure


@dataclass
class Layer:
    name: str
    shape: tuple[int, ...]
    weights: np.ndarray
    content_version: int
    content_fingerprint: str
    source_index: int = 0

    @property
    def element_count(self) -> int:
        return int(self.weights.size)

    def to_input_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "weights": self.weights.reshape(-1, order="C").tolist(),
            "content_version": self.content_version,
        }


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    inputs: tuple[float, ...]
    reference_output: tuple[float, ...]
    layers: tuple[str, ...] = ()
    source_index: int = 0

    def applies_to(self, layer_name: str) -> bool:
        return not self.layers or "all" in self.layers or layer_name in self.layers

    def to_input_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.sample_id,
            "input": list(self.inputs),
            "reference_output": list(self.reference_output),
        }
        if self.layers:
            value["layers"] = list(self.layers)
        return value


@dataclass
class CalibrationSet:
    samples: list[CalibrationSample]
    fingerprint: str
    issues: list[Failure] = field(default_factory=list)

    def sample_ids(self, layer_name: str | None = None) -> list[str]:
        values = [
            sample.sample_id
            for sample in self.samples
            if layer_name is None or sample.applies_to(layer_name)
        ]
        return sorted(values)


@dataclass
class LoadedState:
    layers: dict[str, Layer]
    calibration: CalibrationSet
    failures: list[Failure]
    weights_hash: str = ""
    calibration_hash: str = ""


@dataclass
class QuantResult:
    layer_name: str
    parameters: dict[str, Any]
    dequantized: np.ndarray
    codes: np.ndarray
    distortion: dict[str, Any]
    quantized_sha256: str
    dequantized_sha256: str
    key: str
    dequant_key: str = ""
    reused: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "layer": self.layer_name,
            "parameters_sha256": self.parameters.get("parameters_sha256", ""),
            "quantized_sha256": self.quantized_sha256,
            "dequantized_sha256": self.dequantized_sha256,
            "distortion": self.distortion,
        }


@dataclass
class CalibrationResult:
    layer_name: str
    samples: list[dict[str, Any]]
    summary: dict[str, Any]
    key: str
    reused: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer_name,
            "samples": self.samples,
            "summary": self.summary,
        }


@dataclass
class SchedulerEvent:
    sort_key: str
    layer: str
    stage: str
    points: int
    batch: int
    outcome: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sort_key": self.sort_key,
            "layer": self.layer,
            "stage": self.stage,
            "points": self.points,
            "batch": self.batch,
            "outcome": self.outcome,
        }


@dataclass
class PendingTask:
    sort_key: str
    layer: str
    stage: str
    points: int
    reason: str = "point_budget_exhausted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sort_key": self.sort_key,
            "layer": self.layer,
            "stage": self.stage,
            "points": self.points,
            "reason": self.reason,
        }
