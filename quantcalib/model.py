"""Input and configuration data models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .errors import FailureList
from .utils import clean_float, stable_hash


@dataclass
class Layer:
    name: str
    shape: tuple[int, ...]
    weights: tuple[float, ...]
    content_version: int
    group_granularity: int
    source_line: int

    @property
    def numel(self) -> int:
        return len(self.weights)

    def content_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "weights": [
                clean_float(value) if math.isfinite(float(value)) else str(float(value))
                for value in self.weights
            ],
            "content_version": self.content_version,
            "group_granularity": self.group_granularity,
        }

    @property
    def content_fingerprint(self) -> str:
        return stable_hash(self.content_payload())


@dataclass
class CalibrationSample:
    sample_id: str
    input_vector: tuple[float, ...]
    reference_output: tuple[float, ...]
    source_index: int

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "input": [clean_float(value) for value in self.input_vector],
            "reference_output": [clean_float(value) for value in self.reference_output],
        }


@dataclass
class CalibrationSet:
    input_dim: int
    output_dim: int
    samples: tuple[CalibrationSample, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "samples": [sample.payload() for sample in self.samples],
        }

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples)

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.payload())


@dataclass(frozen=True)
class LayerQuantConfig:
    scheme: str
    bit_width: int
    group_granularity: int
    lower_permille: int
    upper_permille: int
    fractional_bits: int
    quantization_levels: int
    scale_effective_digits: int

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "bit_width": self.bit_width,
            "group_granularity": self.group_granularity,
            "lower_permille": self.lower_permille,
            "upper_permille": self.upper_permille,
            "fractional_bits": self.fractional_bits,
            "quantization_levels": self.quantization_levels,
            "scale_effective_digits": self.scale_effective_digits,
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.fingerprint_payload())

    @property
    def qmin(self) -> int:
        if self.scheme == "symmetric":
            return -(2 ** (self.bit_width - 1) - 1)
        return -(2 ** (self.bit_width - 1))

    @property
    def qmax(self) -> int:
        if self.scheme == "symmetric":
            return 2 ** (self.bit_width - 1) - 1
        return 2 ** (self.bit_width - 1) - 1


@dataclass(frozen=True)
class DistillationConfig:
    temperature: float
    steps: int
    soft_label_weight: float
    tolerance: float

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "temperature": clean_float(self.temperature),
            "steps": self.steps,
            "soft_label_weight": clean_float(self.soft_label_weight),
            "tolerance": clean_float(self.tolerance),
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.fingerprint_payload())


@dataclass(frozen=True)
class SchedulingConfig:
    thread_pool_workers: int
    global_calibration_point_budget: int
    per_layer_calibration_quota: int
    batch_quantization_point_budget: int
    snapshot_commit_interval_batches: int


@dataclass
class Config:
    raw: dict[str, Any]
    defaults: dict[str, Any]
    layer_overrides: dict[str, dict[str, Any]]
    distillation: DistillationConfig
    scheduling: SchedulingConfig

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "raw": self.raw,
                "layer_overrides": self.layer_overrides,
                "distillation": self.distillation.fingerprint_payload(),
                "scheduling": {
                    "thread_pool_workers": self.scheduling.thread_pool_workers,
                    "global_calibration_point_budget": self.scheduling.global_calibration_point_budget,
                    "per_layer_calibration_quota": self.scheduling.per_layer_calibration_quota,
                    "batch_quantization_point_budget": self.scheduling.batch_quantization_point_budget,
                    "snapshot_commit_interval_batches": self.scheduling.snapshot_commit_interval_batches,
                },
            }
        )

    def for_layer(self, layer: Layer) -> LayerQuantConfig:
        merged = dict(self.defaults)
        merged.update(self.layer_overrides.get(layer.name, {}))
        group = int(merged.get("group_granularity", layer.group_granularity))
        return LayerQuantConfig(
            scheme=str(merged["scheme"]),
            bit_width=int(merged["bit_width"]),
            group_granularity=group,
            lower_permille=int(merged["percentile_lower_permille"]),
            upper_permille=int(merged["percentile_upper_permille"]),
            fractional_bits=int(merged["fractional_bits"]),
            quantization_levels=int(merged["quantization_levels"]),
            scale_effective_digits=int(merged["scale_effective_digits"]),
        )


@dataclass
class Receipt:
    receipt_id: str
    source_file: str
    source_line: int
    source_version: str
    layer: str | None
    compression_permille: int | None
    parameter_version: int | None
    elapsed_ms: int | None
    group_scale_digest: str | None
    supersedes: tuple[str, ...]
    missing_fields: tuple[str, ...]
    raw: dict[str, Any]

    def normalized_payload(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "source_version": self.source_version,
            "layer": self.layer,
            "compression_permille": self.compression_permille,
            "parameter_version": self.parameter_version,
            "elapsed_ms": self.elapsed_ms,
            "group_scale_digest": self.group_scale_digest,
            "supersedes": list(self.supersedes),
            "missing_fields": list(self.missing_fields),
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.normalized_payload())


@dataclass
class ReceiptCorpus:
    receipts: list[Receipt] = field(default_factory=list)
    supplied_paths: list[str] = field(default_factory=list)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "receipts": [item.normalized_payload() for item in self.receipts],
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.fingerprint_payload())


@dataclass
class Dataset:
    layers: list[Layer]
    calibration: CalibrationSet | None
    config: Config | None
    receipts: ReceiptCorpus
    failures: FailureList
    weights_path: str | None = None
    calibration_path: str | None = None
    config_path: str | None = None

    @property
    def valid_layers(self) -> list[Layer]:
        return sorted(self.layers, key=lambda item: item.name)

    @property
    def weight_fingerprint(self) -> str:
        return stable_hash([layer.content_payload() for layer in self.layers])

    @property
    def calibration_fingerprint(self) -> str:
        return self.calibration.fingerprint if self.calibration else stable_hash(None)
