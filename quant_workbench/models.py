"""Immutable-ish domain models shared by loaders, engine and reporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .util import canonical_json, fingerprint


@dataclass(slots=True)
class LayerConfig:
    bits: int
    group_size: int
    fractional_bits: int
    scheme: str = "symmetric"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "fractional_bits": self.fractional_bits,
            "scheme": self.scheme,
        }

    @property
    def parameter_fingerprint(self) -> str:
        return fingerprint(self.to_dict())


DEFAULT_BITS = 8
DEFAULT_GROUP_SIZE = 4
DEFAULT_FRACTIONAL_BITS = 4
DEFAULT_SCHEME = "symmetric"


@dataclass(slots=True)
class WorkbenchConfig:
    snapshot_version: int
    allowed_bit_widths: tuple[int, ...]
    max_threads: int
    thread_limit: int
    latency_budget_ms: float
    point_batch_budget: int
    snapshot_commit_interval: int
    distillation_temperature: float
    significant_digits: int
    defaults: LayerConfig
    layers: dict[str, LayerConfig]
    source_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    file_hash: str | None = None
    worker_delay_ms: float = 0.0

    def layer_config(self, layer: str) -> LayerConfig:
        return self.layers.get(layer, self.defaults)

    def config_fingerprint(self) -> str:
        return fingerprint(
            {
                "snapshot_version": self.snapshot_version,
                "allowed_bit_widths": list(self.allowed_bit_widths),
                "max_threads": self.max_threads,
                "thread_limit": self.thread_limit,
                "latency_budget_ms": self.latency_budget_ms,
                "point_batch_budget": self.point_batch_budget,
                "snapshot_commit_interval": self.snapshot_commit_interval,
                "distillation_temperature": self.distillation_temperature,
                "significant_digits": self.significant_digits,
                "defaults": self.defaults.to_dict(),
                "layers": {name: cfg.to_dict() for name, cfg in sorted(self.layers.items())},
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "allowed_bit_widths": list(self.allowed_bit_widths),
            "max_threads": self.max_threads,
            "thread_limit": self.thread_limit,
            "latency_budget_ms": self.latency_budget_ms,
            "point_batch_budget": self.point_batch_budget,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "distillation_temperature": self.distillation_temperature,
            "significant_digits": self.significant_digits,
            "defaults": self.defaults.to_dict(),
            "layers": {name: cfg.to_dict() for name, cfg in sorted(self.layers.items())},
        }

    def copy_with_layer_config(self, layer: str, cfg: LayerConfig) -> None:
        self.layers[layer] = cfg


@dataclass(slots=True)
class WeightLayer:
    layer: str
    version: int
    shape: tuple[int, ...]
    weights: np.ndarray
    order: int
    line_number: int | None = None
    _content_fingerprint: str | None = None

    @property
    def content_fingerprint(self) -> str:
        if self._content_fingerprint is None:
            self._content_fingerprint = fingerprint(
                {
                    "layer": self.layer,
                    "version": self.version,
                    "shape": list(self.shape),
                    "weights": self.weights.tolist(),
                }
            )
        return self._content_fingerprint

    @property
    def point_count(self) -> int:
        return int(self.weights.size)

    def copy(self) -> "WeightLayer":
        return WeightLayer(
            layer=self.layer,
            version=self.version,
            shape=self.shape,
            weights=self.weights.copy(),
            order=self.order,
            line_number=self.line_number,
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "version": self.version,
            "shape": list(self.shape),
            "order": self.order,
            "line_number": self.line_number,
            "points": self.point_count,
            "content_fingerprint": self.content_fingerprint,
        }


@dataclass(slots=True)
class CalibrationSample:
    sample_id: str
    activations: dict[str, np.ndarray]
    line_number: int | None = None
    source: dict[str, Any] | None = None


@dataclass(slots=True)
class CalibrationSet:
    samples: list[CalibrationSample]
    source_path: str | None = None
    _fingerprint: str | None = None

    @property
    def sample_ids(self) -> list[str]:
        return [sample.sample_id for sample in self.samples]

    @property
    def content_fingerprint(self) -> str:
        if self._fingerprint is None:
            payload = [
                {
                    "sample_id": sample.sample_id,
                    "activations": {
                        name: sample.activations[name].tolist()
                        for name in sorted(sample.activations)
                    },
                }
                for sample in self.samples
            ]
            self._fingerprint = fingerprint({"samples": payload})
        return self._fingerprint

    def copy(self) -> "CalibrationSet":
        return CalibrationSet(
            samples=[
                CalibrationSample(
                    sample_id=sample.sample_id,
                    activations={
                        layer: values.copy() for layer, values in sample.activations.items()
                    },
                    line_number=sample.line_number,
                    source=sample.source,
                )
                for sample in self.samples
            ],
            source_path=self.source_path,
            _fingerprint=self._fingerprint,
        )

    def replace_samples(self, samples: list[CalibrationSample], source_path: str | None = None) -> None:
        self.samples = samples
        self.source_path = source_path
        self._fingerprint = None


@dataclass(slots=True)
class Command:
    command_index: int
    line_number: int
    op: str
    layer: str | None
    payload: dict[str, Any]
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_index": self.command_index,
            "line_number": self.line_number,
            "op": self.op,
            "layer": self.layer,
            "payload": self.payload,
        }


@dataclass(slots=True)
class StageCacheEntry:
    cache_key: str
    layer: str
    value: dict[str, Any]
    stage: str
    config_fields: list[str] = field(default_factory=list)
    calibration_sample_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EngineState:
    weights: dict[str, WeightLayer]
    config: WorkbenchConfig
    calibration: CalibrationSet
    generation: int = 0
    command_count: int = 0
    config_revision: int = 0

    def layer_order(self) -> list[str]:
        return sorted(
            self.weights,
            key=lambda name: (self.weights[name].order, name),
        )

    def canonical_order(self) -> list[str]:
        return sorted(self.weights)
