from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import SimError
from .util import hash_mapping


VALID_SCHEMES = ("symmetric", "asymmetric")
VALID_AGGREGATION_MODES = ("mean", "sum")
VALID_ORDER_MODES = ("auto", "name_asc", "bytes_desc")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[Any, ...]

    def resolved_shape(self, hidden_dim: int) -> tuple[int, ...]:
        resolved: list[int] = []
        for dimension in self.shape:
            if isinstance(dimension, str):
                if dimension not in {"hidden_dim", "hidden", "d_model"}:
                    raise SimError("E_CONFIG_INVALID", f"unknown shape placeholder {dimension!r}")
                resolved.append(hidden_dim)
            elif isinstance(dimension, int) and dimension > 0:
                resolved.append(dimension)
            else:
                raise SimError("E_CONFIG_INVALID", f"invalid tensor dimension {dimension!r}")
        return tuple(resolved)


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    hidden_dim: int
    layer_tensors: tuple[TensorSpec, ...]

    def expanded_tensors(self) -> tuple[dict[str, Any], ...]:
        tensors: list[dict[str, Any]] = []
        for layer in range(self.layers):
            for spec in self.layer_tensors:
                tensors.append(
                    {
                        "name": f"layer_{layer}.{spec.name}",
                        "shape": list(spec.resolved_shape(self.hidden_dim)),
                        "layer": layer,
                        "local_name": spec.name,
                    }
                )
        return tuple(tensors)

    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor["name"] for tensor in self.expanded_tensors())


@dataclass(frozen=True)
class CompressionCandidate:
    scheme: str
    bits: int

    @property
    def label(self) -> str:
        return f"{self.scheme}-{self.bits}"

    @property
    def scheme_rank(self) -> int:
        return VALID_SCHEMES.index(self.scheme)


@dataclass(frozen=True)
class CompressionConfig:
    candidates: tuple[CompressionCandidate, ...]
    window_steps: int
    max_bytes_per_window: int


@dataclass(frozen=True)
class AggregationConfig:
    mode: str
    order: str


@dataclass(frozen=True)
class SimConfig:
    seed: int
    model: ModelConfig
    workers: int
    steps: int
    compression: CompressionConfig
    aggregation: AggregationConfig
    fidelity_generations: int
    concurrency: int
    snapshot_interval_steps: int
    communication_error_tolerance: float
    snapshot_dir: Path
    raw: dict[str, Any] = field(default_factory=dict, compare=False)
    config_hash: str = field(default="", compare=False)

    def tensor_names(self) -> tuple[str, ...]:
        return self.model.tensor_names()

    def window_bounds(self, window_index: int) -> tuple[int, int]:
        start = window_index * self.compression.window_steps
        end = min(start + self.compression.window_steps, self.steps)
        return start, end

    def window_count(self) -> int:
        return math.ceil(self.steps / self.compression.window_steps)

    def hash_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "model": {
                "layers": self.model.layers,
                "hidden_dim": self.model.hidden_dim,
                "layer_tensors": [
                    {"name": spec.name, "shape": list(spec.shape)} for spec in self.model.layer_tensors
                ],
            },
            "workers": self.workers,
            "steps": self.steps,
            "compression": {
                "candidates": [
                    {"scheme": candidate.scheme, "bits": candidate.bits}
                    for candidate in self.compression.candidates
                ],
                "window_steps": self.compression.window_steps,
                "max_bytes_per_window": self.compression.max_bytes_per_window,
            },
            "aggregation": {"mode": self.aggregation.mode, "order": self.aggregation.order},
            "fidelity_generations": self.fidelity_generations,
            "concurrency": self.concurrency,
            "snapshot_interval_steps": self.snapshot_interval_steps,
            "communication_error_tolerance": self.communication_error_tolerance,
        }


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SimError("E_CONFIG_INVALID", f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SimError("E_CONFIG_INVALID", f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise SimError("E_CONFIG_INVALID", f"{label} must be >= {minimum}")
    return value


def _parse_tensor_specs(value: Any) -> tuple[TensorSpec, ...]:
    if not isinstance(value, list) or not value:
        raise SimError("E_CONFIG_INVALID", "model.layer_tensors must be a non-empty list")
    result: list[TensorSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if isinstance(item, str):
            name = item
            shape: tuple[Any, ...] = ("hidden_dim",)
        elif isinstance(item, dict):
            name = item.get("name")
            shape_value = item.get("shape", ["hidden_dim"])
            if isinstance(shape_value, int):
                shape = (shape_value,)
            elif isinstance(shape_value, list) and shape_value:
                shape = tuple(shape_value)
            else:
                raise SimError("E_CONFIG_INVALID", f"invalid shape at model.layer_tensors[{index}]")
        else:
            raise SimError("E_CONFIG_INVALID", f"invalid tensor spec at model.layer_tensors[{index}]")
        if not isinstance(name, str) or not name or any(char in name for char in "/\\"):
            raise SimError("E_CONFIG_INVALID", f"invalid tensor name at model.layer_tensors[{index}]")
        if name in seen:
            raise SimError("E_CONFIG_INVALID", f"duplicate per-layer tensor name {name!r}")
        seen.add(name)
        result.append(TensorSpec(name=name, shape=shape))
    return tuple(result)


def _parse_candidates(compression: dict[str, Any]) -> tuple[CompressionCandidate, ...]:
    candidates: list[CompressionCandidate] = []
    raw_candidates = compression.get("candidates")
    if raw_candidates is None:
        schemes = compression.get("schemes", compression.get("scheme_candidates", ["symmetric"]))
        bits_values = compression.get("bits", compression.get("bit_widths", compression.get("bit_width_candidates")))
        if bits_values is None:
            raise SimError("E_CONFIG_INVALID", "compression candidates or bits are required")
        if not isinstance(schemes, list):
            schemes = [schemes]
        if not isinstance(bits_values, list):
            bits_values = [bits_values]
        for scheme in schemes:
            for bits in bits_values:
                candidates.append(CompressionCandidate(str(scheme), int(bits)))
    elif isinstance(raw_candidates, list):
        for item in raw_candidates:
            if not isinstance(item, dict):
                raise SimError("E_CONFIG_INVALID", "each compression candidate must be a mapping")
            scheme = item.get("scheme", "symmetric")
            bits_value = item.get("bits", item.get("bit_width"))
            if isinstance(bits_value, list):
                for bits in bits_value:
                    candidates.append(CompressionCandidate(str(scheme), int(bits)))
            else:
                if bits_value is None:
                    raise SimError("E_CONFIG_INVALID", "compression candidate is missing bits")
                candidates.append(CompressionCandidate(str(scheme), int(bits_value)))
    else:
        raise SimError("E_CONFIG_INVALID", "compression.candidates must be a list")

    normalized: dict[tuple[str, int], CompressionCandidate] = {}
    for candidate in candidates:
        if candidate.scheme not in VALID_SCHEMES:
            raise SimError("E_CONFIG_INVALID", f"unsupported compression scheme {candidate.scheme!r}")
        if candidate.bits < 2 or candidate.bits > 16:
            raise SimError("E_CONFIG_INVALID", "compression bit width must be in [2, 16]")
        normalized[(candidate.scheme, candidate.bits)] = candidate
    if not normalized:
        raise SimError("E_CONFIG_INVALID", "at least one compression candidate is required")
    return tuple(
        sorted(normalized.values(), key=lambda item: (item.scheme_rank, item.bits))
    )


def load_config(path: str | Path) -> SimConfig:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw_value = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise SimError("E_CONFIG_INVALID", f"config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise SimError("E_CONFIG_INVALID", f"invalid YAML config: {exc}") from exc

    raw = _require_mapping(raw_value, "config")
    model_raw = _require_mapping(raw.get("model"), "model")
    compression_raw = _require_mapping(raw.get("compression"), "compression")
    aggregation_raw = _require_mapping(raw.get("aggregation", {}), "aggregation")

    layers = _positive_int(model_raw.get("layers", model_raw.get("num_layers")), "model.layers")
    hidden_dim = _positive_int(model_raw.get("hidden_dim", model_raw.get("hidden")), "model.hidden_dim")
    tensor_specs = _parse_tensor_specs(
        model_raw.get("layer_tensors", model_raw.get("tensors"))
    )
    model = ModelConfig(layers=layers, hidden_dim=hidden_dim, layer_tensors=tensor_specs)

    workers = _positive_int(
        raw.get("data_parallel_workers", raw.get("workers", raw.get("worker_count"))),
        "data_parallel_workers",
    )
    steps = _positive_int(raw.get("steps", raw.get("training_steps")), "steps")
    seed = raw.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SimError("E_CONFIG_INVALID", "seed must be an integer")

    candidates = _parse_candidates(compression_raw)
    window_steps = _positive_int(
        compression_raw.get("window_steps", compression_raw.get("communication_window", 1)),
        "compression.window_steps",
    )
    max_bytes_per_window = _positive_int(
        compression_raw.get("max_bytes_per_window", raw.get("max_bytes_per_window")),
        "compression.max_bytes_per_window",
    )
    compression = CompressionConfig(
        candidates=candidates,
        window_steps=window_steps,
        max_bytes_per_window=max_bytes_per_window,
    )

    mode = str(aggregation_raw.get("mode", raw.get("aggregation_mode", "mean")))
    order = str(aggregation_raw.get("order", raw.get("aggregation_order", "auto")))
    if mode not in VALID_AGGREGATION_MODES:
        raise SimError("E_CONFIG_INVALID", f"unsupported aggregation mode {mode!r}")
    if order not in VALID_ORDER_MODES:
        raise SimError("E_CONFIG_INVALID", f"unsupported aggregation order {order!r}")
    aggregation = AggregationConfig(mode=mode, order=order)

    fidelity_generations = _positive_int(
        raw.get("fidelity_generations", raw.get("retention_generations", 0)),
        "fidelity_generations",
        allow_zero=True,
    )
    concurrency = _positive_int(raw.get("concurrency", 1), "concurrency")
    snapshot_interval_steps = _positive_int(
        raw.get("snapshot_interval_steps", raw.get("snapshot_commit_interval_steps", 1)),
        "snapshot_interval_steps",
    )

    tolerance_value = raw.get(
        "communication_error_tolerance",
        raw.get("error_tolerance", raw.get("tolerance")),
    )
    if isinstance(tolerance_value, bool) or not isinstance(tolerance_value, (int, float)):
        raise SimError("E_CONFIG_INVALID", "communication_error_tolerance is required")
    tolerance = float(tolerance_value)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise SimError("E_CONFIG_INVALID", "communication_error_tolerance must be finite and positive")

    snapshot_value = raw.get("snapshot_dir", raw.get("snapshot_path", "runtime/snapshots"))
    if not isinstance(snapshot_value, str) or not snapshot_value:
        raise SimError("E_CONFIG_INVALID", "snapshot_dir must be a non-empty path string")

    config = SimConfig(
        seed=seed,
        model=model,
        workers=workers,
        steps=steps,
        compression=compression,
        aggregation=aggregation,
        fidelity_generations=fidelity_generations,
        concurrency=concurrency,
        snapshot_interval_steps=snapshot_interval_steps,
        communication_error_tolerance=tolerance,
        snapshot_dir=Path(snapshot_value),
        raw=raw,
    )
    config_hash = hash_mapping(config.hash_payload())
    return SimConfig(
        seed=config.seed,
        model=config.model,
        workers=config.workers,
        steps=config.steps,
        compression=config.compression,
        aggregation=config.aggregation,
        fidelity_generations=config.fidelity_generations,
        concurrency=config.concurrency,
        snapshot_interval_steps=config.snapshot_interval_steps,
        communication_error_tolerance=config.communication_error_tolerance,
        snapshot_dir=config.snapshot_dir,
        raw=config.raw,
        config_hash=config_hash,
    )

