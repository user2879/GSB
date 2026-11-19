from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import BasePackage
from .canonical import atomic_write_bytes, atomic_write_json, safe_component
from .config import PeftConfig
from .errors import E_MERGE_BUDGET_EXHAUSTED, E_MERGE_NOT_REVERSIBLE, BusinessError
from .tasks import TaskSpec, task_order_key
from .training import adapter_delta


MERGED_FORMAT = "peft-lora-merged-v1"


@dataclass
class MergeBudget:
    weight_limit: int | None
    compute_limit: int | None
    weight_used: int = 0
    compute_used: int = 0
    ledger: list[dict[str, Any]] = field(default_factory=list)
    sequence: int = 0

    def request(
        self,
        task: TaskSpec,
        weight_cost: int,
        compute_cost: int,
        *,
        weight_available: bool = True,
    ) -> bool:
        weight_after = self.weight_used + weight_cost
        compute_after = self.compute_used + compute_cost
        weight_exceeded = not weight_available or (
            self.weight_limit is not None and weight_after > self.weight_limit
        )
        compute_exceeded = (
            self.compute_limit is not None and compute_after > self.compute_limit
        )
        allowed = not weight_exceeded and not compute_exceeded
        self.sequence += 1
        entry = {
            "sequence": self.sequence,
            "task_name": task.task_name,
            "target_layer": task.target_layer,
            "requested_weight_elements": weight_cost,
            "requested_compute_units": compute_cost,
            "weight_before": self.weight_used,
            "weight_after": weight_after,
            "compute_before": self.compute_used,
            "compute_after": compute_after,
            "decision": "ALLOW" if allowed else "DENY",
            "weight_limit": self.weight_limit,
            "compute_limit": self.compute_limit,
        }
        if allowed:
            self.weight_used = weight_after
            self.compute_used = compute_after
        else:
            reasons: list[str] = []
            if weight_exceeded:
                reasons.append("weight_elements")
            if compute_exceeded:
                reasons.append("compute_units")
            entry["deny_reasons"] = reasons
        self.ledger.append(entry)
        return allowed

    def summary(self) -> dict[str, Any]:
        return {
            "weight_limit": self.weight_limit,
            "compute_limit": self.compute_limit,
            "weight_used": self.weight_used,
            "compute_used": self.compute_used,
            "ledger": self.ledger,
        }


@dataclass
class MergeResult:
    merged_weight: dict[str, np.ndarray[Any, Any]]
    merged_bias: dict[str, np.ndarray[Any, Any]]
    merged_tasks: list[dict[str, Any]]
    pending_tasks: list[dict[str, Any]]
    budget: MergeBudget
    restore_errors: dict[str, float]
    restore_max_abs_error: float
    quantization_scale: int
    quantization_step: float
    allowed_restore_error: float

    @property
    def business_errors(self) -> list[dict[str, Any]]:
        return [
            {
                "code": E_MERGE_BUDGET_EXHAUSTED,
                "task_name": pending["task_name"],
                "target_layer": pending["target_layer"],
                "message": "merge task remains pending because the configured budget was exhausted",
                "details": {
                    "requested_weight_elements": pending["weight_cost"],
                    "requested_compute_units": pending["compute_cost"],
                },
            }
            for pending in self.pending_tasks
        ]


def _task_cost(
    task: TaskSpec, layer_shape: tuple[int, int], adapter: Mapping[str, Any]
) -> tuple[int, int]:
    output_dim, input_dim = layer_shape
    matrix_a_elements = int(np.asarray(adapter["A"]).size)
    matrix_b_elements = int(np.asarray(adapter["B"]).size)
    weight_elements = output_dim * input_dim + matrix_a_elements + matrix_b_elements
    compute_units = 2 * output_dim * input_dim * task.rank + output_dim * input_dim
    return weight_elements, compute_units


def merge_adapters(
    base: BasePackage,
    tasks: list[TaskSpec],
    adapters_by_task: Mapping[str, dict[str, Any]],
    config: PeftConfig,
) -> MergeResult:
    budget = MergeBudget(config.merge_weight_budget, config.merge_compute_budget)
    merged_weight = {
        name: np.array(layer.weight, dtype=np.float64, copy=True)
        for name, layer in base.layers.items()
    }
    merged_bias = {
        name: np.array(layer.bias, dtype=np.float64, copy=True)
        for name, layer in base.layers.items()
    }
    applied: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    exhausted = False
    ordered_tasks = sorted(tasks, key=task_order_key)
    for index, task in enumerate(ordered_tasks):
        adapter = adapters_by_task.get(task.task_name)
        if adapter is None:
            continue
        layer = base.layers[task.target_layer]
        layer_shape = (int(layer.weight.shape[0]), int(layer.weight.shape[1]))
        weight_cost, compute_cost = _task_cost(task, layer_shape, adapter)
        if exhausted:
            pending.append(
                {
                    **task.as_dict(),
                    "weight_cost": weight_cost,
                    "compute_cost": compute_cost,
                }
            )
            continue
        if not budget.request(task, weight_cost, compute_cost):
            exhausted = True
            pending.append(
                {
                    **task.as_dict(),
                    "weight_cost": weight_cost,
                    "compute_cost": compute_cost,
                }
            )
            continue
        delta = adapter_delta(adapter)
        if delta.shape != layer.weight.shape:
            raise BusinessError(
                "E_ADAPTER_CONFIG",
                "adapter matrix delta shape does not match the base layer",
                task_name=task.task_name,
                target_layer=task.target_layer,
            )
        if not np.isfinite(delta).all():
            raise BusinessError(
                "E_ADAPTER_CONFIG",
                "adapter delta contains non-finite values",
                task_name=task.task_name,
                target_layer=task.target_layer,
            )
        merged_weight[task.target_layer] += delta
        applied.append(
            {
                "task_name": task.task_name,
                "target_layer": task.target_layer,
                "rank": task.rank,
                "scale": task.scale,
                "weight_cost": weight_cost,
                "compute_cost": compute_cost,
                "order_index": index,
            }
        )

    quantization_scale = config.quantization_scale
    quantization_step = 1.0 / float(quantization_scale)
    quantized = {
        name: np.rint(weight * quantization_scale) / quantization_scale
        for name, weight in merged_weight.items()
    }
    restored = {name: np.array(weight, copy=True) for name, weight in quantized.items()}
    for entry in applied:
        task_name = entry["task_name"]
        target_layer = entry["target_layer"]
        restored[target_layer] -= adapter_delta(adapters_by_task[task_name])
    restore_errors = {
        name: float(np.max(np.abs(restored[name] - base.layers[name].weight)))
        if restored[name].size
        else 0.0
        for name in sorted(restored)
    }
    maximum_error = max(restore_errors.values(), default=0.0)
    allowed_error = config.allowed_restore_error
    epsilon = 1e-15
    if maximum_error > allowed_error + epsilon:
        raise BusinessError(
            E_MERGE_NOT_REVERSIBLE,
            "merged artifact cannot restore the base within the configured quantization tolerance",
            details={
                "max_abs_error": maximum_error,
                "allowed_restore_error": allowed_error,
                "quantization_scale": quantization_scale,
                "per_layer": restore_errors,
            },
        )
    return MergeResult(
        merged_weight=quantized,
        merged_bias=merged_bias,
        merged_tasks=applied,
        pending_tasks=pending,
        budget=budget,
        restore_errors=restore_errors,
        restore_max_abs_error=maximum_error,
        quantization_scale=quantization_scale,
        quantization_step=quantization_step,
        allowed_restore_error=allowed_error,
    )


def write_merged_package(
    destination: str | Path,
    result: MergeResult,
    base: BasePackage,
) -> dict[str, Any]:
    root = Path(destination)
    layer_directory = root / "layers"
    layer_directory.mkdir(parents=True, exist_ok=True)
    descriptors: list[dict[str, Any]] = []
    for name in sorted(result.merged_weight):
        weight = np.ascontiguousarray(result.merged_weight[name], dtype="<f8")
        bias = np.ascontiguousarray(result.merged_bias[name], dtype="<f8")
        weight_file = f"{safe_component(name)}.weight.bin"
        bias_file = f"{safe_component(name)}.bias.bin"
        atomic_write_bytes(layer_directory / weight_file, weight.tobytes(order="C"))
        atomic_write_bytes(layer_directory / bias_file, bias.tobytes(order="C"))
        descriptors.append(
            {
                "name": name,
                "weight": {
                    "file": f"layers/{weight_file}",
                    "dtype": "<f8",
                    "shape": list(weight.shape),
                },
                "bias": {
                    "file": f"layers/{bias_file}",
                    "dtype": "<f8",
                    "shape": list(bias.shape),
                },
            }
        )
    manifest = {
        "format": MERGED_FORMAT,
        "base_fingerprint": base.fingerprint,
        "fingerprint_algorithm": "base-fp-v1",
        "quantization": {
            "scale": result.quantization_scale,
            "step": result.quantization_step,
            "tolerance_steps": result.allowed_restore_error / result.quantization_step,
            "allowed_restore_error": result.allowed_restore_error,
        },
        "tasks_applied": result.merged_tasks,
        "pending_tasks": result.pending_tasks,
        "restore_max_abs_error": result.restore_max_abs_error,
        "restore_per_layer": result.restore_errors,
        "budget": result.budget.summary(),
        "layers": descriptors,
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def load_merged_package(
    directory: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]], dict[str, np.ndarray[Any, Any]]]:
    root = Path(directory)
    import json

    with open(root / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("format") != MERGED_FORMAT:
        raise ValueError("invalid merged package format")
    weights: dict[str, np.ndarray[Any, Any]] = {}
    biases: dict[str, np.ndarray[Any, Any]] = {}
    for descriptor in manifest.get("layers", []):
        name = descriptor["name"]
        weight_descriptor = descriptor["weight"]
        bias_descriptor = descriptor["bias"]
        weight_shape = tuple(int(value) for value in weight_descriptor["shape"])
        bias_shape = tuple(int(value) for value in bias_descriptor["shape"])
        weights[name] = np.fromfile(root / weight_descriptor["file"], dtype="<f8").reshape(
            weight_shape
        )
        biases[name] = np.fromfile(root / bias_descriptor["file"], dtype="<f8").reshape(
            bias_shape
        )
        if not np.isfinite(weights[name]).all() or not np.isfinite(biases[name]).all():
            raise ValueError("merged package contains non-finite values")
    return manifest, weights, biases


def restore_base_weights(
    merged_weights: Mapping[str, np.ndarray[Any, Any]],
    tasks_applied: list[dict[str, Any]],
    adapters_by_task: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    restored = {
        name: np.array(weight, dtype=np.float64, copy=True)
        for name, weight in merged_weights.items()
    }
    for entry in tasks_applied:
        restored[entry["target_layer"]] -= adapter_delta(dict(adapters_by_task[entry["task_name"]]))
    return restored


def restoration_metrics(
    restored: Mapping[str, np.ndarray[Any, Any]],
    base: BasePackage,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in base.layers:
        difference = restored[name] - base.layers[name].weight
        metrics[name] = float(np.max(np.abs(difference))) if difference.size else 0.0
    return metrics
