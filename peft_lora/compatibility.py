from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BasePackage
from .canonical import sha256_object
from .errors import E_ADAPTER_CONFIG, E_BASE_VERSION_MISMATCH, BusinessError
from .tasks import TaskSpec
from .training import validate_adapter_payload


@dataclass(frozen=True)
class ExistingAdapter:
    payload: dict[str, Any]
    artifact_file: str
    snapshot_version: int


def compatibility_hash(entries: list[dict[str, Any]]) -> str:
    return sha256_object(entries)


def evaluate_compatibility(
    task: TaskSpec,
    base: BasePackage,
    existing: ExistingAdapter | None,
) -> dict[str, Any]:
    base_entry = {
        "task_name": task.task_name,
        "target_layer": task.target_layer,
        "rank": task.rank,
        "scale": task.scale,
        "current_base_fingerprint": base.fingerprint,
        "current_layer_fingerprint": base.layer_fingerprints[task.target_layer],
        "adapter_base_fingerprint": None,
        "adapter_layer_fingerprint": None,
        "existing_adapter": None,
        "snapshot_version": None,
        "reusable": False,
        "action": "TRAIN",
        "status": "NO_EXISTING_ADAPTER",
        "business_code": None,
        "reason": "no exported adapter was found for this task",
    }
    if existing is None:
        return base_entry
    base_entry.update(
        {
            "adapter_base_fingerprint": existing.payload.get("base_fingerprint"),
            "adapter_layer_fingerprint": existing.payload.get("target_layer_fingerprint"),
            "existing_adapter": existing.artifact_file,
            "snapshot_version": existing.snapshot_version,
        }
    )
    layer = base.layers[task.target_layer]
    try:
        validate_adapter_payload(
            existing.payload,
            expected_task=task,
            expected_layer_shape=(int(layer.weight.shape[0]), int(layer.weight.shape[1])),
        )
    except BusinessError as exc:
        base_entry.update(
            {
                "status": "ADAPTER_INVALID",
                "business_code": E_ADAPTER_CONFIG,
                "reason": exc.message,
            }
        )
        return base_entry

    adapter_base_fingerprint = existing.payload.get("base_fingerprint")
    adapter_layer_fingerprint = existing.payload.get("target_layer_fingerprint")
    current_layer_fingerprint = base.layer_fingerprints[task.target_layer]
    if adapter_base_fingerprint == base.fingerprint:
        base_entry.update(
            {
                "reusable": True,
                "action": "REUSE",
                "status": "COMPATIBLE_EXACT",
                "reason": "adapter base fingerprint exactly matches the current base",
            }
        )
        return base_entry
    if adapter_layer_fingerprint == current_layer_fingerprint:
        base_entry.update(
            {
                "reusable": True,
                "action": "REUSE",
                "status": "COMPATIBLE_LAYER_UNCHANGED",
                "reason": "base changed, but the adapter target layer is byte-for-byte unchanged",
            }
        )
        return base_entry
    base_entry.update(
        {
            "status": "INCOMPATIBLE_BASE_VERSION",
            "business_code": E_BASE_VERSION_MISMATCH,
            "reason": "adapter target layer fingerprint differs from the current base layer",
        }
    )
    return base_entry


def build_compatibility_table(
    tasks: list[TaskSpec],
    base: BasePackage,
    existing_by_task: dict[str, ExistingAdapter],
) -> list[dict[str, Any]]:
    return [
        evaluate_compatibility(task, base, existing_by_task.get(task.task_name))
        for task in tasks
    ]

