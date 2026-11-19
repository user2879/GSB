from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PeftConfig
from .errors import E_ADAPTER_CONFIG, E_TASK_INVALID, BusinessError, Failure
from .canonical import sha256_object


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    target_layer: str
    rank: int
    scale: float
    ordinal: int
    stable_key: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "target_layer": self.target_layer,
            "rank": self.rank,
            "scale": self.scale,
            "ordinal": self.ordinal,
            "stable_key": self.stable_key,
        }


def _rank(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BusinessError(E_ADAPTER_CONFIG, "adapter rank must be a positive integer")
    return value


def _scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BusinessError(E_ADAPTER_CONFIG, "adapter scale must be a positive finite number")
    number = float(value)
    if not number > 0.0 or not math.isfinite(number):
        raise BusinessError(E_ADAPTER_CONFIG, "adapter scale must be a positive finite number")
    return number


def task_order_key(task: TaskSpec) -> tuple[str, str, str]:
    return (task.task_name, task.target_layer, task.stable_key)


def load_tasks(
    path: str | Path,
    config: PeftConfig,
    base_layer_names: set[str] | None = None,
) -> tuple[list[TaskSpec], list[Failure]]:
    failures: list[Failure] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        failures.append(
            Failure(E_TASK_INVALID, f"task manifest cannot be read: {exc}")
        )
        return [], failures
    if isinstance(payload, list):
        raw_tasks = payload
    elif isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        raw_tasks = payload["tasks"]
    else:
        failures.append(Failure(E_TASK_INVALID, "task manifest must contain a tasks list"))
        return [], failures

    whitelist = set(config.target_layer_whitelist)
    rank_candidates = set(config.rank_candidates)
    scale_candidates = set(config.scale_candidates)
    seen_names: set[str] = set()
    tasks: list[TaskSpec] = []
    for index, raw in enumerate(raw_tasks):
        fallback_name = f"task[{index}]"
        if not isinstance(raw, dict):
            failures.append(
                Failure(E_TASK_INVALID, "task entry must be an object", task_name=fallback_name)
            )
            continue
        raw_name = raw.get("task_name", raw.get("name"))
        task_name = raw_name.strip() if isinstance(raw_name, str) else ""
        display_name = task_name or fallback_name
        target_layer = raw.get("target_layer")
        target_display = target_layer if isinstance(target_layer, str) else None
        if not task_name:
            failures.append(
                Failure(
                    E_TASK_INVALID,
                    "task_name must be a non-empty string",
                    task_name=display_name,
                    target_layer=target_display,
                )
            )
            continue
        if task_name in seen_names:
            failures.append(
                Failure(
                    E_TASK_INVALID,
                    f"duplicate task_name: {task_name}",
                    task_name=task_name,
                    target_layer=target_display,
                )
            )
            continue
        seen_names.add(task_name)
        if not isinstance(target_layer, str) or not target_layer.strip():
            failures.append(
                Failure(
                    E_TASK_INVALID,
                    "target_layer must be a non-empty string",
                    task_name=task_name,
                )
            )
            continue
        target_layer = target_layer.strip()
        if target_layer not in whitelist:
            failures.append(
                Failure(
                    E_TASK_INVALID,
                    "target layer is not in the PEFT whitelist",
                    task_name=task_name,
                    target_layer=target_layer,
                )
            )
            continue
        if base_layer_names is not None and target_layer not in base_layer_names:
            failures.append(
                Failure(
                    E_TASK_INVALID,
                    "target layer does not exist in the base package",
                    task_name=task_name,
                    target_layer=target_layer,
                )
            )
            continue
        try:
            rank = _rank(raw.get("rank", raw.get("adapter_rank")))
            scale = _scale(raw.get("scale", raw.get("scaling_factor")))
        except BusinessError as exc:
            failures.append(
                Failure(
                    exc.code,
                    exc.message,
                    task_name=task_name,
                    target_layer=target_layer,
                )
            )
            continue
        if rank not in rank_candidates:
            failures.append(
                Failure(
                    E_ADAPTER_CONFIG,
                    "adapter rank is not one of the configured candidates",
                    task_name=task_name,
                    target_layer=target_layer,
                    details={"rank": rank, "candidates": list(config.rank_candidates)},
                )
            )
            continue
        if scale not in scale_candidates:
            failures.append(
                Failure(
                    E_ADAPTER_CONFIG,
                    "adapter scale is not one of the configured candidates",
                    task_name=task_name,
                    target_layer=target_layer,
                    details={"scale": scale, "candidates": list(config.scale_candidates)},
                )
            )
            continue
        stable_key = "sha256:" + sha256_object(
            {"task_name": task_name, "target_layer": target_layer}
        )
        tasks.append(
            TaskSpec(
                task_name=task_name,
                target_layer=target_layer,
                rank=rank,
                scale=scale,
                ordinal=index,
                stable_key=stable_key,
            )
        )
    tasks.sort(key=task_order_key)
    failures.sort(
        key=lambda failure: (
            failure.task_name or "",
            failure.target_layer or "",
            failure.code,
            failure.message,
        )
    )
    return tasks, failures


def task_set_hash(tasks: list[TaskSpec]) -> str:
    return sha256_object([task.as_dict() for task in tasks])
