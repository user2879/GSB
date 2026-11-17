from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import DiagError, E_SPEC
from .spec import SystemSpec, TaskSpec
from .util import canonical_json, stable_seed


VALUE_SCALE = 1_000_000
SUMMARY_SCHEMA = "deterministic-groups-v1"
# Bounded so the largest allowed sample count cannot overflow int64 sums.
MAX_CONFIG_UNITS = 2**40


def _config_units(value: Any) -> int:
    """Map any JSON configuration value to deterministic fixed-point units."""

    if isinstance(value, bool):
        return VALUE_SCALE if value else 0
    if isinstance(value, int):
        units = int(value) * VALUE_SCALE
        return max(-MAX_CONFIG_UNITS, min(MAX_CONFIG_UNITS, units))
    if isinstance(value, float):
        if not math.isfinite(value):
            return 0
        units = int(round(value * VALUE_SCALE))
        return max(-MAX_CONFIG_UNITS, min(MAX_CONFIG_UNITS, units))
    hashed = stable_seed(canonical_json(value))
    return int(hashed % (10 * VALUE_SCALE)) - 5 * VALUE_SCALE


def _round_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def _aggregate(values: np.ndarray, aggregation: str) -> int:
    if aggregation == "sum":
        return int(np.sum(values, dtype=np.int64))
    if aggregation == "mean":
        return _round_div(int(np.sum(values, dtype=np.int64)), int(values.size))
    if aggregation == "min":
        return int(np.min(values))
    if aggregation == "max":
        return int(np.max(values))
    if aggregation == "count":
        return int(values.size)
    raise DiagError(E_SPEC, f"unsupported aggregation {aggregation!r}", field_path="$.tasks")


def compute_aggregate_summary(
    spec: SystemSpec,
    *,
    cluster_id: str,
    revision_version: int,
    task_id: str,
) -> dict[str, Any]:
    cluster = spec.clusters_by_id.get(cluster_id)
    if cluster is None:
        raise DiagError(
            E_SPEC,
            f"unknown cluster {cluster_id!r}",
            field_path="$.clusters",
        )
    revision = spec.revision(revision_version)
    task = spec.tasks_by_id.get(task_id)
    if task is None:
        raise DiagError(E_SPEC, f"unknown task {task_id!r}", field_path="$.tasks")
    if task_id not in revision.task_ids or cluster_id not in task.targets:
        raise DiagError(
            E_SPEC,
            "task is not assigned to this cluster in the requested revision",
            field_path=f"$.revisions[version={revision_version}].task_ids",
            details={"cluster_id": cluster_id, "task_id": task_id},
        )

    group_count = max(1, min(8, len(cluster.nodes)))
    by_key: dict[str, Any] = {}
    for config_key in sorted(task.config_keys):
        config_value = revision.config[config_key]
        seed = stable_seed(
            spec.seed,
            task.id,
            cluster.id,
            revision.version,
            config_key,
            "samples-v1",
        )
        rng = np.random.default_rng(seed)
        jitter = rng.integers(
            low=-VALUE_SCALE,
            high=VALUE_SCALE + 1,
            size=task.sample_count,
            dtype=np.int64,
        )
        values = jitter + np.int64(_config_units(config_value))
        group_assignments = np.arange(task.sample_count, dtype=np.int64) % group_count
        groups: list[dict[str, Any]] = []
        for group_index in range(group_count):
            group_values = values[group_assignments == group_index]
            groups.append(
                {
                    "group": f"g{group_index}",
                    "count": int(group_values.size),
                    "value_x1e6": _aggregate(group_values, task.aggregation),
                }
            )
        by_key[config_key] = {
            "groups": groups,
            "overall_value_x1e6": _aggregate(values, task.aggregation),
        }

    return {
        "schema": SUMMARY_SCHEMA,
        "aggregation": task.aggregation,
        "value_scale": VALUE_SCALE,
        "grouping": "sample_index_mod_group_count",
        "group_count": group_count,
        "sample_count": task.sample_count,
        "by_key": by_key,
    }


def compute_aggregate_canonical(
    spec: SystemSpec,
    *,
    cluster_id: str,
    revision_version: int,
    task_id: str,
) -> str:
    return canonical_json(
        compute_aggregate_summary(
            spec,
            cluster_id=cluster_id,
            revision_version=revision_version,
            task_id=task_id,
        )
    )
