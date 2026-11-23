from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from drift_diag.config import EvaluationConfig
from drift_diag.declarations import normalize_declaration
from drift_diag.drift import compare_field
from drift_diag.models import FIELD_INDEX, FIELD_ORDER, TargetSet


ROOT = Path(__file__).resolve().parents[1]


def sample_path(name: str) -> Path:
    return ROOT / "samples" / name


def write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_ndjson(path: Path, events: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False, separators=(",", ":")) for event in events)
        + "\n",
        encoding="utf-8",
    )
    return path


def target_v3() -> TargetSet:
    raw = json.loads(sample_path("declaration-v3.json").read_text(encoding="utf-8"))
    return normalize_declaration(raw, defaults_version="defaults-v1")


def complete_actual(target: TargetSet) -> dict[str, dict[str, Any]]:
    return {
        workload.name: deepcopy(workload.fields)
        for workload in target.ordered_workloads()
    }


def decisions_for(
    target: TargetSet,
    actual_state: dict[str, dict[str, Any]],
    *,
    scale_exp: int = 3,
) -> list[dict[str, Any]]:
    decisions = [
        compare_field(workload, field, actual_state, scale_exp=scale_exp)
        for workload in target.ordered_workloads()
        for field in FIELD_ORDER
    ]
    decisions.sort(key=lambda decision: (decision["workload"], FIELD_INDEX[decision["field"]]))
    return decisions


def config(**overrides: Any) -> EvaluationConfig:
    values = {
        "scale_exp": 3,
        "scale_version": "fixed-point-v1",
        "tolerance_num": 0,
        "tolerance_den": 1,
        "global_limit": 3,
        "per_workload_quota": 1,
        "task_delay_ms": 0.0,
        "batch_events": 4,
    }
    values.update(overrides)
    return EvaluationConfig(**values).validate()
