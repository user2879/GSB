"""分批更新计划：解析与累计状态应用，全部为确定性纯数据变换。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .errors import E_BATCH_INVALID, Failure
from .loader import _read_json
from .models import (
    BuildConfig,
    ScoreParams,
    Seed,
    SlotSource,
    WorkspaceState,
)
from .utils import is_nonneg_finite_number


@dataclass(frozen=True)
class BatchPlan:
    batch_no: int
    operations: list[dict[str, Any]]


def load_batch_plan(path: str, failures: list[Failure]) -> list[BatchPlan]:
    try:
        obj = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(Failure(E_BATCH_INVALID, path, path, str(exc)))
        return []
    if not isinstance(obj, dict) or not isinstance(obj.get("batches"), list):
        failures.append(Failure(E_BATCH_INVALID, path, path, "root.batches must be a list"))
        return []
    plans: list[BatchPlan] = []
    seen: set[int] = set()
    for index, raw in enumerate(obj["batches"]):
        ref = f"batches[{index}]"
        if not isinstance(raw, dict):
            failures.append(Failure(E_BATCH_INVALID, path, ref, "batch must be an object"))
            continue
        batch_no = raw.get("batch_no")
        if not isinstance(batch_no, int) or isinstance(batch_no, bool) or batch_no < 1:
            failures.append(Failure(E_BATCH_INVALID, path, ref, "batch_no must be positive int"))
            continue
        if batch_no in seen:
            failures.append(Failure(E_BATCH_INVALID, path, ref, f"duplicate batch_no {batch_no}"))
            continue
        operations = raw.get("operations", [])
        if not isinstance(operations, list):
            failures.append(Failure(E_BATCH_INVALID, path, ref, "operations must be a list"))
            continue
        valid_ops: list[dict[str, Any]] = []
        for op_index, operation in enumerate(operations):
            problem = _validate_operation(operation)
            if problem:
                failures.append(
                    Failure(
                        E_BATCH_INVALID,
                        path,
                        f"{ref}.operations[{op_index}]",
                        problem,
                        batch_no=batch_no,
                    )
                )
                continue
            valid_ops.append(operation)
        seen.add(batch_no)
        plans.append(BatchPlan(batch_no=batch_no, operations=valid_ops))
    plans.sort(key=lambda plan: plan.batch_no)
    return plans


def _validate_operation(operation: Any) -> str | None:
    if not isinstance(operation, dict) or not isinstance(operation.get("op"), str):
        return "operation needs a string op"
    kind = operation["op"]
    allowed = {
        "replace_template",
        "delete_template",
        "add_slot_source",
        "delete_slot_source",
        "set_target_count",
        "set_template_weight",
        "set_slot_weight",
        "set_score_weight",
        "set_score_baseline",
        "set_dedup_threshold",
        "set_boundary_rule",
    }
    if kind not in allowed:
        return f"unknown op {kind}"
    return None


def apply_batch(state: WorkspaceState, plan: BatchPlan) -> WorkspaceState:
    seeds = dict(state.seeds)
    sources = dict(state.slot_sources)
    config = state.config
    params = state.score_params
    target_counts = dict(config.target_counts)
    template_weights = {task: dict(table) for task, table in config.template_weights.items()}
    slot_weights = {task: dict(table) for task, table in config.slot_weights.items()}
    dedup_threshold = config.dedup_threshold
    boundary_rule = config.boundary_rule
    specs = {name: type(spec)(spec.weight, spec.baseline) for name, spec in params.specs.items()}
    max_line = max((seed.line_no for seed in seeds.values()), default=0)

    for operation in plan.operations:
        kind = operation["op"]
        if kind == "replace_template":
            payload = operation["seed"]
            seed_id = str(payload["seed_id"])
            old = seeds.get(seed_id)
            max_line += 1
            seeds[seed_id] = Seed(
                seed_id=seed_id,
                task_type=str(payload["task_type"]),
                template_id=str(payload["template_id"]),
                template=str(payload["template"]),
                slots=tuple(str(name) for name in payload.get("slots", [])),
                answer_template=str(payload["answer_template"]),
                line_no=max_line,
            )
            if old is not None and old.template_id != payload["template_id"]:
                table = template_weights.get(old.task_type, {})
                table.pop(old.template_id, None)
        elif kind == "delete_template":
            seed = seeds.pop(str(operation["seed_id"]), None)
            if seed is not None:
                template_weights.get(seed.task_type, {}).pop(seed.template_id, None)
        elif kind == "add_slot_source":
            payload = operation["source"]
            sources[str(payload["source_id"])] = SlotSource(
                source_id=str(payload["source_id"]),
                task_type=str(payload["task_type"]),
                slot_name=str(payload["slot_name"]),
                values=tuple(str(value) for value in payload["values"]),
            )
        elif kind == "delete_slot_source":
            sources.pop(str(operation["source_id"]), None)
        elif kind == "set_target_count":
            target_counts[str(operation["task_type"])] = int(operation["value"])
        elif kind == "set_template_weight":
            task = str(operation["task_type"])
            template_weights.setdefault(task, {})[str(operation["template_id"])] = float(
                operation["value"]
            )
        elif kind == "set_slot_weight":
            task = str(operation["task_type"])
            slot_weights.setdefault(task, {})[str(operation["source_id"])] = float(
                operation["value"]
            )
        elif kind == "set_score_weight":
            dim = str(operation["dimension"])
            specs[dim] = type(specs[dim])(float(operation["value"]), specs[dim].baseline)
        elif kind == "set_score_baseline":
            dim = str(operation["dimension"])
            specs[dim] = type(specs[dim])(specs[dim].weight, float(operation["value"]))
        elif kind == "set_dedup_threshold":
            dedup_threshold = float(operation["value"])
        elif kind == "set_boundary_rule":
            boundary_rule = str(operation["value"])

    new_config = BuildConfig(
        task_types=config.task_types,
        target_counts=target_counts,
        template_weights={
            task: dict(sorted(table.items())) for task, table in template_weights.items()
        },
        slot_weights={
            task: dict(sorted(table.items())) for task, table in slot_weights.items()
        },
        dedup_threshold=dedup_threshold,
        boundary_rule=boundary_rule,
        max_workers=config.max_workers,
        snapshot_interval_batches=config.snapshot_interval_batches,
    )
    new_params = ScoreParams(
        dimensions=params.dimensions,
        specs=specs,
        length_lower=params.length_lower,
        length_upper=params.length_upper,
    )
    return WorkspaceState(
        seeds=seeds,
        slot_sources=sources,
        config=new_config,
        score_params=new_params,
        vocab=state.vocab,
    )


def validate_runtime_weights(
    state: WorkspaceState, failures: list[Failure], batch_no: int
) -> None:
    """批次应用后校验权重仍覆盖现存模板/来源且和为正，非法则入失败清单。"""
    config = state.config
    for task in config.task_types:
        seeds = [seed for seed in state.seeds.values() if seed.task_type == task]
        template_ids = {seed.template_id for seed in seeds}
        table = config.template_weights.get(task, {})
        positive = {
            tid: weight
            for tid, weight in table.items()
            if tid in template_ids and is_nonneg_finite_number(weight) and weight > 0.0
        }
        if seeds and not positive:
            failures.append(
                Failure(
                    E_BATCH_INVALID,
                    "batch_plan",
                    f"batch {batch_no} task {task}",
                    "no positive template weight for any surviving template",
                    batch_no=batch_no,
                )
            )
        source_ids = {source.source_id for source in state.slot_sources.values()
                      if source.task_type == task}
        slot_table = config.slot_weights.get(task, {})
        positive_slots = {
            sid: weight
            for sid, weight in slot_table.items()
            if sid in source_ids and is_nonneg_finite_number(weight) and weight > 0.0
        }
        if source_ids and not positive_slots:
            failures.append(
                Failure(
                    E_BATCH_INVALID,
                    "batch_plan",
                    f"batch {batch_no} task {task}",
                    "no positive slot weight for any surviving slot source",
                    batch_no=batch_no,
                )
            )
