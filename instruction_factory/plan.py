from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .constants import DIMENSION_ORDER, E_CONFIG_INVALID, E_WEIGHT_INVALID
from .errors import Failure, FailureCollector, PlanApplyError
from .loader import LoadedInputs, PLACEHOLDER_RE
from .util import as_decimal, fingerprint_object, finite_number, is_safe_id


@dataclass
class ConstructionState:
    config: dict[str, Any]
    scoring: dict[str, Any]
    templates: dict[str, dict[str, dict[str, Any]]]
    lexicon: dict[str, Any]

    def clone(self) -> "ConstructionState":
        return ConstructionState(
            config=copy.deepcopy(self.config),
            scoring=copy.deepcopy(self.scoring),
            templates=copy.deepcopy(self.templates),
            lexicon=copy.deepcopy(self.lexicon),
        )

    @property
    def tasks(self) -> dict[str, dict[str, Any]]:
        return self.config["task_types"]

    @property
    def quotas(self) -> dict[str, int]:
        return self.config["task_quotas"]

    @property
    def sources(self) -> dict[str, list[str]]:
        return self.lexicon["sources"]

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "scoring": self.scoring,
            "templates": self.templates,
            "lexicon": self.lexicon,
        }

    def fingerprint(self) -> str:
        return fingerprint_object(self.to_manifest_dict())


def initial_state(inputs: LoadedInputs) -> ConstructionState:
    return ConstructionState(
        config=copy.deepcopy(inputs.config),
        scoring=copy.deepcopy(inputs.scoring),
        templates=copy.deepcopy(inputs.seeds_by_variant),
        lexicon=copy.deepcopy(inputs.lexicon),
    )


def state_from_manifest(data: dict[str, Any]) -> ConstructionState:
    return ConstructionState(
        config=copy.deepcopy(data["config"]),
        scoring=copy.deepcopy(data["scoring"]),
        templates=copy.deepcopy(data["templates"]),
        lexicon=copy.deepcopy(data["lexicon"]),
    )


def _failure(
    code: str,
    message: str,
    *,
    inputs: LoadedInputs,
    batch_id: str,
    operation_index: int,
    field: str,
) -> Failure:
    prefix = f"batches.{batch_id}.operations[{operation_index}]"
    return Failure(
        code=code,
        message=message,
        stage="plan",
        file=str(inputs.paths.plan),
        field=f"{prefix}.{field}",
        path=f"{prefix}.{field}",
        batch_id=batch_id,
        fatal=True,
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_weight(value: Any) -> bool:
    return finite_number(value) and as_decimal(value) >= 0


def _weight_sum(weights: dict[str, float]) -> Any:
    total = as_decimal(0)
    for value in weights.values():
        total += as_decimal(value)
    return total


def _validate_state_failures(
    state: ConstructionState,
    *,
    inputs: LoadedInputs,
    batch_id: str,
    default_operation_index: int,
) -> list[Failure]:
    failures: list[Failure] = []

    def add(code: str, message: str, field_path: str) -> None:
        failures.append(
            _failure(
                code,
                message,
                inputs=inputs,
                batch_id=batch_id,
                operation_index=default_operation_index,
                field=field_path,
            )
        )

    for task_type in sorted(state.tasks):
        task = state.tasks[task_type]
        if not _positive_int(task.get("target_count")):
            add(E_CONFIG_INVALID, f"task {task_type!r} target_count must be a positive integer", f"task_types.{task_type}.target_count")
        if not _positive_int(state.quotas.get(task_type)):
            add(E_CONFIG_INVALID, f"task {task_type!r} quota must be a positive integer", f"task_quotas.{task_type}")
        template_weights = task.get("template_weights")
        if not isinstance(template_weights, dict) or not template_weights:
            add(E_WEIGHT_INVALID, f"task {task_type!r} template_weights must be non-empty", f"task_types.{task_type}.template_weights")
        else:
            if _weight_sum(template_weights) <= 0:
                add(E_WEIGHT_INVALID, f"task {task_type!r} template weight sum must be positive", f"task_types.{task_type}.template_weights")
            for variant_id, weight in template_weights.items():
                if not _nonnegative_weight(weight):
                    add(E_WEIGHT_INVALID, f"template weight {variant_id!r} must be non-negative and finite", f"task_types.{task_type}.template_weights.{variant_id}")
                if variant_id not in state.templates.get(task_type, {}):
                    add(E_CONFIG_INVALID, f"template variant {variant_id!r} has no definition", f"task_types.{task_type}.template_weights.{variant_id}")
        source_weights = task.get("slot_source_weights")
        if not isinstance(source_weights, dict) or not source_weights:
            add(E_WEIGHT_INVALID, f"task {task_type!r} slot_source_weights must be non-empty", f"task_types.{task_type}.slot_source_weights")
        else:
            if _weight_sum(source_weights) <= 0:
                add(E_WEIGHT_INVALID, f"task {task_type!r} slot source weight sum must be positive", f"task_types.{task_type}.slot_source_weights")
            for source_name, weight in source_weights.items():
                if not _nonnegative_weight(weight):
                    add(E_WEIGHT_INVALID, f"source weight {source_name!r} must be non-negative and finite", f"task_types.{task_type}.slot_source_weights.{source_name}")
                if source_name not in state.sources:
                    add(E_CONFIG_INVALID, f"slot source {source_name!r} is absent from lexicon", f"task_types.{task_type}.slot_source_weights.{source_name}")
    score_total = as_decimal(0)
    dimensions = state.scoring.get("dimensions", {})
    for dimension in DIMENSION_ORDER:
        item = dimensions.get(dimension)
        if not isinstance(item, dict) or not _nonnegative_weight(item.get("weight")):
            add(E_WEIGHT_INVALID, f"score dimension {dimension!r} weight must be non-negative and finite", f"scoring.dimensions.{dimension}.weight")
            continue
        score_total += as_decimal(item["weight"])
        if not finite_number(item.get("baseline")) or not 0 <= as_decimal(item["baseline"]) <= 100:
            add(E_CONFIG_INVALID, f"score dimension {dimension!r} baseline must be in [0, 100]", f"scoring.dimensions.{dimension}.baseline")
    if score_total <= 0:
        add(E_WEIGHT_INVALID, "score dimension weight sum must be positive", "scoring.dimensions")
    if not finite_number(state.scoring.get("pass_threshold")) or not 0 <= as_decimal(state.scoring["pass_threshold"]) <= 100:
        add(E_CONFIG_INVALID, "pass_threshold must be in [0, 100]", "scoring.pass_threshold")
    if not finite_number(state.config.get("near_duplicate_threshold")) or not 0 <= as_decimal(state.config["near_duplicate_threshold"]) <= 1:
        add(E_CONFIG_INVALID, "near_duplicate_threshold must be in [0, 1]", "near_duplicate_threshold")
    return failures


def apply_batch(state: ConstructionState, batch: dict[str, Any], inputs: LoadedInputs) -> ConstructionState:
    updated = state.clone()
    failures: list[Failure] = []
    batch_id = batch["batch_id"]

    for index, operation in enumerate(batch["operations"]):
        op = operation["op"]
        task_type = operation.get("task_type")
        if op == "add_task":
            if task_type in updated.tasks:
                failures.append(_failure(E_CONFIG_INVALID, f"task {task_type!r} already exists", inputs=inputs, batch_id=batch_id, operation_index=index, field="task_type"))
                continue
            updated.tasks[task_type] = {
                "target_count": operation["target_count"],
                "template_weights": copy.deepcopy(operation["template_weights"]),
                "slot_source_weights": copy.deepcopy(operation["slot_source_weights"]),
            }
            updated.quotas[task_type] = operation["task_quota"]
        elif op == "delete_task":
            if task_type not in updated.tasks:
                failures.append(_failure(E_CONFIG_INVALID, f"task {task_type!r} does not exist", inputs=inputs, batch_id=batch_id, operation_index=index, field="task_type"))
                continue
            del updated.tasks[task_type]
            updated.quotas.pop(task_type, None)
        elif op == "set_target":
            if task_type not in updated.tasks:
                failures.append(_failure(E_CONFIG_INVALID, f"task {task_type!r} does not exist", inputs=inputs, batch_id=batch_id, operation_index=index, field="task_type"))
            elif not _positive_int(operation.get("target_count")):
                failures.append(_failure(E_CONFIG_INVALID, "target_count must be a positive integer", inputs=inputs, batch_id=batch_id, operation_index=index, field="target_count"))
            else:
                updated.tasks[task_type]["target_count"] = operation["target_count"]
        elif op == "set_template_weights":
            if task_type not in updated.tasks:
                failures.append(_failure(E_CONFIG_INVALID, f"task {task_type!r} does not exist", inputs=inputs, batch_id=batch_id, operation_index=index, field="task_type"))
                continue
            weights = operation.get("template_weights")
            if not isinstance(weights, dict) or not weights or _weight_sum(weights) <= 0 or any(not _nonnegative_weight(value) for value in weights.values()):
                failures.append(_failure(E_WEIGHT_INVALID, "template_weights must be non-empty, non-negative, finite, and have a positive sum", inputs=inputs, batch_id=batch_id, operation_index=index, field="template_weights"))
                continue
            missing = sorted(set(weights) - set(updated.templates.get(task_type, {})))
            if missing:
                failures.append(_failure(E_CONFIG_INVALID, f"missing template definitions: {', '.join(missing)}", inputs=inputs, batch_id=batch_id, operation_index=index, field="template_weights"))
                continue
            updated.tasks[task_type]["template_weights"] = copy.deepcopy(weights)
        elif op == "replace_template":
            if task_type not in updated.tasks:
                failures.append(_failure(E_CONFIG_INVALID, f"task {task_type!r} does not exist", inputs=inputs, batch_id=batch_id, operation_index=index, field="task_type"))
                continue
            variant_id = operation["variant_id"]
            instruction_template = operation["instruction_template"]
            placeholders = sorted(set(PLACEHOLDER_RE.findall(instruction_template)))
            slots = operation.get("slots")
            if slots is None:
                existing = updated.templates.get(task_type, {}).get(variant_id)
                slots = list(existing["slots"]) if existing else placeholders
            reference_answer = operation.get("reference_answer", "")
            missing = sorted(set(placeholders) - set(slots))
            if missing:
                failures.append(_failure(E_CONFIG_INVALID, f"replacement template is missing slots: {', '.join(missing)}", inputs=inputs, batch_id=batch_id, operation_index=index, field="slots"))
                continue
            updated.templates.setdefault(task_type, {})[variant_id] = {
                "seed_id": f"plan:{task_type}:{variant_id}",
                "task_type": task_type,
                "template_variant_id": variant_id,
                "instruction_template": instruction_template,
                "slots": list(slots),
                "reference_answer": reference_answer,
                "line": None,
            }
        elif op == "delete_template":
            if task_type not in updated.tasks:
                failures.append(_failure(E_CONFIG_INVALID, f"task {task_type!r} does not exist", inputs=inputs, batch_id=batch_id, operation_index=index, field="task_type"))
                continue
            variant_id = operation["variant_id"]
            weights = updated.tasks[task_type]["template_weights"]
            if variant_id not in weights:
                failures.append(_failure(E_CONFIG_INVALID, f"template variant {variant_id!r} is not active for task {task_type!r}", inputs=inputs, batch_id=batch_id, operation_index=index, field="variant_id"))
                continue
            remaining = {key: value for key, value in weights.items() if key != variant_id}
            if not remaining or _weight_sum(remaining) <= 0:
                failures.append(_failure(E_WEIGHT_INVALID, "deleting this template would leave no positive template weight", inputs=inputs, batch_id=batch_id, operation_index=index, field="variant_id"))
                continue
            updated.tasks[task_type]["template_weights"] = remaining
        elif op == "set_slot_source_weights":
            if task_type not in updated.tasks:
                failures.append(_failure(E_CONFIG_INVALID, f"task {task_type!r} does not exist", inputs=inputs, batch_id=batch_id, operation_index=index, field="task_type"))
                continue
            weights = operation.get("slot_source_weights")
            if not isinstance(weights, dict) or not weights or _weight_sum(weights) <= 0 or any(not _nonnegative_weight(value) for value in weights.values()):
                failures.append(_failure(E_WEIGHT_INVALID, "slot_source_weights must be non-empty, non-negative, finite, and have a positive sum", inputs=inputs, batch_id=batch_id, operation_index=index, field="slot_source_weights"))
                continue
            unknown = sorted(set(weights) - set(updated.sources))
            if unknown:
                failures.append(_failure(E_CONFIG_INVALID, f"unknown slot sources: {', '.join(unknown)}", inputs=inputs, batch_id=batch_id, operation_index=index, field="slot_source_weights"))
                continue
            updated.tasks[task_type]["slot_source_weights"] = copy.deepcopy(weights)
        elif op == "add_slot_source":
            if task_type not in updated.tasks:
                failures.append(_failure(E_CONFIG_INVALID, f"task {task_type!r} does not exist", inputs=inputs, batch_id=batch_id, operation_index=index, field="task_type"))
                continue
            source_name = operation["source"]
            if source_name not in updated.sources:
                failures.append(_failure(E_CONFIG_INVALID, f"unknown slot source {source_name!r}", inputs=inputs, batch_id=batch_id, operation_index=index, field="source"))
                continue
            weights = updated.tasks[task_type]["slot_source_weights"]
            if source_name in weights:
                failures.append(_failure(E_CONFIG_INVALID, f"slot source {source_name!r} is already active", inputs=inputs, batch_id=batch_id, operation_index=index, field="source"))
                continue
            if not _nonnegative_weight(operation.get("weight")):
                failures.append(_failure(E_WEIGHT_INVALID, "slot source weight must be non-negative and finite", inputs=inputs, batch_id=batch_id, operation_index=index, field="weight"))
                continue
            weights[source_name] = float(operation["weight"])
        elif op == "remove_slot_source":
            if task_type not in updated.tasks:
                failures.append(_failure(E_CONFIG_INVALID, f"task {task_type!r} does not exist", inputs=inputs, batch_id=batch_id, operation_index=index, field="task_type"))
                continue
            source_name = operation["source"]
            weights = updated.tasks[task_type]["slot_source_weights"]
            if source_name not in weights:
                failures.append(_failure(E_CONFIG_INVALID, f"slot source {source_name!r} is not active", inputs=inputs, batch_id=batch_id, operation_index=index, field="source"))
                continue
            remaining = {key: value for key, value in weights.items() if key != source_name}
            if not remaining or _weight_sum(remaining) <= 0:
                failures.append(_failure(E_WEIGHT_INVALID, "deleting this slot source would leave no positive source weight", inputs=inputs, batch_id=batch_id, operation_index=index, field="source"))
                continue
            updated.tasks[task_type]["slot_source_weights"] = remaining
        elif op == "set_score_weight":
            dimension = operation["dimension"]
            weight = operation.get("weight")
            if dimension not in DIMENSION_ORDER or not _nonnegative_weight(weight):
                failures.append(_failure(E_WEIGHT_INVALID, "score weight must target a known dimension with a non-negative finite weight", inputs=inputs, batch_id=batch_id, operation_index=index, field="weight"))
                continue
            updated.scoring["dimensions"][dimension]["weight"] = float(weight)
            if _weight_sum({name: item["weight"] for name, item in updated.scoring["dimensions"].items()}) <= 0:
                failures.append(_failure(E_WEIGHT_INVALID, "score dimension weight sum must remain positive", inputs=inputs, batch_id=batch_id, operation_index=index, field="weight"))
        elif op == "set_quality_threshold":
            value = operation["pass_threshold"]
            if not finite_number(value) or not 0 <= as_decimal(value) <= 100:
                failures.append(_failure(E_CONFIG_INVALID, "pass_threshold must be in [0, 100]", inputs=inputs, batch_id=batch_id, operation_index=index, field="pass_threshold"))
                continue
            updated.scoring["pass_threshold"] = float(value)
        elif op == "set_near_duplicate_threshold":
            value = operation["threshold"]
            if not finite_number(value) or not 0 <= as_decimal(value) <= 1:
                failures.append(_failure(E_CONFIG_INVALID, "threshold must be in [0, 1]", inputs=inputs, batch_id=batch_id, operation_index=index, field="threshold"))
                continue
            updated.config["near_duplicate_threshold"] = float(value)
        else:
            failures.append(_failure(E_CONFIG_INVALID, f"unknown plan operation {op!r}", inputs=inputs, batch_id=batch_id, operation_index=index, field="op"))

        if failures:
            raise PlanApplyError(failures)

    failures.extend(
        _validate_state_failures(
            updated,
            inputs=inputs,
            batch_id=batch_id,
            default_operation_index=max(0, len(batch["operations"]) - 1),
        )
    )
    if failures:
        raise PlanApplyError(failures)
    return updated


def final_state(inputs: LoadedInputs) -> tuple[ConstructionState, list[Failure]]:
    state = initial_state(inputs)
    failures: list[Failure] = []
    for batch in inputs.plan["batches"]:
        try:
            state = apply_batch(state, batch, inputs)
        except PlanApplyError as exc:
            failures.extend(exc.failures)
            break
    return state, failures


def plan_apply_failures_to_collector(
    failures: list[Failure],
    collector: FailureCollector,
) -> None:
    collector.extend(failures)

