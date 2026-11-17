"""Independent exception-injection planning and hit accounting."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .data import AdapterConfigDocument
from .errors import PipelineConfigError


_EXPECTED_CODES = {
    "missing_sample": "E_TASK_DESC_INVALID",
    "missing_sample_id": "E_TASK_DESC_INVALID",
    "illegal_classes": "E_TASK_DESC_INVALID",
    "invalid_classes": "E_TASK_DESC_INVALID",
    "zero_rank": "E_ADAPTER_CONFIG_INVALID",
    "negative_lr": "E_ADAPTER_CONFIG_INVALID",
    "train_error": "E_TRAINING_FAILED",
    "evaluate_error": "E_EVALUATION_FAILED",
    "eval_error": "E_EVALUATION_FAILED",
}


@dataclass(frozen=True)
class InjectionPlan:
    tasks_document: Any
    adapter_config: AdapterConfigDocument
    applications: tuple[Mapping[str, Any], ...]
    injection_ids_by_task: Mapping[str, tuple[str, ...]]
    forced_errors: Mapping[str, tuple[str, ...]]
    public_document: Any

    def report_hits(self, failures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        failure_pairs = {
            (str(item.get("task_name") or ""), str(item.get("code")), injection_id)
            for item in failures
            for injection_id in item.get("injection_ids", [])
        }
        result: list[dict[str, Any]] = []
        for application in self.applications:
            expected_code = _EXPECTED_CODES.get(str(application["type"]))
            target = str(application["target"])
            hit = bool(
                application.get("applied")
                and expected_code
                and (target, expected_code, str(application["id"])) in failure_pairs
            )
            item = copy.deepcopy(dict(application))
            item["expected_code"] = expected_code
            item["hit"] = hit
            result.append(item)
        result.sort(key=lambda item: (str(item.get("id", "")), str(item.get("target", ""))))
        return result


def _task_list(document: Any) -> list[Any]:
    if isinstance(document, list):
        return document
    return document.get("tasks", [])


def _task_name(task: Any) -> str | None:
    if isinstance(task, Mapping):
        value = task.get("name", task.get("task_name"))
        if isinstance(value, str) and value:
            return value
    return None


def build_injection_plan(
    tasks_document: Any,
    adapter_config: AdapterConfigDocument,
    injection_document: Any | None,
) -> InjectionPlan:
    """Apply documented fault examples without touching unrelated tasks."""

    copied_tasks = copy.deepcopy(tasks_document)
    copied_config = AdapterConfigDocument.from_mapping(copy.deepcopy(dict(adapter_config.raw)))
    if injection_document is None:
        return InjectionPlan(
            tasks_document=copied_tasks,
            adapter_config=copied_config,
            applications=(),
            injection_ids_by_task={},
            forced_errors={},
            public_document=None,
        )
    document = injection_document
    if isinstance(document, Mapping) and "injections" in document:
        items = document.get("injections")
    else:
        items = document
    if not isinstance(items, list):
        raise PipelineConfigError("injection config must be a list or {'injections': [...]}")

    raw_tasks = _task_list(copied_tasks)
    names = sorted(
        name for name in (_task_name(task) for task in raw_tasks) if name is not None
    )
    applications: list[dict[str, Any]] = []
    injection_ids: dict[str, list[str]] = {name: [] for name in names}
    overrides: dict[str, dict[str, Any]] = {}
    forced_errors: dict[str, list[str]] = {name: [] for name in names}
    seen_ids: set[str] = set()

    for index, raw_item in enumerate(items):
        if not isinstance(raw_item, Mapping):
            raise PipelineConfigError(f"injections[{index}] must be an object")
        injection_id = raw_item.get("id", f"INJ-{index + 1:03d}")
        if not isinstance(injection_id, str) or not injection_id:
            raise PipelineConfigError(f"injections[{index}].id must be a non-empty string")
        if injection_id in seen_ids:
            raise PipelineConfigError(f"duplicate injection ID {injection_id!r}")
        seen_ids.add(injection_id)
        injection_type = raw_item.get("type")
        if not isinstance(injection_type, str) or not injection_type:
            raise PipelineConfigError(f"injection {injection_id!r}.type must be a string")
        target_raw = raw_item.get("task", raw_item.get("task_name", raw_item.get("target")))
        if not isinstance(target_raw, str) or not target_raw:
            raise PipelineConfigError(f"injection {injection_id!r}.task must be a string")
        targets = names if target_raw == "*" else [target_raw]
        for target in targets:
            base_application = {
                "id": injection_id,
                "type": injection_type,
                "target": target,
                "applied": False,
                "reason": "",
            }
            if target not in injection_ids:
                base_application["reason"] = "target task does not exist"
                applications.append(base_application)
                continue
            task = next(item for item in raw_tasks if _task_name(item) == target)
            if not isinstance(task, Mapping):
                base_application["reason"] = "target task is not an object"
                applications.append(base_application)
                continue
            try:
                if injection_type in {"missing_sample", "missing_sample_id"}:
                    sample_id = raw_item.get("sample_id", raw_item.get("value", "missing-sample"))
                    if not isinstance(sample_id, str) or not sample_id:
                        raise ValueError("missing_sample requires a non-empty sample_id")
                    key = "train_ids" if "train_ids" in task else "training_sample_ids"
                    if not isinstance(task.get(key), list):
                        task[key] = []
                    task[key].append(sample_id)
                elif injection_type in {"illegal_classes", "invalid_classes"}:
                    task["classes"] = copy.deepcopy(raw_item.get("classes", []))
                elif injection_type == "zero_rank":
                    overrides.setdefault(target, {})["rank"] = 0
                elif injection_type == "negative_lr":
                    value = raw_item.get("learning_rate", raw_item.get("value", -0.1))
                    overrides.setdefault(target, {})["learning_rate"] = value
                    overrides[target].pop("lr", None)
                elif injection_type in {"train_error", "evaluate_error", "eval_error"}:
                    stage = "train" if injection_type == "train_error" else "evaluate"
                    forced_errors[target].append(stage)
                else:
                    raise ValueError(f"unknown injection type {injection_type!r}")
            except ValueError as exc:
                base_application["reason"] = str(exc)
                applications.append(base_application)
                continue
            base_application["applied"] = True
            applications.append(base_application)
            injection_ids[target].append(injection_id)

    if overrides:
        copied_config = copied_config.with_overrides(overrides)
    return InjectionPlan(
        tasks_document=copied_tasks,
        adapter_config=copied_config,
        applications=tuple(applications),
        injection_ids_by_task={
            name: tuple(sorted(set(values))) for name, values in sorted(injection_ids.items())
        },
        forced_errors={
            name: tuple(sorted(set(values))) for name, values in sorted(forced_errors.items())
        },
        public_document=copy.deepcopy(injection_document),
    )
