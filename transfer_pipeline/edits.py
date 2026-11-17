"""Declarative edit operations used by tests, verification, and tools."""

from __future__ import annotations

import copy
from typing import Any, Mapping


def _tasks(document: Any) -> list[Any]:
    if isinstance(document, list):
        return document
    if not isinstance(document, dict):
        raise ValueError("task document must be an object or list")
    value = document.get("tasks")
    if not isinstance(value, list):
        raise ValueError("task document.tasks must be a list")
    return value


def _task(document: Any, name: str) -> dict[str, Any]:
    for item in _tasks(document):
        if isinstance(item, dict) and item.get("name") == name:
            return item
    raise ValueError(f"task {name!r} not found")


def apply_edit(documents: Mapping[str, Any], edit: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one edit in-place to a mutable document set and return its metadata."""

    operation = edit.get("op")
    tasks_document = documents["tasks"]
    features_document = documents["features"]
    config_document = documents["config"]
    if operation == "add_task":
        task = copy.deepcopy(edit.get("task"))
        if not isinstance(task, dict):
            raise ValueError("add_task requires a task object")
        _tasks(tasks_document).append(task)
        return {"op": operation, "task_name": task.get("name")}
    if operation == "delete_task":
        name = edit.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("delete_task requires a task name")
        raw_tasks = _tasks(tasks_document)
        before = len(raw_tasks)
        raw_tasks[:] = [
            item
            for item in raw_tasks
            if not (isinstance(item, dict) and item.get("name") == name)
        ]
        if len(raw_tasks) == before:
            raise ValueError(f"task {name!r} not found")
        return {"op": operation, "task_name": name}
    if operation == "set_classes":
        name = edit.get("name")
        classes = edit.get("classes")
        if not isinstance(name, str) or not isinstance(classes, list):
            raise ValueError("set_classes requires name and classes list")
        _task(tasks_document, name)["classes"] = copy.deepcopy(classes)
        return {"op": operation, "task_name": name}
    if operation == "replace_sample":
        sample_id = edit.get("sample_id")
        if not isinstance(sample_id, str) or not isinstance(features_document, dict):
            raise ValueError("replace_sample requires sample_id")
        samples = features_document.get("samples")
        if not isinstance(samples, dict) or sample_id not in samples:
            raise ValueError(f"sample {sample_id!r} not found")
        if "features" not in edit:
            raise ValueError("replace_sample requires features")
        samples[sample_id]["features"] = copy.deepcopy(edit["features"])
        if "label" in edit:
            samples[sample_id]["label"] = edit["label"]
        return {"op": operation, "sample_id": sample_id}
    if operation == "set_adapter_params":
        name = edit.get("name")
        changes = edit.get("changes")
        if not isinstance(name, str) or not isinstance(changes, dict):
            raise ValueError("set_adapter_params requires name and changes")
        if name == "*":
            config_document.update(copy.deepcopy(changes))
        else:
            overrides = config_document.setdefault("task_overrides", {})
            overrides.setdefault(name, {}).update(copy.deepcopy(changes))
        return {"op": operation, "task_name": name}
    raise ValueError(f"unsupported edit op {operation!r}")
