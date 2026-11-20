"""Batch edit-plan loading and structural validation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .errors import E_PLAN_INVALID, WorkbenchError, diagnostic
from .util import read_json, sha256_obj


APPEND_OPS = {"append_row", "row_append"}
CORRECT_OPS = {"correct_row", "update_row"}
IMPUTER_OPS = {"change_imputer", "imputer_change"}
GROUP_OPS = {"change_group_mapping", "group_mapping_change"}


def load_plan(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkbenchError(E_PLAN_INVALID, f"cannot load edit plan: {exc}") from exc
    failures: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        raise WorkbenchError(E_PLAN_INVALID, "edit plan root must be an object")
    batches = raw.get("batches")
    if not isinstance(batches, list):
        raise WorkbenchError(E_PLAN_INVALID, "edit plan batches must be a list")
    normalized_batches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            failures.append(
                diagnostic(E_PLAN_INVALID, "batch must be an object", field=f"batches[{index}]", stage="plan")
            )
            continue
        batch_id = str(batch.get("batch_id", index + 1))
        if batch_id in seen_ids:
            failures.append(
                diagnostic(E_PLAN_INVALID, "duplicate batch_id", field=f"batches[{index}].batch_id", stage="plan")
            )
        seen_ids.add(batch_id)
        operations = batch.get("operations", batch.get("ops"))
        if not isinstance(operations, list) or not operations:
            failures.append(
                diagnostic(
                    E_PLAN_INVALID,
                    "batch must contain a non-empty operations list",
                    field=f"batches[{index}].operations",
                    stage="plan",
                )
            )
            continue
        normalized_operations: list[dict[str, Any]] = []
        for op_index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                failures.append(
                    diagnostic(
                        E_PLAN_INVALID,
                        "operation must be an object",
                        field=f"batches[{index}].operations[{op_index}]",
                        stage="plan",
                    )
                )
                continue
            op = operation.get("op", operation.get("kind"))
            if op in APPEND_OPS:
                if not isinstance(operation.get("row"), dict):
                    failures.append(
                        diagnostic(
                            E_PLAN_INVALID,
                            "append_row requires a row object",
                            field=f"batches[{index}].operations[{op_index}].row",
                            stage="plan",
                        )
                    )
                    continue
            elif op in CORRECT_OPS:
                if not isinstance(operation.get("row_id"), (str, int)) or not isinstance(
                    operation.get("changes"), dict
                ):
                    failures.append(
                        diagnostic(
                            E_PLAN_INVALID,
                            "correct_row requires row_id and changes",
                            field=f"batches[{index}].operations[{op_index}]",
                            stage="plan",
                        )
                    )
                    continue
            elif op in IMPUTER_OPS:
                if not isinstance(operation.get("column"), str) or not isinstance(
                    operation.get("changes"), dict
                ):
                    failures.append(
                        diagnostic(
                            E_PLAN_INVALID,
                            "change_imputer requires column and changes",
                            field=f"batches[{index}].operations[{op_index}]",
                            stage="plan",
                        )
                    )
                    continue
            elif op in GROUP_OPS:
                if not isinstance(operation.get("key_column"), str) or not isinstance(
                    operation.get("changes"), dict
                ):
                    failures.append(
                        diagnostic(
                            E_PLAN_INVALID,
                            "change_group_mapping requires key_column and changes",
                            field=f"batches[{index}].operations[{op_index}]",
                            stage="plan",
                        )
                    )
                    continue
            else:
                failures.append(
                    diagnostic(
                        E_PLAN_INVALID,
                        f"unknown operation {op!r}",
                        field=f"batches[{index}].operations[{op_index}].op",
                        stage="plan",
                    )
                )
                continue
            normalized_operations.append(deepcopy(operation))
        normalized_batches.append(
            {
                "batch_id": batch_id,
                "operations": normalized_operations,
                "index": index,
            }
        )
    normalized = {
        "format": raw.get("format", 1),
        "batches": normalized_batches,
    }
    return normalized, failures


def plan_hash(plan: dict[str, Any]) -> str:
    material = {
        "format": plan.get("format", 1),
        "batches": [
            {"batch_id": batch["batch_id"], "operations": batch["operations"]}
            for batch in plan["batches"]
        ],
    }
    return sha256_obj(material)

