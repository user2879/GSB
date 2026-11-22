from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import deep_set, fingerprint, sorted_failures
from .loader import SampleStore


@dataclass
class EditOperation:
    index: int
    operation_type: str
    payload: dict[str, Any]


@dataclass
class EditBatch:
    batch_id: str
    operations: list[EditOperation]


@dataclass
class EditPlan:
    batches: list[EditBatch]
    failures: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""

    def batch(self, batch_id: str) -> EditBatch | None:
        for batch in self.batches:
            if batch.batch_id == batch_id:
                return batch
        return None


def load_edit_plan(path_text: str | Path) -> EditPlan:
    path = Path(path_text)
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return EditPlan(
            batches=[],
            failures=[
                {
                    "scope": "plan",
                    "codes": ["E_PLAN_PARSE"],
                    "message": str(exc),
                }
            ],
            fingerprint=fingerprint("edit-plan-file:v1", path_text),
        )

    failures: list[dict[str, Any]] = []
    batches: list[EditBatch] = []
    if not isinstance(document, dict):
        failures.append(
            {
                "scope": "plan",
                "codes": ["E_PLAN_OBJECT"],
                "message": "edit plan must be a JSON object",
            }
        )
        document = {}
    raw_batches = document.get("batches", [])
    if not isinstance(raw_batches, list):
        failures.append(
            {
                "scope": "plan",
                "path": "batches",
                "codes": ["E_PLAN_BATCHES"],
                "message": "batches must be a list",
            }
        )
        raw_batches = []

    seen_batch_ids: set[str] = set()
    for batch_index, raw_batch in enumerate(raw_batches):
        if not isinstance(raw_batch, dict):
            failures.append(
                {
                    "scope": "plan",
                    "path": f"batches.{batch_index}",
                    "codes": ["E_PLAN_BATCH_OBJECT"],
                    "message": "batch must be an object",
                }
            )
            continue
        batch_id = raw_batch.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            batch_id = f"batch-{batch_index + 1:04d}"
            failures.append(
                {
                    "scope": "plan",
                    "path": f"batches.{batch_index}.batch_id",
                    "batch_id": batch_id,
                    "codes": ["E_PLAN_BATCH_ID"],
                    "message": "batch_id is missing or invalid; a deterministic id is used",
                }
            )
        if batch_id in seen_batch_ids:
            failures.append(
                {
                    "scope": "plan",
                    "path": f"batches.{batch_index}.batch_id",
                    "batch_id": batch_id,
                    "codes": ["E_PLAN_BATCH_DUPLICATE"],
                    "message": "duplicate batch_id is ignored",
                }
            )
            continue
        seen_batch_ids.add(batch_id)
        raw_operations = raw_batch.get("operations", [])
        if not isinstance(raw_operations, list):
            failures.append(
                {
                    "scope": "plan",
                    "path": f"batches.{batch_index}.operations",
                    "batch_id": batch_id,
                    "codes": ["E_PLAN_OPERATIONS"],
                    "message": "operations must be a list",
                }
            )
            raw_operations = []
        operations: list[EditOperation] = []
        for operation_index, raw_operation in enumerate(raw_operations):
            if not isinstance(raw_operation, dict):
                failures.append(
                    {
                        "scope": "plan",
                        "path": f"batches.{batch_index}.operations.{operation_index}",
                        "batch_id": batch_id,
                        "operation_index": operation_index,
                        "codes": ["E_EDIT_OBJECT"],
                        "message": "operation must be an object",
                    }
                )
                continue
            operation_type = raw_operation.get("type")
            if not isinstance(operation_type, str) or not operation_type:
                failures.append(
                    {
                        "scope": "plan",
                        "path": f"batches.{batch_index}.operations.{operation_index}.type",
                        "batch_id": batch_id,
                        "operation_index": operation_index,
                        "codes": ["E_EDIT_TYPE"],
                        "message": "operation type must be a non-empty string",
                    }
                )
                continue
            operations.append(
                EditOperation(
                    index=operation_index,
                    operation_type=operation_type,
                    payload=raw_operation,
                )
            )
        batches.append(EditBatch(batch_id=batch_id, operations=operations))

    return EditPlan(
        batches=batches,
        failures=sorted_failures(failures),
        fingerprint=fingerprint("edit-plan:v1", document),
    )


def _operation_failure(
    batch_id: str,
    operation_index: int,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "scope": "edit",
        "batch_id": batch_id,
        "operation_index": operation_index,
        "codes": [code],
        "message": message,
    }


def _normalized_patch(
    payload: dict[str, Any], default_prefix: str
) -> dict[str, Any]:
    patch = payload.get("patch")
    if not isinstance(patch, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, value in patch.items():
        if not isinstance(key, str):
            raise ValueError("patch paths must be strings")
        path = key if "." in key else f"{default_prefix}.{key}"
        normalized[path] = value
    return normalized


def apply_batch(
    batch: EditBatch,
    *,
    sample_store: SampleStore,
    raw_config: dict[str, Any],
    config_id_field: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    updated_config = copy.deepcopy(raw_config)

    for operation in sorted(batch.operations, key=lambda item: item.index):
        payload = operation.payload
        op_type = operation.operation_type
        if op_type == "append_samples":
            records = payload.get("samples")
            if not isinstance(records, list):
                failures.append(
                    _operation_failure(
                        batch.batch_id,
                        operation.index,
                        "E_EDIT_APPEND",
                        "append_samples.samples must be a list",
                    )
                )
                continue
            acceptable: list[dict[str, Any]] = []
            for record_index, record in enumerate(records):
                if isinstance(record, dict):
                    acceptable.append(record)
                else:
                    failures.append(
                        _operation_failure(
                            batch.batch_id,
                            operation.index,
                            "E_EDIT_SAMPLE_OBJECT",
                            f"appended sample {record_index} must be an object",
                        )
                    )
            sample_store.append_records(acceptable)
        elif op_type == "correct_sample":
            sample_id = payload.get("sample_id")
            patch = payload.get("patch")
            if not isinstance(sample_id, str) or not sample_id:
                failures.append(
                    _operation_failure(
                        batch.batch_id,
                        operation.index,
                        "E_EDIT_SAMPLE_ID",
                        "correct_sample.sample_id must be a non-empty string",
                    )
                )
                continue
            if not isinstance(patch, dict):
                failures.append(
                    _operation_failure(
                        batch.batch_id,
                        operation.index,
                        "E_EDIT_PATCH",
                        "correct_sample.patch must be an object",
                    )
                )
                continue
            ok, code = sample_store.correct_record(
                sample_id, patch, id_field=config_id_field
            )
            if not ok:
                failures.append(
                    _operation_failure(
                        batch.batch_id,
                        operation.index,
                        code or "E_EDIT_SAMPLE",
                        f"cannot correct sample {sample_id!r}",
                    )
                )
        elif op_type in (
            "update_feature_params",
            "update_model_hyperparams",
            "update_quantization",
            "update_config",
        ):
            prefixes = {
                "update_feature_params": "features",
                "update_model_hyperparams": "training",
                "update_quantization": "quantization",
                "update_config": "",
            }
            try:
                patch = _normalized_patch(payload, prefixes[op_type])
            except ValueError as exc:
                failures.append(
                    _operation_failure(
                        batch.batch_id,
                        operation.index,
                        "E_EDIT_CONFIG_PATH",
                        str(exc),
                    )
                )
                continue
            if op_type == "update_config":
                direct_patch = payload.get("patch")
                if not isinstance(direct_patch, dict):
                    failures.append(
                        _operation_failure(
                            batch.batch_id,
                            operation.index,
                            "E_EDIT_PATCH",
                            "update_config.patch must be an object",
                        )
                    )
                    continue
                patch = {str(key): value for key, value in direct_patch.items()}
            if not patch:
                failures.append(
                    _operation_failure(
                        batch.batch_id,
                        operation.index,
                        "E_EDIT_PATCH",
                        "patch must be a non-empty object",
                    )
                )
                continue
            candidate_config = copy.deepcopy(updated_config)
            try:
                for path, value in sorted(patch.items()):
                    deep_set(candidate_config, path, value)
            except (KeyError, IndexError, TypeError, ValueError):
                failures.append(
                    _operation_failure(
                        batch.batch_id,
                        operation.index,
                        "E_EDIT_CONFIG_PATH",
                        f"invalid configuration path in {sorted(patch)}",
                    )
                )
                continue
            updated_config = candidate_config
        else:
            failures.append(
                _operation_failure(
                    batch.batch_id,
                    operation.index,
                    "E_EDIT_TYPE",
                    f"unsupported operation type: {op_type}",
                )
            )

    return updated_config, sorted_failures(failures)
