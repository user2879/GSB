"""Edit-plan and fault-injection manifest loading with business validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import FingerprintConfig, config_from_overrides
from .errors import E_PLAN_INVALID, E_TEXT_INVALID, Failure
from .text_store import TextRecord, validate_text_payload


EDITABLE_CONFIG_FIELDS = {
    "window_count",
    "signature_dimension",
    "similarity_threshold",
    "bucket_bits",
    "fixed_decimal_places",
    "quantization_levels",
    "signature_batch_budget",
    "max_workers",
    "snapshot_commit_interval",
    "source_labels",
    "vocabulary",
    "hash_seed",
}


@dataclass(frozen=True)
class EditOperation:
    batch: int
    index: int
    kind: str
    text_id: str | None = None
    text: TextRecord | None = None
    config_key: str | None = None
    config_value: Any = None


def _plan_failure(message: str, *, source: str, index: int | None = None, item_id: str | None = None) -> Failure:
    detail = {"operation_index": index} if index is not None else None
    return Failure(E_PLAN_INVALID, "plan_validation", message, item_id=item_id, source=source, detail=detail)


def _json_no_constants(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def load_edit_plan(
    path: str,
    config: FingerprintConfig,
) -> tuple[dict[int, list[EditOperation]], list[Failure]]:
    failures: list[Failure] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=_json_no_constants)
    except Exception as exc:
        return {}, [_plan_failure(f"cannot load edit plan: {exc}", source=path)]
    if not isinstance(payload, dict):
        return {}, [_plan_failure("edit plan must be an object", source=path)]
    if payload.get("schema_version") != 1:
        failures.append(_plan_failure("schema_version must be integer 1", source=path))
    batches_raw = payload.get("batches")
    if not isinstance(batches_raw, list) or not batches_raw:
        failures.append(_plan_failure("batches must be a non-empty list", source=path))
        return {}, failures

    result: dict[int, list[EditOperation]] = {}
    seen_batches: set[int] = set()
    global_index = 0
    for batch_pos, batch_raw in enumerate(batches_raw):
        if not isinstance(batch_raw, dict):
            failures.append(_plan_failure(f"batch {batch_pos} must be an object", source=path, index=global_index))
            continue
        batch_number = batch_raw.get("batch")
        if not isinstance(batch_number, int) or isinstance(batch_number, bool) or batch_number <= 0:
            failures.append(
                _plan_failure("batch number must be a positive integer", source=path, index=global_index)
            )
            continue
        if batch_number in seen_batches:
            failures.append(
                _plan_failure("duplicate batch number", source=path, index=global_index, item_id=str(batch_number))
            )
            continue
        seen_batches.add(batch_number)
        operations_raw = batch_raw.get("operations")
        if not isinstance(operations_raw, list):
            failures.append(
                _plan_failure("operations must be a list", source=path, index=global_index, item_id=str(batch_number))
            )
            continue
        operations: list[EditOperation] = []
        for op_pos, op_raw in enumerate(operations_raw):
            index = global_index
            global_index += 1
            if not isinstance(op_raw, dict):
                failures.append(_plan_failure("operation must be an object", source=path, index=index))
                continue
            kind = op_raw.get("op")
            if kind in {"append", "replace"}:
                text_payload = op_raw.get("text")
                if not isinstance(text_payload, dict):
                    failures.append(
                        _plan_failure(f"{kind} requires an object field 'text'", source=path, index=index)
                    )
                    continue
                text_record, text_failures = validate_text_payload(text_payload, config, source=path)
                for item in text_failures:
                    if item.detail is None:
                        item = Failure(item.code, item.stage, item.message, item.item_id, item.source, item.line, {"operation_index": index})
                    else:
                        item = Failure(item.code, item.stage, item.message, item.item_id, item.source, item.line, {**item.detail, "operation_index": index})
                    failures.append(item)
                if kind == "replace":
                    text_id = op_raw.get("text_id")
                    if not isinstance(text_id, str) or not text_id:
                        failures.append(
                            _plan_failure("replace requires a non-empty text_id", source=path, index=index)
                        )
                        continue
                    if text_record is not None and text_record.text_id != text_id:
                        failures.append(
                            _plan_failure(
                                "replacement text_id must match text.text_id",
                                source=path,
                                index=index,
                                item_id=text_id,
                            )
                        )
                        continue
                elif text_record is not None:
                    text_id = text_record.text_id
                else:
                    text_id = text_payload.get("text_id") if isinstance(text_payload.get("text_id"), str) else None
                operations.append(
                    EditOperation(batch_number, index, kind, text_id=text_id, text=text_record)
                )
            elif kind == "delete":
                text_id = op_raw.get("text_id")
                if not isinstance(text_id, str) or not text_id:
                    failures.append(
                        _plan_failure("delete requires a non-empty text_id", source=path, index=index)
                    )
                    continue
                operations.append(EditOperation(batch_number, index, kind, text_id=text_id))
            elif kind == "set_config":
                key = op_raw.get("key")
                if key not in EDITABLE_CONFIG_FIELDS:
                    failures.append(
                        _plan_failure("set_config key is not editable", source=path, index=index, item_id=str(key))
                    )
                    continue
                if "value" not in op_raw:
                    failures.append(
                        _plan_failure("set_config requires a value", source=path, index=index, item_id=key)
                    )
                    continue
                operations.append(
                    EditOperation(batch_number, index, kind, config_key=key, config_value=op_raw["value"])
                )
            else:
                failures.append(
                    _plan_failure("unknown operation kind", source=path, index=index, item_id=str(kind))
                )
        result[batch_number] = operations
    return dict(sorted(result.items())), failures


def apply_edit_batch(
    records: dict[str, TextRecord],
    config: FingerprintConfig,
    operations: list[EditOperation],
    *,
    source: str = "edit_plan",
) -> tuple[dict[str, TextRecord], FingerprintConfig, list[Failure]]:
    updated_records = dict(records)
    overrides: list[tuple[str, Any]] = []
    failures: list[Failure] = []
    for operation in operations:
        if operation.kind == "append":
            if operation.text_id is None:
                failures.append(_plan_failure("append text is invalid", source=source, index=operation.index))
                continue
            if operation.text_id in updated_records:
                failures.append(
                    _plan_failure("append text_id already exists", source=source, index=operation.index, item_id=operation.text_id)
                )
                continue
            if operation.text is None:
                continue
            updated_records[operation.text_id] = operation.text
        elif operation.kind == "replace":
            if operation.text_id is None or operation.text_id not in updated_records:
                failures.append(
                    _plan_failure("replace target does not exist", source=source, index=operation.index, item_id=operation.text_id)
                )
                continue
            if operation.text is None:
                continue
            updated_records[operation.text_id] = operation.text
        elif operation.kind == "delete":
            if operation.text_id is None or operation.text_id not in updated_records:
                failures.append(
                    _plan_failure("delete target does not exist", source=source, index=operation.index, item_id=operation.text_id)
                )
                continue
            updated_records.pop(operation.text_id, None)
        elif operation.kind == "set_config":
            overrides.append((operation.config_key or "", operation.config_value))

    final_config = config
    if overrides:
        final_config, config_failures = config_from_overrides(config, overrides, source=source)
        failures.extend(config_failures)
        if final_config is None:
            final_config = config
    return updated_records, final_config, failures


@dataclass(frozen=True)
class FaultInjection:
    injection_id: str
    kind: str
    payload: dict[str, Any]


def load_injections(path: str) -> tuple[list[FaultInjection], list[Failure]]:
    failures: list[Failure] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=_json_no_constants)
    except Exception as exc:
        return [], [_plan_failure(f"cannot load injection manifest: {exc}", source=path)]
    if not isinstance(payload, dict) or not isinstance(payload.get("injections"), list):
        return [], [_plan_failure("injection manifest must contain an injections list", source=path)]
    supported = {
        "zero_signature_dimension",
        "non_finite_similarity",
        "bucket_bits_out_of_range",
        "cluster_id_conflict",
    }
    result: list[FaultInjection] = []
    seen: set[str] = set()
    for index, item in enumerate(payload["injections"]):
        if not isinstance(item, dict):
            failures.append(_plan_failure("injection item must be an object", source=path, index=index))
            continue
        injection_id = item.get("injection_id")
        kind = item.get("kind")
        if not isinstance(injection_id, str) or not injection_id:
            failures.append(_plan_failure("injection_id must be a non-empty string", source=path, index=index))
            continue
        if injection_id in seen:
            failures.append(_plan_failure("duplicate injection_id", source=path, index=index, item_id=injection_id))
            continue
        seen.add(injection_id)
        if kind not in supported:
            failures.append(_plan_failure("unsupported injection kind", source=path, index=index, item_id=injection_id))
            continue
        result.append(FaultInjection(injection_id, kind, dict(item)))
    return result, failures
