from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import ConfigError, merge_config, normalize_config
from .loader import _failure, _sort_failures


class EditError(ValueError):
    pass


def load_edit_batches(path: str | Path) -> list[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EditError(f"cannot load edit stream: {exc}") from exc
    if isinstance(value, list):
        batches = value
    elif isinstance(value, dict):
        batches = value.get("batches", [])
    else:
        raise EditError("edit stream must be an object or array")
    if not isinstance(batches, list):
        raise EditError("batches must be an array")
    normalized: list[dict[str, Any]] = []
    for index, batch in enumerate(batches, start=1):
        if not isinstance(batch, dict):
            raise EditError(f"batch {index} must be an object")
        operations = batch.get("operations", [])
        if not isinstance(operations, list):
            raise EditError(f"batch {index} operations must be an array")
        normalized.append(
            {
                "batch_id": str(batch.get("batch_id", f"batch-{index}")),
                "operations": operations,
                **(
                    {"config_patch": batch["config_patch"]}
                    if "config_patch" in batch
                    else {}
                ),
                **(
                    {"set_compression_window": batch["set_compression_window"]}
                    if "set_compression_window" in batch
                    else {}
                ),
            }
        )
    return normalized


def apply_batch(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    batch: dict[str, Any],
    order_counter: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], int]:
    result_records = deepcopy(records)
    result_config = deepcopy(config)
    failures: list[dict[str, Any]] = []
    next_order = order_counter

    operations = batch.get("operations", [])
    for operation in operations:
        if not isinstance(operation, dict):
            failures.append(
                _failure(
                    stage="load",
                    code="E_BAD_EVENT",
                    message="edit operation must be an object",
                )
            )
            continue
        op = operation.get("op")
        session_id = operation.get("session_id", operation.get("session"))
        if not isinstance(session_id, str) or not session_id:
            failures.append(
                _failure(
                    stage="load",
                    code="E_EMPTY_SESSION",
                    message="edit operation requires a non-empty session_id",
                )
            )
            continue
        session_records = [
            item for item in result_records if item.get("session_id") == session_id
        ]
        if op == "delete":
            turn = _operation_turn(operation)
            if turn is None:
                failures.append(_edit_target_failure(session_id, None, "delete"))
                continue
            target = [item for item in session_records if item.get("turn") == turn]
            if not target:
                failures.append(_edit_target_failure(session_id, turn, "delete"))
                continue
            result_records = [
                item
                for item in result_records
                if not (
                    item.get("session_id") == session_id and item.get("turn") == turn
                )
            ]
            for item in result_records:
                if item.get("session_id") == session_id and item.get("turn", 0) > turn:
                    item["turn"] -= 1
        elif op == "replace":
            turn = _operation_turn(operation)
            target = [
                item
                for item in session_records
                if item.get("turn") == turn
            ] if turn is not None else []
            if turn is None or not target:
                failures.append(_edit_target_failure(session_id, turn, "replace"))
                continue
            item = target[0]
            for key in ("speaker", "text", "timestamp"):
                if key in operation:
                    item[key] = operation[key]
        elif op == "add":
            requested_turn = _operation_turn(operation)
            if requested_turn is None:
                failures.append(_edit_target_failure(session_id, None, "add"))
                continue
            maximum = len(session_records)
            insert_turn = min(requested_turn, maximum + 1)
            for item in result_records:
                if item.get("session_id") == session_id and item.get("turn", 0) >= insert_turn:
                    item["turn"] += 1
            next_order += 1
            result_records.append(
                {
                    "session_id": session_id,
                    "turn": insert_turn,
                    "speaker": operation.get("speaker", "system"),
                    "text": operation.get("text", ""),
                    "timestamp": operation.get(
                        "timestamp", "2026-11-18T00:00:00+00:00"
                    ),
                    "_line": None,
                    "_order": next_order,
                }
            )
        else:
            failures.append(
                _failure(
                    stage="load",
                    code="E_BAD_EVENT",
                    message=f"unknown edit operation: {op}",
                    session_id=session_id,
                )
            )

    if "set_compression_window" in batch:
        patch = {"compression_window": batch["set_compression_window"]}
        try:
            result_config = merge_config(result_config, patch)
        except ConfigError as exc:
            failures.append(
                _failure(
                    stage="load",
                    code="E_BAD_EVENT",
                    message=str(exc),
                )
            )
    if "config_patch" in batch:
        try:
            result_config = merge_config(result_config, batch["config_patch"])
        except ConfigError as exc:
            failures.append(
                _failure(
                    stage="load",
                    code="E_BAD_EVENT",
                    message=str(exc),
                )
            )

    result_records.sort(
        key=lambda item: (
            item.get("session_id", ""),
            item.get("turn", 0),
            item.get("_line") if item.get("_line") is not None else 10**12,
            item.get("_order", 0),
        )
    )
    for sequence, record in enumerate(result_records, start=1):
        record["_order"] = sequence
    next_order = max(next_order, len(result_records))
    return result_records, normalize_config(result_config), _sort_failures(failures), next_order


def _operation_turn(operation: dict[str, Any]) -> int | None:
    turn = operation.get("turn", operation.get("round"))
    if isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0:
        return None
    return turn


def _edit_target_failure(
    session_id: str, turn: int | None, operation: str
) -> dict[str, Any]:
    return _failure(
        stage="load",
        code="E_EDIT_TARGET",
        message=f"{operation} target does not exist",
        session_id=session_id,
        turn=turn,
    )
