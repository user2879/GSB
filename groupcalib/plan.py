"""Edit-plan loading and transactional application of table edits."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .config import CalibrationConfig
from .errors import E_CONFIG_INVALID, E_DATA_INVALID, E_GROUP_UNKNOWN, Failure, failure_sort_key
from .hashing import sha256_value
from .table import FeatureRow, FeatureTable, normalise_feature_value


SUPPORTED_ACTIONS = {"update", "append", "delete", "regroup", "repartition_columns"}


@dataclass(frozen=True)
class PlanBatch:
    batch_id: str
    actions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EditPlan:
    batches: tuple[PlanBatch, ...]
    plan_hash: str
    raw: dict[str, Any]


@dataclass
class PlanLoadResult:
    plan: EditPlan | None
    failures: list[Failure]


@dataclass
class BatchApplyResult:
    table: FeatureTable
    column_groups: dict[str, tuple[str, ...]]
    failures: list[Failure] = field(default_factory=list)
    affected_groups: set[str] = field(default_factory=set)
    applied_action_count: int = 0


def _plan_failure(
    message: str,
    *,
    field: str,
    batch_index: int = -1,
    action_index: int = -1,
    row_id: str = "",
) -> Failure:
    return Failure(
        code=E_CONFIG_INVALID,
        message=message,
        sort_key=failure_sort_key(
            source="plan",
            batch_index=batch_index,
            action_index=action_index,
            code=E_CONFIG_INVALID,
            row_id=row_id,
            field=field,
            message=message,
        ),
        details={
            "field": field,
            "tag": "plan",
            "batch_index": batch_index,
            "action_index": action_index,
        },
    )


def _data_failure(
    code: str,
    message: str,
    *,
    field: str,
    batch_index: int,
    action_index: int,
    batch_id: str,
    row_id: str = "",
) -> Failure:
    return Failure(
        code=code,
        message=message,
        sort_key=failure_sort_key(
            source="plan",
            batch_index=batch_index,
            action_index=action_index,
            code=code,
            row_id=row_id,
            field=field,
            message=message,
        ),
        details={
            "field": field,
            "tag": "action",
            "batch_index": batch_index,
            "batch_id": batch_id,
            "action_index": action_index,
            "row_id": row_id,
        },
    )


def load_plan(path: str | os.PathLike[str]) -> PlanLoadResult:
    failures: list[Failure] = []
    if not os.path.exists(path):
        failures.append(_plan_failure(f"edit plan does not exist: {path}", field="path"))
        return PlanLoadResult(None, failures)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(_plan_failure(f"cannot read edit plan JSON: {exc}", field="json"))
        return PlanLoadResult(None, failures)
    if not isinstance(raw, dict):
        failures.append(_plan_failure("edit plan root must be a JSON object", field="$"))
        return PlanLoadResult(None, failures)
    if raw.get("schema_version", 1) != 1:
        failures.append(_plan_failure("edit plan schema_version must be 1", field="schema_version"))
    batches_raw = raw.get("batches")
    if not isinstance(batches_raw, list):
        failures.append(_plan_failure("edit plan batches must be an array", field="batches"))
        return PlanLoadResult(None, failures)
    seen_batch_ids: set[str] = set()
    batches: list[PlanBatch] = []
    for batch_index, batch_raw in enumerate(batches_raw):
        if not isinstance(batch_raw, dict):
            failures.append(
                _plan_failure(
                    "batch must be an object", field=f"batches[{batch_index}]", batch_index=batch_index
                )
            )
            continue
        batch_id = batch_raw.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            failures.append(
                _plan_failure(
                    "batch_id must be a non-empty string",
                    field=f"batches[{batch_index}].batch_id",
                    batch_index=batch_index,
                )
            )
            continue
        if batch_id in seen_batch_ids:
            failures.append(
                _plan_failure(
                    f"duplicate batch_id {batch_id!r}",
                    field="batch_id",
                    batch_index=batch_index,
                )
            )
            continue
        seen_batch_ids.add(batch_id)
        actions_raw = batch_raw.get("actions", [])
        if not isinstance(actions_raw, list):
            failures.append(
                _plan_failure(
                    "batch actions must be an array",
                    field=f"batches[{batch_index}].actions",
                    batch_index=batch_index,
                )
            )
            continue
        actions: list[dict[str, Any]] = []
        for action_index, action_raw in enumerate(actions_raw):
            if not isinstance(action_raw, dict):
                failures.append(
                    _plan_failure(
                        "action must be an object",
                        field=f"batches[{batch_index}].actions[{action_index}]",
                        batch_index=batch_index,
                        action_index=action_index,
                    )
                )
                continue
            op = action_raw.get("op")
            if op not in SUPPORTED_ACTIONS:
                failures.append(
                    _plan_failure(
                        f"op must be one of {sorted(SUPPORTED_ACTIONS)}",
                        field="op",
                        batch_index=batch_index,
                        action_index=action_index,
                    )
                )
                continue
            actions.append(action_raw)
        batches.append(PlanBatch(batch_id=batch_id, actions=tuple(actions)))
    if failures:
        return PlanLoadResult(None, failures)
    return PlanLoadResult(EditPlan(tuple(batches), sha256_value(raw), raw), [])


def _identifier(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _replace_row(table: FeatureTable, index: int, row: FeatureRow) -> FeatureTable:
    rows = list(table.rows)
    rows[index] = row
    return FeatureTable(rows, table.columns)


def apply_batch(
    table: FeatureTable,
    column_groups: dict[str, tuple[str, ...]],
    config: CalibrationConfig,
    batch: PlanBatch,
    batch_index: int,
) -> BatchApplyResult:
    current_table = table
    current_groups = {group_id: tuple(columns) for group_id, columns in column_groups.items()}
    failures: list[Failure] = []
    affected: set[str] = set()
    applied_count = 0

    for action_index, action in enumerate(batch.actions):
        op = action.get("op")

        if op == "update":
            row_id = _identifier(action.get("row_id"))
            if row_id is None:
                failures.append(
                    _data_failure(
                        E_DATA_INVALID,
                        "update.row_id must be a non-empty string or integer",
                        field="row_id",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                    )
                )
                continue
            row_map = current_table.row_map()
            row = row_map.get(row_id)
            if row is None:
                failures.append(
                    _data_failure(
                        E_DATA_INVALID,
                        f"update targets unknown row_id {row_id!r}",
                        field="row_id",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                        row_id=row_id,
                    )
                )
                continue
            values_raw = action.get("values", {})
            if not isinstance(values_raw, dict):
                failures.append(
                    _data_failure(
                        E_DATA_INVALID,
                        "update.values must be an object",
                        field="values",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                        row_id=row_id,
                    )
                )
                continue
            new_values = dict(row.values)
            invalid = False
            for column in sorted(values_raw):
                field = f"values.{column}"
                if column not in current_table.columns:
                    failures.append(
                        _data_failure(
                            E_DATA_INVALID,
                            f"update targets unknown column {column!r}",
                            field=field,
                            batch_index=batch_index,
                            action_index=action_index,
                            batch_id=batch.batch_id,
                            row_id=row_id,
                        )
                    )
                    invalid = True
                    break
                valid, number, value_error = normalise_feature_value(
                    values_raw[column], config.missing_value_markers
                )
                if not valid:
                    failures.append(
                        _data_failure(
                            E_DATA_INVALID,
                            f"update column {column!r}: {value_error}",
                            field=field,
                            batch_index=batch_index,
                            action_index=action_index,
                            batch_id=batch.batch_id,
                            row_id=row_id,
                        )
                    )
                    invalid = True
                    break
                new_values[column] = number
            if invalid:
                continue

            new_group_id = row.group_id
            if "group_id" in action:
                parsed_group = _identifier(action.get("group_id"))
                if parsed_group is None:
                    failures.append(
                        _data_failure(
                            E_DATA_INVALID,
                            "update.group_id must be a non-empty string or integer",
                            field="group_id",
                            batch_index=batch_index,
                            action_index=action_index,
                            batch_id=batch.batch_id,
                            row_id=row_id,
                        )
                    )
                    continue
                if parsed_group not in current_groups:
                    failures.append(
                        _data_failure(
                            E_GROUP_UNKNOWN,
                            f"update targets unknown group_id {parsed_group!r}",
                            field="group_id",
                            batch_index=batch_index,
                            action_index=action_index,
                            batch_id=batch.batch_id,
                            row_id=row_id,
                        )
                    )
                    continue
                new_group_id = parsed_group

            index = next(i for i, candidate in enumerate(current_table.rows) if candidate.row_id == row_id)
            updated = FeatureRow(
                order=row.order,
                source_line=row.source_line,
                row_id=row.row_id,
                group_id=new_group_id,
                values=new_values,
            )
            current_table = _replace_row(current_table, index, updated)
            affected.add(row.group_id)
            affected.add(new_group_id)
            applied_count += 1
            continue

        if op == "append":
            row_raw = action.get("row")
            if not isinstance(row_raw, dict):
                failures.append(
                    _data_failure(
                        E_DATA_INVALID,
                        "append.row must be an object",
                        field="row",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                    )
                )
                continue
            row_id = _identifier(row_raw.get(config.row_id_field))
            group_id = _identifier(row_raw.get(config.group_id_field))
            if row_id is None or group_id is None:
                failures.append(
                    _data_failure(
                        E_DATA_INVALID,
                        "append row requires valid row_id and group_id",
                        field=config.row_id_field,
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                    )
                )
                continue
            if row_id in current_table.row_map():
                failures.append(
                    _data_failure(
                        E_DATA_INVALID,
                        f"append duplicates row_id {row_id!r}",
                        field=config.row_id_field,
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                        row_id=row_id,
                    )
                )
                continue
            if group_id not in current_groups:
                failures.append(
                    _data_failure(
                        E_GROUP_UNKNOWN,
                        f"append targets unknown group_id {group_id!r}",
                        field=config.group_id_field,
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                        row_id=row_id,
                    )
                )
                continue
            values: dict[str, float | None] = {}
            invalid = False
            for column in sorted(row_raw):
                if column in {config.row_id_field, config.group_id_field}:
                    continue
                if column not in current_table.columns:
                    failures.append(
                        _data_failure(
                            E_DATA_INVALID,
                            f"append contains unknown column {column!r}",
                            field=column,
                            batch_index=batch_index,
                            action_index=action_index,
                            batch_id=batch.batch_id,
                            row_id=row_id,
                        )
                    )
                    invalid = True
                    break
                valid, number, value_error = normalise_feature_value(
                    row_raw[column], config.missing_value_markers
                )
                if not valid:
                    failures.append(
                        _data_failure(
                            E_DATA_INVALID,
                            f"append column {column!r}: {value_error}",
                            field=column,
                            batch_index=batch_index,
                            action_index=action_index,
                            batch_id=batch.batch_id,
                            row_id=row_id,
                        )
                    )
                    invalid = True
                    break
                values[column] = number
            if invalid:
                continue
            for column in current_table.columns:
                values.setdefault(column, None)
            rows = list(current_table.rows)
            rows.append(
                FeatureRow(
                    order=len(rows),
                    source_line=-1,
                    row_id=row_id,
                    group_id=group_id,
                    values=values,
                )
            )
            current_table = FeatureTable(rows, current_table.columns)
            affected.add(group_id)
            applied_count += 1
            continue

        if op == "delete":
            row_id = _identifier(action.get("row_id"))
            if row_id is None:
                failures.append(
                    _data_failure(
                        E_DATA_INVALID,
                        "delete.row_id must be a non-empty string or integer",
                        field="row_id",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                    )
                )
                continue
            row = current_table.row_map().get(row_id)
            if row is None:
                failures.append(
                    _data_failure(
                        E_DATA_INVALID,
                        f"delete targets unknown row_id {row_id!r}",
                        field="row_id",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                        row_id=row_id,
                    )
                )
                continue
            current_table = FeatureTable(
                [candidate for candidate in current_table.rows if candidate.row_id != row_id],
                current_table.columns,
            )
            affected.add(row.group_id)
            applied_count += 1
            continue

        if op == "regroup":
            row_id = _identifier(action.get("row_id"))
            group_id = _identifier(action.get("group_id"))
            if row_id is None or group_id is None:
                failures.append(
                    _data_failure(
                        E_DATA_INVALID,
                        "regroup requires row_id and group_id",
                        field="row_id",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                    )
                )
                continue
            row = current_table.row_map().get(row_id)
            if row is None:
                failures.append(
                    _data_failure(
                        E_DATA_INVALID,
                        f"regroup targets unknown row_id {row_id!r}",
                        field="row_id",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                        row_id=row_id,
                    )
                )
                continue
            if group_id not in current_groups:
                failures.append(
                    _data_failure(
                        E_GROUP_UNKNOWN,
                        f"regroup targets unknown group_id {group_id!r}",
                        field="group_id",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                        row_id=row_id,
                    )
                )
                continue
            index = next(i for i, candidate in enumerate(current_table.rows) if candidate.row_id == row_id)
            updated = FeatureRow(
                order=row.order,
                source_line=row.source_line,
                row_id=row.row_id,
                group_id=group_id,
                values=dict(row.values),
            )
            current_table = _replace_row(current_table, index, updated)
            affected.add(row.group_id)
            affected.add(group_id)
            applied_count += 1
            continue

        if op == "repartition_columns":
            groups_raw = action.get("column_groups")
            if not isinstance(groups_raw, dict) or not groups_raw:
                failures.append(
                    _data_failure(
                        E_CONFIG_INVALID,
                        "repartition_columns.column_groups must be a non-empty object",
                        field="column_groups",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                    )
                )
                continue
            new_groups: dict[str, tuple[str, ...]] = {}
            seen_columns: set[str] = set()
            invalid = False
            for group_id in sorted(groups_raw, key=str):
                columns_raw = groups_raw[group_id]
                if not isinstance(group_id, str) or not group_id:
                    failures.append(
                        _data_failure(
                            E_CONFIG_INVALID,
                            "repartition group identifiers must be non-empty strings",
                            field="column_groups",
                            batch_index=batch_index,
                            action_index=action_index,
                            batch_id=batch.batch_id,
                        )
                    )
                    invalid = True
                    break
                if not isinstance(columns_raw, list) or not columns_raw:
                    failures.append(
                        _data_failure(
                            E_CONFIG_INVALID,
                            f"repartition group {group_id!r} must have a non-empty column array",
                            field=f"column_groups.{group_id}",
                            batch_index=batch_index,
                            action_index=action_index,
                            batch_id=batch.batch_id,
                        )
                    )
                    invalid = True
                    break
                columns: list[str] = []
                for column in columns_raw:
                    if not isinstance(column, str) or column not in current_table.columns:
                        failures.append(
                            _data_failure(
                                E_CONFIG_INVALID,
                                f"repartition references unknown column {column!r}",
                                field=f"column_groups.{group_id}",
                                batch_index=batch_index,
                                action_index=action_index,
                                batch_id=batch.batch_id,
                            )
                        )
                        invalid = True
                        break
                    if column in seen_columns:
                        failures.append(
                            _data_failure(
                                E_CONFIG_INVALID,
                                f"repartition assigns column {column!r} more than once",
                                field=f"column_groups.{group_id}",
                                batch_index=batch_index,
                                action_index=action_index,
                                batch_id=batch.batch_id,
                            )
                        )
                        invalid = True
                        break
                    seen_columns.add(column)
                    columns.append(column)
                if invalid:
                    break
                new_groups[group_id] = tuple(sorted(columns))
            if invalid:
                continue
            missing_columns = sorted(set(current_table.columns) - seen_columns)
            if missing_columns:
                failures.append(
                    _data_failure(
                        E_CONFIG_INVALID,
                        f"repartition leaves columns unassigned: {missing_columns}",
                        field="column_groups",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                    )
                )
                continue
            unknown_row_groups = sorted(
                {row.group_id for row in current_table.rows if row.group_id not in new_groups}
            )
            if unknown_row_groups:
                failures.append(
                    _data_failure(
                        E_GROUP_UNKNOWN,
                        f"repartition removes groups still referenced by rows: {unknown_row_groups}",
                        field="column_groups",
                        batch_index=batch_index,
                        action_index=action_index,
                        batch_id=batch.batch_id,
                    )
                )
                continue
            affected.update(current_groups)
            affected.update(new_groups)
            current_groups = new_groups
            applied_count += 1
            continue

    return BatchApplyResult(
        table=current_table,
        column_groups=current_groups,
        failures=failures,
        affected_groups=affected,
        applied_action_count=applied_count,
    )


def select_batches(plan: EditPlan, through_batch: str | None) -> tuple[PlanBatch, ...]:
    if through_batch is None:
        return plan.batches
    selected: list[PlanBatch] = []
    for batch in plan.batches:
        selected.append(batch)
        if batch.batch_id == through_batch:
            return tuple(selected)
    raise KeyError(through_batch)

