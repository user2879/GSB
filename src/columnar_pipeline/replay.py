from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .canonical import canonical_json
from .errors import replay_error
from .execution import (
    ShardOutcome,
    TaskSpec,
    build_final_result,
    execute_task,
    hot_groups,
    merge_aggregate_maps,
    result_digest,
)
from .ledger import Ledger
from .model import Dataset, Query, RunConfig


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    snapshot_id: str
    result_json: str
    result_digest: str
    replayed_tasks: int
    task_count: int
    split_count: int
    ledger_digest: str


def group_types_for_dataset(dataset: Dataset, query: Query) -> tuple[str, ...]:
    table = dataset.table(dataset.query.from_tables[0])
    columns = table.column_map()
    return tuple(columns[name].type_name for name in query.group_by)


def _parse_json(value: str, *, task_id: str, field_name: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise replay_error(
            "ledger field is not valid JSON",
            task_id=task_id,
            field_name=field_name,
            details={"error": error.msg, "value": value},
        ) from error


def _task_from_row(row: Mapping[str, Any]) -> TaskSpec:
    try:
        referenced = tuple(json.loads(row["referenced_columns_json"]))
    except Exception as error:
        raise replay_error(
            "task referenced_columns_json is invalid",
            task_id=str(row["task_id"]),
            field_name="referenced_columns",
            details={"error": str(error)},
        ) from error
    return TaskSpec(
        task_id=str(row["task_id"]),
        table_name=str(row["table_name"]),
        table_version=int(row["table_version"]),
        table_digest=str(row["table_digest"]),
        row_start=int(row["row_start"]),
        row_end=int(row["row_end"]),
        referenced_columns=referenced,
        split_eligible=bool(row["split_eligible"]),
        parent_task_id=row["parent_task_id"],
        split_source=row["split_source"],
        estimated_bytes=int(row["estimate_bytes"]),
    )


def _assert_equal(
    *,
    task_id: str,
    field_name: str,
    expected: Any,
    actual: Any,
) -> None:
    if expected != actual:
        raise replay_error(
            f"replay mismatch at task {task_id!r}, field {field_name!r}",
            task_id=task_id,
            field_name=field_name,
            details={"expected": expected, "actual": actual},
        )


def replay_ledger_run(
    *,
    ledger: Ledger,
    snapshot: Mapping[str, Any],
    dataset: Dataset,
    config: RunConfig,
) -> ReplayResult:
    run_id = str(snapshot["run_id"])
    session = ledger.get_session(run_id)
    if session is None:
        raise replay_error(
            "snapshot run does not exist in the ledger",
            task_id="<session>",
            field_name="run_id",
            details={"run_id": run_id},
        )
    if session.get("status") != "complete":
        raise replay_error(
            "ledger session is not complete and cannot be replayed",
            task_id="<session>",
            field_name="status",
            details={"expected": "complete", "actual": session.get("status")},
        )
    _assert_equal(
        task_id="<session>",
        field_name="query_id",
        expected=snapshot["query_id"],
        actual=session.get("query_id"),
    )
    task_rows = ledger.get_tasks(run_id)
    group_types = group_types_for_dataset(dataset, dataset.query)
    included_outcomes: list[ShardOutcome] = []
    replay_orders: list[int] = []
    for row in task_rows:
        task_id = str(row["task_id"])
        task = _task_from_row(row)
        try:
            table = dataset.table(task.table_name)
        except KeyError as error:
            raise replay_error(
                "task references a table missing from current data",
                task_id=task_id,
                field_name="table",
                details={"table": task.table_name},
            ) from error
        _assert_equal(
            task_id=task_id,
            field_name="table_version",
            expected=int(row["table_version"]),
            actual=table.version,
        )
        _assert_equal(
            task_id=task_id,
            field_name="table_digest",
            expected=str(row["table_digest"]),
            actual=table.digest,
        )
        outcome = execute_task(dataset, dataset.query, task)
        detected = hot_groups(outcome, config.split_threshold)
        computed_status = "completed"
        if detected and task.split_eligible:
            if detected[0][1] > config.skew_safety_limit:
                computed_status = "skew_rejected"
            else:
                computed_status = "split_parent"
        computed_included = computed_status == "completed"
        _assert_equal(
            task_id=task_id,
            field_name="status",
            expected=str(row["status"]),
            actual=computed_status,
        )
        _assert_equal(
            task_id=task_id,
            field_name="rows_completed",
            expected=int(row["rows_completed"]),
            actual=outcome.rows_completed,
        )
        stored_aggregate = _parse_json(
            str(row["aggregate_json"]),
            task_id=task_id,
            field_name="aggregate",
        )
        _assert_equal(
            task_id=task_id,
            field_name="aggregate",
            expected=canonical_json(stored_aggregate),
            actual=canonical_json(outcome.aggregate_state),
        )
        _assert_equal(
            task_id=task_id,
            field_name="digest",
            expected=str(row["digest"]),
            actual=outcome.digest,
        )
        _assert_equal(
            task_id=task_id,
            field_name="included",
            expected=bool(row["included"]),
            actual=computed_included,
        )
        if computed_included:
            included_outcomes.append(outcome)
        replay_orders.append(int(row["replay_order"]))
    if replay_orders != sorted(replay_orders):
        first_bad = next(
            index for index, value in enumerate(replay_orders) if value != index
        )
        raise replay_error(
            "ledger replay_order is not contiguous and stable",
            task_id=str(task_rows[first_bad]["task_id"]),
            field_name="replay_order",
            details={"expected": first_bad, "actual": replay_orders[first_bad]},
        )

    merged = merge_aggregate_maps(included_outcomes, dataset.query, group_types)
    final_result = build_final_result(dataset.query, group_types, merged)
    final_json = canonical_json(final_result)
    final_digest = result_digest(final_result)
    _assert_equal(
        task_id="<final>",
        field_name="result_json",
        expected=snapshot["result_json"],
        actual=final_json,
    )
    _assert_equal(
        task_id="<final>",
        field_name="result_digest",
        expected=snapshot["result_digest"],
        actual=final_digest,
    )
    _assert_equal(
        task_id="<session>",
        field_name="result_json",
        expected=snapshot["result_json"],
        actual=session.get("result_json"),
    )
    _assert_equal(
        task_id="<session>",
        field_name="result_digest",
        expected=snapshot["result_digest"],
        actual=session.get("result_digest"),
    )
    ledger_digest = ledger.semantic_digest(run_id)
    _assert_equal(
        task_id="<ledger>",
        field_name="ledger_digest",
        expected=snapshot["ledger_digest"],
        actual=ledger_digest,
    )
    _assert_equal(
        task_id="<snapshot>",
        field_name="task_count",
        expected=snapshot.get("task_count", len(task_rows)),
        actual=len(task_rows),
    )
    return ReplayResult(
        run_id=run_id,
        snapshot_id=str(snapshot["snapshot_id"]),
        result_json=final_json,
        result_digest=final_digest,
        replayed_tasks=len(task_rows),
        task_count=len(task_rows),
        split_count=sum(1 for row in task_rows if row["status"] == "split_parent"),
        ledger_digest=ledger_digest,
    )
