from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from vscan.canonical import INT64_MAX, clamp_int64, trunc_div_toward_zero
from vscan.codec import ColumnData, read_column
from vscan.executor import (
    AggregateState,
    evaluate_predicate,
    execute_batch_for_query,
    merge_query_result,
)
from vscan.planner import compile_plan
from vscan.session import Session
from vscan.spec import parse_spec
from vscan.store import build_store, column_path


def _naive_query(spec, root: Path):
    query = spec.queries[0]
    segment = spec.segments[0]
    columns = {
        column.name: read_column(column_path(root, segment.id, column.name))
        for column in segment.columns
    }
    grouped: dict[tuple[int | None, ...], list[tuple[int | None, ...]]] = {}
    for row in range(segment.rows):
        values = {name: (None if bool(data.null_mask[row]) else int(data.values[row])) for name, data in columns.items()}
        keep = True
        for arg in query.predicate["args"]:
            value = values[arg["column"]]
            if arg["op"] == "ge":
                keep &= value is not None and value >= arg["value"]
            elif arg["op"] == "ne":
                keep &= value is not None and value != arg["value"]
        if not keep:
            continue
        key = tuple(values[column] for column in query.group_by)
        grouped.setdefault(key, []).append(
            tuple(values[agg.column] if agg.column else 1 for agg in query.aggregates)
        )
    return grouped


def test_vectorized_result_matches_naive_scan(tmp_path, spec):
    with Session(spec, tmp_path / "data", tmp_path / "ledger.sqlite") as session:
        report = session.run()
        expected = _naive_query(spec, tmp_path / "data")
        groups = report["queries"][0]["groups"]
        assert {tuple(group["key"]) for group in groups} == set(expected)
        for group in groups:
            key = tuple(group["key"])
            rows = expected[key]
            by_name = {state["name"]: state for state in group["aggregates"]}
            assert by_name["n"]["value"] == len(rows)
            values = [row[1] for row in rows if row[1] is not None]
            nulls = sum(row[1] is None for row in rows)
            assert by_name["sum_v"]["value"] == clamp_int64(sum(values))
            assert by_name["min_v"]["value"] == (min(values) if values else None)
            assert by_name["max_v"]["value"] == (max(values) if values else None)
            assert by_name["avg_v"]["value"] == (
                clamp_int64(trunc_div_toward_zero(sum(values), len(values))) if values else 0
            )
            assert by_name["sum_v"]["null_count"] == nulls


def test_predicate_is_one_numpy_boolean_mask_with_null_false(tmp_path, spec):
    segment = spec.segments[0]
    store = build_store(spec, tmp_path / "data")
    predicate = spec.queries[0].predicate
    columns = {
        "g": read_column(column_path(tmp_path / "data", segment.id, "g")),
        "v": read_column(column_path(tmp_path / "data", segment.id, "v")),
    }
    mask = evaluate_predicate(predicate, columns, segment.rows)
    assert mask.dtype == np.dtype("?")
    assert mask.shape == (segment.rows,)
    for row in range(segment.rows):
        if bool(columns["g"].null_mask[row]) or bool(columns["v"].null_mask[row]):
            assert not bool(mask[row])
        else:
            expected = int(columns["v"].values[row]) >= 5 and int(columns["g"].values[row]) != 1
            assert bool(mask[row]) == expected
    assert store.segments[0].id == segment.id


def test_batch_execution_order_does_not_change_aggregate(tmp_path, spec):
    store = build_store(spec, tmp_path / "data")
    plan = compile_plan(spec)
    query_plan = plan.queries[0]
    computations = [
        execute_batch_for_query(spec, store, tmp_path / "data", query_plan.query, batch)
        for batch in query_plan.batches
    ]
    from vscan.executor import BatchResult

    forward = [BatchResult(batch, computation) for batch, computation in zip(query_plan.batches, computations, strict=True)]
    reverse = list(reversed(forward))
    left = merge_query_result(spec, query_plan.query, forward)
    right = merge_query_result(spec, query_plan.query, reverse)
    assert left.aggregate_digest == right.aggregate_digest
    assert [group.to_dict() for group in left.groups] == [group.to_dict() for group in right.groups]
    assert {result.key: result.aggregate_digest for result in forward} == {
        result.key: result.aggregate_digest for result in reverse
    }


def test_fixed_point_saturation_and_truncation():
    state = AggregateState(
        name="sum",
        op="sum",
        column="v",
        count_rows=2,
        non_null_count=2,
        null_count=0,
        sum_raw=INT64_MAX + 10,
        min_value=None,
        max_value=None,
    )
    assert state.value() == INT64_MAX
    average = AggregateState(
        name="avg",
        op="avg",
        column="v",
        count_rows=2,
        non_null_count=2,
        null_count=0,
        sum_raw=-5,
        min_value=None,
        max_value=None,
    )
    assert average.value() == -2
    assert trunc_div_toward_zero(-5, 2) == -2
    assert trunc_div_toward_zero(5, -2) == -2


def test_column_data_empty_and_null_only():
    data = ColumnData(np.array([], dtype="<i8"), np.array([], dtype="?"))
    assert data.rows == 0
    mask = evaluate_predicate(None, {}, 0)
    assert mask.tolist() == []


def test_multi_column_grouping_with_null_group_keys(tmp_path, spec_dict):
    data = copy.deepcopy(spec_dict)
    data["queries"][0] = {
        "id": "q_multi",
        "project": ["g", "r", "v"],
        "predicate": {
            "op": "or",
            "args": [
                {"op": "is_null", "column": "g"},
                {"op": "in", "column": "v", "values": [3, 5, 9, 13]},
            ],
        },
        "group_by": ["g", "r"],
        "aggregates": [
            {"name": "n", "op": "count"},
            {"name": "nv", "op": "count", "column": "v"},
        ],
    }
    spec = parse_spec(data)
    with Session(spec, tmp_path / "data", tmp_path / "ledger.sqlite") as session:
        report = session.run()
        groups = report["queries"][0]["groups"]
        assert any(group["key"][0] is None for group in groups)
        assert all(len(group["key"]) == 2 for group in groups)
        total_rows = sum(
            next(state["value"] for state in group["aggregates"] if state["name"] == "n")
            for group in groups
        )
        assert total_rows == sum(batch["passed_rows"] for batch in report["batch_results"])
        assert session.verify_against_full_rebuild()["consistent"] is True
