from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from columnar_pipeline.dataset import clone_dataset_with_query, load_dataset
from columnar_pipeline.errors import PipelineError
from columnar_pipeline.execution import make_plan
from columnar_pipeline.query import parse_query


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_query_normalization_and_plan_identity(small_data: Path) -> None:
    dataset = load_dataset(small_data)
    document = json.loads(small_data.read_text(encoding="utf-8"))
    equivalent_where = (
        "(((amount <= 900000)) AND (channel='online')) "
        "AND ((amount>=100))"
    )
    query_doc = copy.deepcopy(document["query"])
    query_doc["where"] = equivalent_where
    query_doc["aggregates"] = list(reversed(query_doc["aggregates"]))
    normalized = parse_query(query_doc, dataset.tables)
    assert normalized.query_id == dataset.query.query_id

    alternate = clone_dataset_with_query(dataset, normalized)
    plan_a = make_plan(dataset, 16)
    plan_b = make_plan(alternate, 16)
    assert plan_a.plan_digest == plan_b.plan_digest
    assert [
        (task.task_id, task.row_start, task.row_end, task.referenced_columns)
        for task in plan_a.tasks
    ] == [
        (task.task_id, task.row_start, task.row_end, task.referenced_columns)
        for task in plan_b.tasks
    ]


@pytest.mark.parametrize(
    ("mutation", "location_prefix"),
    [
        (lambda doc: doc["tables"][1].update({"name": doc["tables"][0]["name"]}), "$.tables[1].name"),
        (lambda doc: doc["tables"][0]["columns"][1].update({"name": "row_id"}), "$.tables[0].columns[1].name"),
        (lambda doc: doc["tables"][0]["columns"][1].update({"type": "float"}), "$.tables[0].columns[1].type"),
        (lambda doc: doc["tables"][0]["columns"][0]["null_bitmap"].pop(), "$.tables[0].columns[0].null_bitmap"),
        (lambda doc: doc["tables"][0]["columns"][0]["null_bitmap"].__setitem__(0, 2), "$.tables[0].columns[0].null_bitmap[0]"),
        (lambda doc: doc["tables"][0]["columns"][3].__setitem__("values", [0] * doc["tables"][0]["row_count"]), "$.tables[0].columns[3].values"),
        (lambda doc: doc["query"]["group_by"].append("missing_group"), "query.group_by[1]"),
        (lambda doc: doc["query"]["aggregates"][1].update({"column": "channel"}), "query.aggregates[1].column"),
        (lambda doc: doc["query"].update({"where": "missing = 1"}), "query.where.column.missing"),
        (lambda doc: doc["query"].update({"from": ["missing_table"]}), "query.from[0]"),
        (lambda doc: doc["query"].update({"where": "(channel = 'online'"}), "query.where@"),
    ],
)
def test_invalid_data_is_rejected_with_location(
    tmp_path: Path,
    small_data: Path,
    mutation,
    location_prefix: str,
) -> None:
    document = json.loads(small_data.read_text(encoding="utf-8"))
    mutation(document)
    invalid_path = _write_json(tmp_path / "invalid.json", document)
    with pytest.raises(PipelineError) as captured:
        load_dataset(invalid_path)
    error = captured.value
    assert error.code == "E_DATA"
    assert error.location is not None
    assert error.location.startswith(location_prefix)


def test_declared_table_digest_mismatch_uses_digest_code(
    tmp_path: Path,
    small_data: Path,
) -> None:
    document = json.loads(small_data.read_text(encoding="utf-8"))
    document["tables"][0]["digest"] = "0" * 64
    path = _write_json(tmp_path / "bad_digest.json", document)
    with pytest.raises(PipelineError) as captured:
        load_dataset(path)
    assert captured.value.code == "E_DIGEST"
    assert captured.value.location == "$.tables[0].digest"


def test_invalid_json_reports_line_and_column(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"tables": [}', encoding="utf-8")
    with pytest.raises(PipelineError) as captured:
        load_dataset(path)
    assert captured.value.code == "E_DATA"
    assert ":1:" in str(captured.value.location)


def test_count_on_fixed_column_has_zero_scale(small_data: Path) -> None:
    dataset = load_dataset(small_data)
    document = json.loads(small_data.read_text(encoding="utf-8"))
    query_doc = document["query"]
    query_doc["aggregates"].append(
        {"name": "amount_nonnull", "op": "count", "column": "amount"}
    )
    query = parse_query(query_doc, dataset.tables)
    aggregate = next(item for item in query.aggregates if item.name == "amount_nonnull")
    assert aggregate.scale == 0
