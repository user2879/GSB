from __future__ import annotations

import copy
import json

import pytest

from joinlab import LabError, load_workload
from joinlab.loader import load_workload_dict

from conftest import SAMPLE, WHITESPACE_SAMPLE


def test_valid_workload_and_null_bitmap(workload):
    assert [table.name for table in workload.tables] == ["customers", "orders", "regions"]
    orders = workload.table_map()["orders"]
    assert orders.row_count == 480
    assert len(orders.null_bitmap) == orders.row_count
    assert all(len(bits) == len(orders.columns) for bits in orders.null_bitmap)
    assert orders.rows[36][5] is None
    assert orders.histograms["discount"].version == 1


@pytest.mark.parametrize(
    ("sample", "field_prefix"),
    [
        ("bad_overlap_histogram.json", "histograms.amount.bins"),
        ("bad_schema_bitmap.json", "null_bitmap[0]"),
    ],
)
def test_invalid_samples_report_first_schema_location(sample, field_prefix):
    with pytest.raises(LabError) as caught:
        load_workload(SAMPLE.parent / sample)
    error = caught.value
    assert error.code == "E_SCHEMA"
    assert error.table == "orders" or error.table == "customers"
    assert error.field.startswith(field_prefix)


def test_duplicate_column_and_unknown_query_reference():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    duplicate = copy.deepcopy(raw)
    duplicate["tables"][0]["columns"][1]["name"] = duplicate["tables"][0]["columns"][0]["name"]
    with pytest.raises(LabError) as caught:
        load_workload_dict(duplicate)
    assert caught.value.code == "E_SCHEMA"
    assert caught.value.table == "regions"
    assert caught.value.field.endswith(".name")

    unknown = copy.deepcopy(raw)
    unknown["query"]["sql"] = unknown["query"]["sql"].replace(
        "customers.id", "customers.missing_id", 1
    )
    with pytest.raises(LabError) as caught:
        load_workload_dict(unknown)
    assert caught.value.code == "E_SCHEMA"
    assert caught.value.table == "customers"
    assert caught.value.column == "missing_id"
    assert caught.value.field == "query.joins.right"


def test_histogram_gap_rejected():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    # Locate the orders table independent of declaration order changes.
    orders = next(table for table in raw["tables"] if table["name"] == "orders")
    bins = orders["histograms"]["amount"]["bins"]
    bins[2]["lo"] = bins[2]["lo"] + 0.25
    with pytest.raises(LabError) as caught:
        load_workload_dict(raw)
    assert caught.value.code == "E_SCHEMA"
    assert caught.value.table == "orders"
    assert caught.value.column == "amount"
    assert "空隙" in caught.value.message


def test_whitespace_and_parentheses_have_identical_normalized_digest():
    first = load_workload(SAMPLE)
    second = load_workload(WHITESPACE_SAMPLE)
    assert first.digest == second.digest
    assert first.query.canonical() == second.query.canonical()
