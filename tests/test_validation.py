from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from imputation_workbench.config import load_config, normalise_config
from imputation_workbench.loader import (
    group_coverage_failures,
    load_group_map,
    load_raw_table,
)
from imputation_workbench.engine import _failed_row_ids
from imputation_workbench.samplegen import generate_sample


def _read_lines(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_lines(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def test_bad_rows_are_isolated_and_later_rows_load(workspace: Path, tmp_path: Path) -> None:
    config, errors = load_config(workspace / "imputation.yaml")
    assert errors == []
    rows = _read_lines(workspace / "raw_data.jsonl")
    rows[1]["age"] = {"missing": True, "value": 44}
    rows[2]["city"] = "not-in-domain"
    rows[3]["age"] = float("nan")
    rows.append(deepcopy(rows[0]))  # duplicate row id
    bad_path = tmp_path / "bad.jsonl"
    _write_lines(bad_path, rows)

    valid_rows, failures, schema = load_raw_table(bad_path, config)
    assert schema
    assert len(failures) >= 4
    assert {item["business_code"] for item in failures} == {"E_DATA_ROW_INVALID"}
    assert any(item.get("row_id") == rows[0]["row_id"] for item in failures)
    # Four invalid lines were skipped; all remaining lines still loaded.
    assert len(valid_rows) == len(rows) - 4
    assert [row["row_id"] for row in valid_rows] == sorted(row["row_id"] for row in valid_rows)


def test_group_coverage_uses_group_business_code(workspace: Path) -> None:
    config, errors = load_config(workspace / "imputation.yaml")
    assert errors == []
    group_map, map_errors = load_group_map(workspace / "group_map.json")
    assert map_errors == []
    rows, data_failures, _ = load_raw_table(workspace / "raw_data.jsonl", config)
    assert data_failures == []
    rows[0]["values"]["region"] = "unmapped"
    failures = group_coverage_failures(rows, config, group_map)
    assert failures
    assert failures[0]["business_code"] == "E_GROUP_MAP_INVALID"
    assert failures[0]["row_id"] == rows[0]["row_id"]


def test_config_validation_reports_all_invalid_fields(workspace: Path) -> None:
    config, errors = load_config(workspace / "imputation.yaml")
    assert errors == []
    bad = deepcopy(config)
    bad["batch_cell_budget"] = 0
    bad["columns"]["age"]["min_valid_samples"] = 0
    bad["columns"]["income"]["decimal_places"] = 99
    bad["columns"]["city"]["imputer"] = "median"
    bad["columns"]["score"]["group_key"] = "age"
    _, bad_errors = normalise_config(bad)
    codes = {error["business_code"] for error in bad_errors}
    assert codes == {"E_CONFIG_INVALID"}
    fields = {error.get("field") for error in bad_errors}
    assert "batch_cell_budget" in fields
    assert "columns.age.min_valid_samples" in fields
    assert "columns.income.decimal_places" in fields
    assert "columns.city.imputer" in fields
    assert "columns.score.group_key" in fields


def test_seeded_sample_generation_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_sample(first, seed=20261121, rows=24)
    generate_sample(second, seed=20261121, rows=24)
    for name in ("imputation.yaml", "raw_data.jsonl", "group_map.json", "edit_plan.json", "injections.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_duplicate_line_does_not_exclude_the_valid_row_from_training() -> None:
    state = {
        "rows": [{"row_id": "r1", "line_no": 1, "values": {}}],
        "data_failures": [
            {
                "row_id": "r1",
                "line_no": 2,
                "business_code": "E_DATA_ROW_INVALID",
                "stage": "load",
            },
            {
                "row_id": "r9",
                "line_no": 3,
                "business_code": "E_DATA_ROW_INVALID",
                "stage": "load",
            },
        ],
        "group_failures": [],
        "group_map_failures": [],
        "operation_failures": [],
        "imputation_failures": [],
    }
    assert _failed_row_ids(state) == {"r9"}
