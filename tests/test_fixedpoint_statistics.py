from __future__ import annotations

import json
from pathlib import Path
import shutil
import yaml

from imputation_workbench.config import load_config
from imputation_workbench.engine import WorkbenchEngine
from imputation_workbench.stats import aggregate_stats_once

from conftest import load_state, make_paths


def test_mode_tie_breaks_by_lexicographic_value(workspace) -> None:
    config, errors = load_config(workspace / "imputation.yaml")
    assert errors == []
    rows = [
        {"row_id": "r1", "values": {"city": "river"}},
        {"row_id": "r2", "values": {"city": "delta"}},
        {"row_id": "r3", "values": {"city": "river"}},
        {"row_id": "r4", "values": {"city": "delta"}},
    ]
    record, metrics = aggregate_stats_once(
        "city", config, None, rows, {row["row_id"] for row in rows}, {}
    )
    assert metrics["computed"] is True
    assert record["value"] == "delta"
    assert record["value_fixed"] == sorted(config["columns"]["city"]["domain"]).index("delta")


def _copy_workspace_without_plan_effect(workspace: Path, target: Path, decimal_places: int, quantum: int) -> Path:
    shutil.copytree(workspace, target)
    config_path = target / "imputation.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["decimal_places"] = decimal_places
    config["quantization_level"] = quantum
    with config_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    plan = {
        "format": 1,
        "batches": [
            {
                "batch_id": "no-op",
                "operations": [
                    {
                        "op": "change_group_mapping",
                        "key_column": "region",
                        "changes": {},
                    }
                ],
            }
        ],
    }
    with (target / "edit_plan.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle)
    return target


def _first_income_cell(state):
    for row_id in sorted(state["imputations"]):
        cell = state["imputations"][row_id].get("income")
        if cell is not None:
            return row_id, cell
    raise AssertionError("sample has no imputed income cell")


def test_width_and_quantization_change_imputed_values_and_fingerprints(workspace, tmp_path) -> None:
    variants = {"base": (2, 5), "width": (3, 5), "quantization": (2, 7)}
    cells = {}
    for name, (places, quantum) in variants.items():
        variant = _copy_workspace_without_plan_effect(workspace, tmp_path / name, places, quantum)
        paths = make_paths(variant, "work", injections=False)
        WorkbenchEngine(paths).run(fresh=True)
        state, _ = load_state(paths)
        row_id, cell = _first_income_cell(state)
        cells[name] = (row_id, cell)
    assert len({row_id for row_id, _cell in cells.values()}) == 1
    assert cells["base"][1]["value"] != cells["width"][1]["value"]
    assert cells["base"][1]["value"] != cells["quantization"][1]["value"]
    for name, (_row_id, cell) in cells.items():
        assert cell["fingerprint"] != cells["base"][1]["fingerprint"] or name == "base"
        assert cell["cache_key"] != cells["base"][1]["cache_key"] or name == "base"
    assert cells["base"][1]["decimal_places"] == 2
    assert cells["width"][1]["decimal_places"] == 3
    assert cells["base"][1]["quantization_level"] == 5
    assert cells["quantization"][1]["quantization_level"] == 7
