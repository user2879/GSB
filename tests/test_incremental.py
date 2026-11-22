from __future__ import annotations

import json
from pathlib import Path

from qcolumn_diag.pipeline import run_workbench
from tests.helpers import ROOT, assert_deterministic_equal, write_json


def _examples() -> tuple[dict, dict]:
    config = json.loads((ROOT / "examples" / "config.json").read_text(encoding="utf-8"))
    sample = json.loads((ROOT / "examples" / "sample.json").read_text(encoding="utf-8"))
    return config, sample


def _run(
    root: Path,
    config: dict,
    sample: dict,
    *,
    label: str,
    state: Path,
    full_rebuild: bool = False,
) -> dict:
    config_path = write_json(root / "current.config.json", config)
    sample_path = write_json(root / "current.sample.json", sample)
    return run_workbench(
        config_path=config_path,
        sample_path=sample_path,
        state_dir=state,
        report_path=root / f"{label}.report.json",
        full_rebuild=full_rebuild,
    )


def test_edit_sequence_incremental_equals_clean_full_rebuild(tmp_path: Path) -> None:
    config, sample = _examples()
    state = tmp_path / "incremental-state"
    initial = _run(tmp_path, config, sample, label="initial", state=state)
    assert initial["summary"]["column_count_completed"] == 20
    assert initial["cache"]["recomputed_columns"]

    # 1. Replace content in exactly one source column.
    sample["tensors"]["activation_tile"]["injections"][0]["value"] = 8.5
    sample_edit = _run(tmp_path, config, sample, label="sample_edit", state=state)
    assert sample_edit["cache"]["recomputed_columns"] == ["activation_tile:c000000"]
    assert len(sample_edit["cache"]["reused_columns"]) == 19
    assert sample_edit["cache"]["stage_computed_counts"]["calibration"] == 1
    assert sample_edit["cache"]["stage_computed_counts"]["quant_params"] == 2
    assert sample_edit["cache"]["stage_computed_counts"]["requantize_errors"] == 2

    # 2. Saturation strategy affects only the downstream requant/error stage.
    config["tensors"][0]["saturation_strategy"] = "wrap"
    saturation_edit = _run(
        tmp_path, config, sample, label="saturation_edit", state=state
    )
    assert len(saturation_edit["cache"]["recomputed_columns"]) == 12
    assert all(
        column.startswith("activation_tile:")
        for column in saturation_edit["cache"]["recomputed_columns"]
    )
    assert saturation_edit["cache"]["stage_computed_counts"]["calibration"] == 0
    assert saturation_edit["cache"]["stage_computed_counts"]["quant_params"] == 0
    assert saturation_edit["cache"]["stage_computed_counts"]["requantize_errors"] == 24

    # 3. Fixed fractional width invalidates quant parameters and downstream only.
    config["tensors"][0]["fractional_bits"] = 5
    fractional_edit = _run(
        tmp_path, config, sample, label="fractional_edit", state=state
    )
    assert fractional_edit["cache"]["stage_computed_counts"]["calibration"] == 0
    assert fractional_edit["cache"]["stage_computed_counts"]["quant_params"] == 24
    assert fractional_edit["cache"]["stage_computed_counts"]["requantize_errors"] == 24

    # 4. Switching active quantization width changes the selected downstream result.
    config["tensors"][0]["active_bit_width"] = 4
    width_edit = _run(tmp_path, config, sample, label="width_edit", state=state)
    assert width_edit["cache"]["stage_computed_counts"]["quant_params"] == 0
    assert width_edit["cache"]["stage_computed_counts"]["requantize_errors"] == 24
    assert width_edit["columns"][0]["active_bit_width"] == 4

    # 5. Add a new candidate width. Only that width's artifacts recompute.
    config["tensors"][0]["bit_width_candidates"] = [4, 6, 8]
    candidate_edit = _run(
        tmp_path, config, sample, label="candidate_edit", state=state
    )
    assert candidate_edit["cache"]["stage_computed_counts"]["quant_params"] == 12
    assert candidate_edit["cache"]["stage_computed_counts"]["requantize_errors"] == 12

    # 6. Re-layering groups does not invalidate any numerical column artifact.
    config["tensors"][0]["column_groups"] = [
        {"group_id": "g-low", "columns": [0, 1, 2, 3, 4, 5]},
        {"group_id": "g-high", "columns": [6, 7, 8, 9, 10, 11]},
    ]
    config["tensors"][1]["column_groups"] = [
        {"group_id": "g-weights", "columns": [0, 1, 2, 3, 4, 5, 6, 7]}
    ]
    final_incremental = _run(
        tmp_path, config, sample, label="group_edit", state=state
    )
    assert final_incremental["cache"]["recomputed_columns"] == []
    assert len(final_incremental["cache"]["reused_columns"]) == 20
    assert {entry["group_id"] for entry in final_incremental["sensitivity"]["groups"]} == {
        "g-low",
        "g-high",
        "g-weights",
    }

    clean_full = _run(
        tmp_path,
        config,
        sample,
        label="clean_full",
        state=tmp_path / "full-state",
        full_rebuild=True,
    )
    assert clean_full["cache"]["mode"] == "full_rebuild"
    assert_deterministic_equal(final_incremental, clean_full)
    assert final_incremental["failures"] == clean_full["failures"]
    assert final_incremental["sensitivity"] == clean_full["sensitivity"]
