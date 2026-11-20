from __future__ import annotations

from copy import deepcopy
import json
import shutil

from imputation_workbench.engine import WorkbenchEngine, compare_semantic_states
from imputation_workbench.errors import E_UPSTREAM_STALE, WorkbenchError
import imputation_workbench.impute as impute_module
from imputation_workbench.model import assert_model_fresh, train_and_evaluate
from conftest import load_state, make_paths


def test_full_run_is_deterministic_and_reuses_everything(workspace) -> None:
    paths = make_paths(workspace, "work-a")
    first = WorkbenchEngine(paths).run(fresh=True)
    assert first["status"] == "complete"
    state_a, _manifest_a = load_state(paths)

    second = WorkbenchEngine(paths).run()
    assert second["status"] == "complete"
    state_b, _manifest_b = load_state(paths)
    assert compare_semantic_states(state_a, state_b)["equal"]
    report = second["report"]
    assert report["stats"]["recomputed_columns"] == []
    assert report["stats"]["recomputed_groups"] == []
    assert report["imputation"]["computed_cell_count"] == 0
    assert report["model"]["reused"] is True
    assert {item["injection_id"] for item in report["injection_hits"]} == {
        "INJ-001",
        "INJ-002",
        "INJ-003",
        "INJ-004",
    }
    assert all(item["matched"] for item in report["injection_hits"])

    report_a = first["report"]
    assert "E_INSUFFICIENT_SAMPLES" in report_a["business_codes"]
    assert report_a["stats"]["columns"]["score"]["insufficient_groups"] == ["highland"]
    assert report_a["stats"]["columns"]["score"]["groups"]["highland"]["fallback_used"] is True
    inland = report_a["stats"]["columns"]["score"]["groups"]["inland"]
    assert inland["group_key_values"] == ["north", "south"]
    assert inland["referenced_row_ids"]
    assert report_a["model"]["excluded_row_ids"] == ["r0007", "r0014"]
    assert report_a["model"]["loss_trajectory"]
    assert 0.0 <= report_a["model"]["accuracy"] <= 1.0
    assert report_a["model"]["log_loss"] >= 0.0
    assert report_a["snapshot"]["stage_versions"]
    assert set(report_a["timings_seconds"]) >= {"stats", "imputation", "model", "snapshot", "total"}


def test_incremental_matches_fresh_rebuild(workspace) -> None:
    incremental_paths = make_paths(workspace, "incremental")
    partial = WorkbenchEngine(incremental_paths).run(fresh=True, max_batches=2)
    assert partial["status"] == "partial"
    assert partial["report"]["completed_batches"] == ["b1-append", "b2-correct"]
    final = WorkbenchEngine(incremental_paths).run()
    assert final["status"] == "complete"
    incremental_state, _ = load_state(incremental_paths)

    fresh_paths = make_paths(workspace, "fresh")
    WorkbenchEngine(fresh_paths).run(fresh=True)
    fresh_state, _ = load_state(fresh_paths)
    comparison = compare_semantic_states(incremental_state, fresh_state)
    assert comparison["equal"], comparison


def test_column_change_only_recomputes_that_column(workspace) -> None:
    paths = make_paths(workspace, "precise")
    WorkbenchEngine(paths).run(fresh=True, max_batches=2)
    result = WorkbenchEngine(paths).run(max_batches=1)
    assert result["status"] == "partial"
    report = result["report"]
    assert report["completed_batches_this_run"] == ["b3-imputer-change"]
    assert report["stats"]["recomputed_columns"] == ["income"]
    assert "age" in report["stats"]["reused_columns"]
    assert "score" in report["stats"]["reused_columns"]
    assert report["stats"]["recomputed_groups"] == []
    assert report["model"]["reused"] is False


def test_group_mapping_change_reuses_unaffected_groups(workspace) -> None:
    paths = make_paths(workspace, "groups")
    WorkbenchEngine(paths).run(fresh=True, max_batches=3)
    result = WorkbenchEngine(paths).run()
    report = result["report"]
    assert result["status"] == "complete"
    recomputed = {(item["column"], item["group"]) for item in report["stats"]["recomputed_groups"]}
    reused = {(item["column"], item["group"]) for item in report["stats"]["reused_groups"]}
    assert ("score", "highland") in recomputed
    assert ("score", "coastal") in reused
    assert ("score", "inland") in reused


def test_model_rejects_stale_upstream_fingerprint(workspace) -> None:
    paths = make_paths(workspace, "stale")
    WorkbenchEngine(paths).run(fresh=True)
    state, _ = load_state(paths)
    model = state["model"]
    try:
        assert_model_fresh(model, "definitely-not-the-current-fingerprint")
    except WorkbenchError as exc:
        assert exc.code == E_UPSTREAM_STALE
    else:
        raise AssertionError("stale model was not rejected")

    stale = deepcopy(model)
    stale["imputation_fingerprint"] = "stale-input-fingerprint"
    _, metrics, diagnostics = train_and_evaluate(
        state["config"],
        state["rows"],
        state["imputations"],
        set(model["excluded_row_ids"]),
        model["imputation_fingerprint"],
        stale,
    )
    assert metrics["recomputed"] is True
    assert any(item["business_code"] == E_UPSTREAM_STALE for item in diagnostics)


def test_parallel_and_serial_results_are_identical(workspace) -> None:
    serial_paths = make_paths(workspace, "serial")
    parallel_paths = make_paths(workspace, "parallel")
    WorkbenchEngine(serial_paths).run(fresh=True, workers_override=1)
    WorkbenchEngine(parallel_paths).run(fresh=True, workers_override=4)
    serial_state, _ = load_state(serial_paths)
    parallel_state, _ = load_state(parallel_paths)
    assert compare_semantic_states(serial_state, parallel_state)["equal"]


def test_statistics_change_without_missing_cells_still_invalidates_model(workspace, tmp_path) -> None:
    variant = tmp_path / "stats-only"
    shutil.copytree(workspace, variant)
    plan = {
        "format": 1,
        "batches": [
            {
                "batch_id": "baseline",
                "operations": [
                    {"op": "change_group_mapping", "key_column": "region", "changes": {}}
                ],
            },
            {
                "batch_id": "target-width-only",
                "operations": [
                    {
                        "op": "change_imputer",
                        "column": "churn",
                        "changes": {"decimal_places": 3},
                    }
                ],
            },
        ],
    }
    with (variant / "edit_plan.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle)
    paths = make_paths(variant, "work", injections=False)
    WorkbenchEngine(paths).run(fresh=True, max_batches=1)
    result = WorkbenchEngine(paths).run(max_batches=1)
    assert result["report"]["stats"]["recomputed_columns"] == ["churn"]
    assert result["report"]["imputation"]["computed_cell_count"] == 0
    assert result["report"]["model"]["reused"] is False
    assert any(
        item["business_code"] == E_UPSTREAM_STALE
        for item in result["report"]["stale_diagnostics"]
    )


def test_imputation_exception_is_isolated_to_its_row(workspace, monkeypatch) -> None:
    original = impute_module._compute_cell

    def flaky_compute(job, rows_by_id, config, stats, group_map):
        if job["row_id"] == "r0001" and job["column"] == "score":
            raise RuntimeError("injected cell computation failure")
        return original(job, rows_by_id, config, stats, group_map)

    monkeypatch.setattr(impute_module, "_compute_cell", flaky_compute)
    paths = make_paths(workspace, "cell-failure", injections=False)
    result = WorkbenchEngine(paths).run(fresh=True)
    assert result["status"] == "complete"
    failures = [
        failure
        for failure in result["report"]["failures"]
        if failure["business_code"] == "E_IMPUTATION_FAILED"
    ]
    assert failures
    assert failures[0]["row_id"] == "r0001"
    assert failures[0]["column"] == "score"
    assert failures[0]["stage"] == "imputation"
    assert "r0001" in result["report"]["failed_row_ids"]
    assert "r0001" not in result["report"]["model"]["training_row_ids"]
    assert result["report"]["imputation"]["imputed_cell_count"] > 0
