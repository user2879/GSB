import json

from mlimgedit.cachekeys import DECISION, FEATURES, METRICS, SCORING
from mlimgedit.config import load_config_dict
from mlimgedit.injection import load_injections
from mlimgedit.plan import EditPlan
from mlimgedit.runner import (
    comparable_state,
    full_run,
    rebuild_final_state,
    run_plan,
)
from scripts.bootstrap_demo import build_config, build_edit_plan


def _plan_paths(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(build_config(threads=2, budget=50000), sort_keys=True), encoding="utf-8"
    )
    (root / "edit_plan.json").write_text(
        json.dumps(build_edit_plan(), sort_keys=True), encoding="utf-8"
    )
    return root


def test_full_run_is_deterministic(config, sample_dir, tmp_path):
    first = full_run(config, sample_dir, tmp_path / "store_a")
    second = full_run(config, sample_dir, tmp_path / "store_b")
    assert comparable_state(first) == comparable_state(second)


def test_concurrent_and_serial_identical(config, sample_dir, tmp_path):
    serial_config = config.with_updates({"runtime": {"threads": 1}})
    parallel_config = config.with_updates({"runtime": {"threads": 8}})
    serial = full_run(serial_config, sample_dir, tmp_path / "serial")
    parallel = full_run(parallel_config, sample_dir, tmp_path / "parallel")
    assert comparable_state(serial) == comparable_state(parallel)
    assert serial["failures"] == parallel["failures"]


def test_incremental_equals_full_rebuild(tmp_path):
    root = _plan_paths(tmp_path)
    plan = EditPlan.load(root / "edit_plan.json")
    # Build base sample dir from the plan base config.
    from mlimgedit.datagen import generate_dataset

    config = load_config_dict(json.loads((root / "config.json").read_text()))
    generate_dataset(root / "samples", config, 12)

    reports = run_plan(plan, tmp_path / "inc", tmp_path / "inc_store")
    final_config, final_samples = rebuild_final_state(plan, tmp_path / "final")
    full = full_run(final_config, final_samples, tmp_path / "full_store")
    assert comparable_state(reports[-1]) == comparable_state(full)


def test_cache_reuse_only_affects_dependent_labels(tmp_path):
    root = _plan_paths(tmp_path)
    plan = EditPlan.load(root / "edit_plan.json")
    from mlimgedit.datagen import generate_dataset

    config = load_config_dict(json.loads((root / "config.json").read_text()))
    generate_dataset(root / "samples", config, 12)
    reports = run_plan(plan, tmp_path / "inc", tmp_path / "inc_store")

    # Batch 4 changes the global threshold plus logreg hyper-parameters.
    # Only logreg scoring is invalidated; centroid/kNN scoring is reused while
    # every decision/metric artifact is recomputed because the threshold moved.
    threshold_report = reports[4]
    scoring = [item for item in threshold_report["recompute"] if item["stage"] == SCORING]
    scoring_reuse = [item for item in threshold_report["reuse"] if item["stage"] == SCORING]
    decisions = [item for item in threshold_report["recompute"] if item["stage"] == DECISION]
    assert {item["name"].split("/")[0] for item in scoring} == {"logreg"}
    assert {item["name"].split("/")[0] for item in scoring_reuse} == {"centroid", "knn"}
    expected_labels = list(config.labels) + ['diagonal_blue']
    assert len(decisions) == 3 * len(expected_labels)
    metric_recompute = [
        item for item in threshold_report["recompute"] if item["stage"] == METRICS
    ]
    assert len(metric_recompute) == len(decisions)

    # The add-label batch keeps feature artifacts and recomputes only models.
    add_label_report = reports[3]
    feature_recompute = [
        item for item in add_label_report["recompute"] if item["stage"] == FEATURES
    ]
    assert feature_recompute == []
    model_names = {
        item["name"] for item in add_label_report["recompute"] if item["stage"] == SCORING
    }
    assert any(name.endswith("diagonal_blue") for name in model_names)


def test_feature_switch_invalidates_all_features_and_models(tmp_path):
    root = _plan_paths(tmp_path)
    plan = EditPlan.load(root / "edit_plan.json")
    from mlimgedit.datagen import generate_dataset

    config = load_config_dict(json.loads((root / "config.json").read_text()))
    generate_dataset(root / "samples", config, 12)
    reports = run_plan(plan, tmp_path / "inc", tmp_path / "inc_store")
    switch_report = reports[6]
    feature_entries = [item for item in switch_report["audit"] if item["stage"] == FEATURES]
    assert feature_entries and all(item["action"] == "recompute" for item in feature_entries)
    assert {item["stage"] for item in switch_report["audit"]} >= {
        FEATURES,
        SCORING,
        DECISION,
        METRICS,
    }
    assert switch_report["feature_mode"] == "hog"


def test_audit_entries_are_sorted_and_referenced_samples_recorded(tmp_path):
    root = _plan_paths(tmp_path)
    plan = EditPlan.load(root / "edit_plan.json")
    from mlimgedit.datagen import generate_dataset

    config = load_config_dict(json.loads((root / "config.json").read_text()))
    generate_dataset(root / "samples", config, 12)
    reports = run_plan(plan, tmp_path / "inc", tmp_path / "inc_store")
    for report in reports:
        # The combined audit list is sorted by stage rank then stable key.
        entries = report["audit"]
        stage_rank = {
            stage: index
            for index, stage in enumerate(
                ("validation", "features", "scoring", "decision", "metrics")
            )
        }
        ordered = sorted(
            entries, key=lambda entry: (stage_rank[entry["stage"]], entry["sort_key"])
        )
        assert entries == ordered
        # The split lists partition the same audit entries.
        assert len(report["reuse"]) + len(report["recompute"]) == len(entries)
        models = [entry for entry in entries if entry["stage"] == SCORING]
        if models:
            assert all(entry["referenced_samples"] for entry in models)


def test_failures_are_isolated_and_stable_across_runs(config, sample_dir, injections_path, tmp_path):
    injections = load_injections(injections_path)
    first = full_run(
        config, sample_dir, tmp_path / "a", injections=injections
    )
    second = full_run(
        config, sample_dir, tmp_path / "b", injections=injections
    )
    assert first["failures"] == second["failures"]
    codes = {failure["code"] for failure in first["failures"]}
    assert {"E_SAMPLE_FORMAT", "E_LABEL_UNKNOWN", "E_BBOX_OUT_OF_RANGE"} <= codes
    # Valid samples still produce predictions despite isolated failures.
    assert len(first["per_sample_predictions"]) == 10 - 4
