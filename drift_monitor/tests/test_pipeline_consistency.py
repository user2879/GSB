import json
from dataclasses import replace
from pathlib import Path

import pytest

from drift_monitor.config import load_config
from drift_monitor.interactive import Session
from drift_monitor.runner import run_pipeline

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default_monitor.yaml"

EDIT_SEQUENCE = [
    "run",
    "set-windows 6",
    "run",
    "set-threshold 0.5",
    "run",
    "set-tier coarse",
    "run",
    "set-tier fine",
    "set-metric age mean_shift",
    "run",
    "set-size-tier medium",
    "run",
    "set-concurrency 8",
    "run",
]


def _comparable(report: dict) -> dict:
    return {
        "scores": [
            (row["column"], row["window_index"], row["metric"],
             row["raw_score"],
             tuple((t["tier"], t["score_q"], t["significant"])
                   for t in row["tiers"]))
            for row in report["scores_and_judgements"]
        ],
        "flips": [
            (f["column"], f["window_index"], f["lower_tier"], f["higher_tier"],
             f["flipped"], f["first_breach_stage"], f["lower_score_q"],
             f["higher_score_q"])
            for f in report["flip_diagnostics"]
        ],
        "strategy": report["strategy_decision"],
        "failures": report["failure_list"],
        "windows": report["windows"],
        "data_fingerprint": report["data_fingerprint"],
    }


def test_incremental_edits_match_clean_full_rebuild(tmp_path):
    incremental_dir = tmp_path / "inc"
    session = Session(load_config(CONFIG), "small", incremental_dir)
    for command in EDIT_SEQUENCE:
        result = session.apply(command)
        assert result.get("status") != "E_CONFIG_INVALID", result
    incremental_report = run_pipeline(session.config, session.tier_name,
                                      incremental_dir, resume=False)

    full_dir = tmp_path / "full"
    full_report = run_pipeline(session.config, session.tier_name,
                               full_dir, resume=False)
    assert _comparable(incremental_report) == _comparable(full_report)
    # Cache keys are content derived and independent of run history.
    assert incremental_report["incremental"]["required_keys"] == \
        full_report["incremental"]["required_keys"]
    assert incremental_report["budget_deduction_sequence"] == \
        full_report["budget_deduction_sequence"]


def test_reuse_set_equals_recomputed_set_on_clean_missing_keys(tmp_path):
    work_dir = tmp_path / "work"
    session = Session(load_config(CONFIG), "small", work_dir)
    session.apply("run")
    first = run_pipeline(session.config, "small", work_dir, resume=False)
    second = run_pipeline(session.config, "small", work_dir, resume=False)
    # Same required key universe regardless of cache history.
    assert first["incremental"]["required_keys"] == \
        second["incremental"]["required_keys"]
    # Second run physically reuses everything and recomputes nothing.
    assert second["incremental"]["recomputed_count"] == 0
    assert second["incremental"]["reused_count"] > 0


def test_partial_invalidation_only_affected_columns(tmp_path):
    work_dir = tmp_path / "work"
    session = Session(load_config(CONFIG), "small", work_dir)
    session.apply("run")
    session.apply("set-metric income hist_density")
    after = run_pipeline(session.config, "small", work_dir, resume=False)
    recomputed_metric_keys = {
        key for key in after["incremental"]["recomputed_keys"]
        if key.startswith("metric:")
    }
    # Four windows of the single changed column are recomputed.
    assert len(recomputed_metric_keys) == 4
    reused_count = sum(
        1 for key in after["incremental"]["reused_keys"]
        if key.startswith("metric:")
    )
    assert reused_count == 20


def test_concurrency_levels_produce_identical_results(tmp_path):
    reports = {}
    for workers in (1, 2, 4, 8):
        config = replace(load_config(CONFIG), concurrency=workers)
        report = run_pipeline(config, "small", tmp_path / f"c{workers}", resume=False)
        reports[workers] = report
    baseline = _comparable(reports[1])
    for workers in (2, 4, 8):
        assert _comparable(reports[workers]) == baseline
        assert reports[workers]["budget_deduction_sequence"] == \
            reports[1]["budget_deduction_sequence"]
        assert reports[workers]["failure_list"] == reports[1]["failure_list"]


def test_failure_list_is_deterministic_and_sorted(tmp_path):
    config = replace(load_config(CONFIG),
                     fail_keys=("city_tier#w1", "income#w0"))
    report = run_pipeline(config, "small", tmp_path / "fail", resume=False)
    failures = report["failure_list"]
    assert [f["sort_key"] for f in failures] == sorted(f["sort_key"] for f in failures)
    assert {f["code"] for f in failures} == {"E_METRIC_FAILED"}
    report_again = run_pipeline(config, "small", tmp_path / "fail2", resume=False)
    assert report_again["failure_list"] == failures


def test_snapshot_resume_and_stale_detection(tmp_path):
    work_dir = tmp_path / "snap"
    session = Session(load_config(CONFIG), "small", work_dir)
    session.apply("run")
    healthy = run_pipeline(session.config, "small", work_dir, resume=True)
    assert healthy["snapshot"]["resume_status"] == "resumed"

    session.apply("set-threshold 0.33")
    stale = run_pipeline(session.config, "small", work_dir, resume=True)
    assert stale["snapshot"]["resume_status"] == "E_SNAPSHOT_STALE"
    manifest_path = work_dir / "snapshot" / "current" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["global_version"] >= 1
    assert "stage_versions" in manifest


def test_snapshot_uses_atomic_replacement_no_partial_dirs(tmp_path):
    work_dir = tmp_path / "snap"
    config = load_config(CONFIG)
    run_pipeline(config, "small", work_dir, resume=False)
    current = work_dir / "snapshot" / "current"
    assert (current / "manifest.json").exists()
    assert (current / "stages" / "report" / "artifact.json").exists()
    leftover = list((work_dir / "snapshot" / "tmp").glob("snap_*"))
    assert leftover == []


def test_report_contains_all_required_sections(tmp_path):
    report = run_pipeline(load_config(CONFIG), "small",
                          tmp_path / "rep", resume=False)
    for section in ("scores_and_judgements", "flip_diagnostics",
                    "incremental", "failure_list", "strategy_decision",
                    "snapshot", "budget_deduction_sequence"):
        assert section in report
    svg = tmp_path / "rep" / "reports" / "heatmap.svg"
    assert svg.exists() and svg.read_text().startswith("<svg")


def test_concurrency_never_exceeds_configured_value(tmp_path, monkeypatch):
    import drift_monitor.engine as engine_mod
    seen = {}
    real_pool = engine_mod.ThreadPoolExecutor

    class RecordingPool(real_pool):
        def __init__(self, max_workers=None, *args, **kwargs):
            seen["max_workers"] = max_workers
            super().__init__(max_workers, *args, **kwargs)

    monkeypatch.setattr(engine_mod, "ThreadPoolExecutor", RecordingPool)
    config = replace(load_config(CONFIG), concurrency=3)
    run_pipeline(config, "small", tmp_path / "cap", resume=False)
    assert seen["max_workers"] <= 3
