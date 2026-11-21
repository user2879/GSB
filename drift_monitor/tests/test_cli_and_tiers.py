import json
from pathlib import Path

from drift_monitor.interactive import Session, replay
from drift_monitor.config import load_config
from drift_monitor.runner import run_pipeline

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default_monitor.yaml"


def test_replay_edit_sequence(tmp_path):
    session = Session(load_config(CONFIG), "small", tmp_path / "sess")
    results = replay(session, [
        "run",
        "set-windows 5",
        "set-threshold 0.4",
        "set-tier standard",
        "set-metric channel freq_diff",
        "set-size-tier medium",
        "run",
        "show",
        "history",
    ])
    assert "history" in results[-1]
    run_result = results[6]
    assert run_result["status"] == "ok"
    assert session.config.window_count == 5
    assert session.config.active_tier == "standard"
    assert session.config.metric_overrides["channel"] == "freq_diff"
    assert session.tier_name == "medium"


def test_invalid_command_returns_business_error(tmp_path):
    session = Session(load_config(CONFIG), "small", tmp_path / "sess")
    results = replay(session, ["set-windows notanint", "run"])
    assert results[0]["status"] == "E_CONFIG_INVALID"


def test_large_size_tier_runs_and_scales(tmp_path):
    config = load_config(CONFIG)
    report = run_pipeline(config, "large", tmp_path / "large", resume=False)
    assert report["tier"] == "large"
    assert len(report["scores_and_judgements"]) > 0
    assert report["strategy_decision"]["indicators"]["n_columns"] > \
        len(config.data.base_columns)
    assert report["snapshot"]["global_version"] == 1


def test_json_report_is_utf8_and_serializable(tmp_path):
    report = run_pipeline(load_config(CONFIG), "small",
                          tmp_path / "json", resume=False)
    encoded = json.dumps(report, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["config_hash"] == report["config_hash"]
    assert decoded["flip_diagnostics"] == report["flip_diagnostics"]


def test_snapshot_manifest_versions_change_with_config(tmp_path):
    work_dir = tmp_path / "snap"
    session = Session(load_config(CONFIG), "small", work_dir)
    session.apply("run")
    first = run_pipeline(session.config, "small", work_dir, resume=False)
    session.apply("set-threshold 0.2")
    second = run_pipeline(session.config, "small", work_dir, resume=False)
    assert second["snapshot"]["global_version"] > first["snapshot"]["global_version"]
    assert second["snapshot"]["config_hash"] != first["snapshot"]["config_hash"]
    # Identical rerun must not bump the version (version gate).
    third = run_pipeline(session.config, "small", work_dir, resume=False)
    assert third["snapshot"]["global_version"] == second["snapshot"]["global_version"]
