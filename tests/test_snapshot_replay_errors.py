from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from columnar_pipeline.errors import PipelineError
from columnar_pipeline.model import RunConfig
from columnar_pipeline.runner import replay_snapshot, resume_snapshot, run_pipeline


def _config(version: int = 1, **overrides) -> RunConfig:
    values = {
        "parallelism": 2,
        "memory_budget_bytes": 2048,
        "budget_version": version,
        "split_threshold": 24,
        "split_parts": 2,
        "row_group_size": 16,
        "ledger_batch_size": 2,
        "skew_safety_limit": 100000,
    }
    values.update(overrides)
    return RunConfig(**values)


def _run_small(tmp_path: Path, data_path: Path, version: int = 1):
    return run_pipeline(
        data_path,
        _config(version),
        out_dir=tmp_path / "run",
        ledger_path=tmp_path / "ledger.sqlite",
        snapshot_dir=tmp_path / "snapshots",
        log_path=tmp_path / "pipeline.jsonl",
    )


def test_resume_and_strict_replay(tmp_path: Path, small_data: Path) -> None:
    summary = _run_small(tmp_path, small_data)
    replayed = replay_snapshot(
        summary["snapshot_path"],
        data_path=small_data,
        ledger_path=tmp_path / "ledger.sqlite",
        out_dir=tmp_path / "replay",
        log_path=tmp_path / "pipeline.jsonl",
    )
    assert replayed["result_digest"] == summary["result_digest"]
    assert replayed["replayed_tasks"] == summary["task_count"]
    resumed = resume_snapshot(
        summary["snapshot_path"],
        data_path=small_data,
        ledger_path=tmp_path / "ledger.sqlite",
        snapshot_dir=tmp_path / "snapshots",
        out_dir=tmp_path / "resume",
        log_path=tmp_path / "pipeline.jsonl",
    )
    assert resumed["restored"] is True
    assert resumed["replaced"] is False
    assert resumed["result_digest"] == summary["result_digest"]


def test_snapshot_mismatch_replaces_all_state(tmp_path: Path, small_data: Path) -> None:
    summary = _run_small(tmp_path, small_data)
    document = json.loads(small_data.read_text(encoding="utf-8"))
    document["tables"][0]["version"] = 9
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    replacement = resume_snapshot(
        summary["snapshot_path"],
        data_path=changed,
        ledger_path=tmp_path / "ledger.sqlite",
        snapshot_dir=tmp_path / "snapshots",
        out_dir=tmp_path / "replacement",
        log_path=tmp_path / "pipeline.jsonl",
    )
    assert replacement["replaced"] is True
    assert replacement["restored"] is False
    assert replacement["run_id"] != summary["run_id"]
    assert "data_digest" in replacement["replacement_reasons"]
    with sqlite3.connect(tmp_path / "ledger.sqlite") as connection:
        old_count = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE run_id = ?", (summary["run_id"],)
        ).fetchone()[0]
        new_count = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE run_id = ?", (replacement["run_id"],)
        ).fetchone()[0]
        replacement_rows = connection.execute(
            "SELECT COUNT(*) FROM snapshot_replacements WHERE old_run_id = ? AND new_run_id = ?",
            (summary["run_id"], replacement["run_id"]),
        ).fetchone()[0]
    assert old_count > 0
    assert new_count > 0
    assert replacement_rows == 1
    assert old_count == new_count


def test_replay_reports_first_mismatched_shard_and_field(
    tmp_path: Path,
    small_data: Path,
) -> None:
    summary = _run_small(tmp_path, small_data)
    with sqlite3.connect(tmp_path / "ledger.sqlite") as connection:
        row = connection.execute(
            "SELECT task_id, rows_completed FROM tasks WHERE run_id = ? ORDER BY replay_order LIMIT 1",
            (summary["run_id"],),
        ).fetchone()
        task_id, rows_completed = row
        connection.execute(
            "UPDATE tasks SET rows_completed = ? WHERE run_id = ? AND task_id = ?",
            (rows_completed + 1, summary["run_id"], task_id),
        )
        connection.commit()
    with pytest.raises(PipelineError) as captured:
        replay_snapshot(
            summary["snapshot_path"],
            data_path=small_data,
            ledger_path=tmp_path / "ledger.sqlite",
            out_dir=tmp_path / "bad_replay",
            log_path=tmp_path / "pipeline.jsonl",
        )
    error = captured.value
    assert error.code == "E_REPLAY"
    assert error.task_id == task_id
    assert error.field_name == "rows_completed"
    assert error.details["expected"] == rows_completed + 1
    assert error.details["actual"] == rows_completed


def test_three_anomaly_business_codes(
    tmp_path: Path,
    small_data: Path,
    skew_data: Path,
    params: Path,
) -> None:
    cases = [
        ("skew", skew_data, "inject_skew.json", "E_SKEW"),
        ("budget", small_data, "inject_budget.json", "E_BUDGET"),
        ("digest", small_data, "inject_digest.json", "E_DIGEST"),
    ]
    observed: list[str] = []
    for name, data_path, config_name, expected_code in cases:
        config = RunConfig.from_json_dict(
            json.loads((params / config_name).read_text(encoding="utf-8"))
        )
        with pytest.raises(PipelineError) as captured:
            run_pipeline(
                data_path,
                config,
                out_dir=tmp_path / name,
                ledger_path=tmp_path / "anomaly_ledger.sqlite",
                snapshot_dir=tmp_path / "anomaly_snapshots",
                log_path=tmp_path / "anomaly.jsonl",
            )
        assert captured.value.code == expected_code
        observed.append(captured.value.code)
    assert observed == ["E_SKEW", "E_BUDGET", "E_DIGEST"]
    log_codes = {
        json.loads(line)["code"]
        for line in (tmp_path / "anomaly.jsonl").read_text(encoding="utf-8").splitlines()
        if "code" in json.loads(line)
    }
    assert {"E_SKEW", "E_BUDGET", "E_DIGEST"}.issubset(log_codes)


def test_environment_has_numpy_and_sqlite() -> None:
    import numpy
    import sqlite3

    assert numpy.__version__
    assert sqlite3.sqlite_version
