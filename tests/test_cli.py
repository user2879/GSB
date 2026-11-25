from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .helpers import ROOT, SAMPLE_CONFIG, SAMPLE_MATERIAL, SAMPLE_TEXTURE, write_config


def _run_cli(args: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "fixedtex", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )


def test_validate_cli_returns_business_exit_code_and_valid_items(tmp_path: Path) -> None:
    raw = json.loads(SAMPLE_CONFIG.read_text(encoding="utf-8"))
    raw["texture"] = str(SAMPLE_TEXTURE)
    raw["noise"]["layers"][0]["frequency"] = -2
    bad_config = tmp_path / "bad.json"
    bad_config.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    result = _run_cli(
        ["validate", "--config", str(bad_config), "--material", str(SAMPLE_MATERIAL)]
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["exit_code"] == 2
    assert any(item["code"] == "E_FREQ" for item in report["errors"])
    assert report["valid_items"]["config"]["layer_count"] == 5
    assert report["valid_items"]["texture"] is not None


def test_run_cli_applies_commands_and_writes_output(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    result = _run_cli(
        [
            "run",
            "--config",
            str(SAMPLE_CONFIG),
            "--material",
            str(SAMPLE_MATERIAL),
            "--commands",
            str(ROOT / "samples/commands.jsonl"),
            "--state-dir",
            str(state_dir),
            "--fresh",
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["final_summary"]["complete"] is True
    assert report["status"]["version"] >= 1
    assert (state_dir / "output.png").exists()
    assert (state_dir / "ledger.jsonl").exists()
    assert (state_dir / "CURRENT").exists()
    assert len(report["command_results"]) == 10


def test_run_cli_no_drain_marks_point_budget_pending(tmp_path: Path) -> None:
    config = write_config(tmp_path, source=ROOT / "samples/config_low_budget.json", name="cli_low")
    commands = tmp_path / "one.jsonl"
    commands.write_text('{"command":"render"}\n', encoding="utf-8")
    state_dir = tmp_path / "low_state"
    result = _run_cli(
        [
            "run",
            "--config",
            str(config),
            "--material",
            str(SAMPLE_MATERIAL),
            "--commands",
            str(commands),
            "--state-dir",
            str(state_dir),
            "--fresh",
            "--no-drain",
            "--no-commit",
        ]
    )
    assert result.returncode == 3
    report = json.loads(result.stdout)
    assert report["pending"]
    assert any(item["code"] == "E_POINT_BUDGET_PENDING" for item in report["pending"])


def test_verify_cli_runs_all_acceptance_checks(tmp_path: Path) -> None:
    output_dir = tmp_path / "verification"
    result = _run_cli(
        [
            "verify",
            "--config",
            str(SAMPLE_CONFIG),
            "--material",
            str(SAMPLE_MATERIAL),
            "--commands",
            str(ROOT / "samples/commands.jsonl"),
            "--injections",
            str(ROOT / "samples/injections.json"),
            "--low-budget-config",
            str(ROOT / "samples/config_low_budget.json"),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["incremental_vs_full_render"]["ok"] is True
    assert checks["incremental_vs_full_render"]["png_pixels_equal"] is True
    assert checks["partial_snapshot_recovery"]["ok"] is True
    assert checks["partial_snapshot_recovery"]["pending_count_at_snapshot"] > 0
    assert checks["fault_isolation"]["ok"] is True
    assert set(checks["fault_isolation"]["expected_codes"]) <= set(
        checks["fault_isolation"]["injection_hit_codes"]
    )
    assert (output_dir / "report.json").exists()
    assert (output_dir / "incremental.png").exists()
    assert (output_dir / "full_rebuild.png").exists()
    assert (output_dir / "diagnostics" / "stage_diffs.npz").stat().st_size > 0
