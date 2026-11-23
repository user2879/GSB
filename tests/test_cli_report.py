from __future__ import annotations

import json
from pathlib import Path

from quant_workbench.cli import main
from quant_workbench.samples import generate_samples


def test_cli_end_to_end_emits_report_and_dashboard(tmp_path: Path, capsys) -> None:
    sample_dir = tmp_path / "samples"
    paths = generate_samples(sample_dir)
    report = tmp_path / "reports" / "diagnostic.json"
    html = tmp_path / "reports" / "dashboard.html"
    snapshot = tmp_path / "snapshot"
    exit_code = main(
        [
            "run",
            "--config",
            str(paths["config"]),
            "--weights",
            str(paths["weights"]),
            "--calibration",
            str(paths["calibration"]),
            "--commands",
            str(paths["commands"]),
            "--snapshot",
            str(snapshot),
            "--report",
            str(report),
            "--html",
            str(html),
        ]
    )
    assert exit_code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["validation"]["incremental_equals_full"] is True
    assert payload["validation"]["snapshot_resume_equals_full"] is True
    assert payload["validation"]["all_commands_within_budget"] is True
    assert payload["validation"]["stale_response_count"] >= 1
    assert payload["validation"]["point_budget_pending_count"] >= 1
    assert payload["validation"]["snapshot_mismatch"]["detected"] is True
    assert payload["validation"]["snapshot_mismatch"]["snapshot_unchanged"] is True
    assert len(payload["stable_result"]["layers"]) == 5
    assert (snapshot / "manifest.json").exists()
    dashboard = html.read_text(encoding="utf-8")
    assert "模型压缩与部署诊断工作台" in dashboard
    assert "逐层误差归因" in dashboard
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert output["completed_layers"] == 5
