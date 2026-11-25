from __future__ import annotations

from pathlib import Path

from edge_ledger.acceptance import run_acceptance
from edge_ledger.replay_cli import main as replay_main


def test_full_local_acceptance_and_cli_replay(tmp_path: Path, capsys):
    report = run_acceptance(tmp_path / "acceptance")
    assert report["ok"] is True
    assert report["adjudication_race"]["double_winners_or_none"] == 0
    assert report["adjudication_race"]["expiry_wins"] + report["adjudication_race"]["quota_wins"] == 24
    assert report["force_kill_recovery_v1"]["unknown_shards"] == ["shard-0000"]
    assert report["force_kill_recovery_v1"]["rerun_digest_matches"] is True
    assert report["quota_evolution"]["long_lease_state"] == "quota_revoked"
    assert "UNKNOWN/RERUN" in Path(report["ui"]["unknown_text_snapshot"]).read_text()

    code = replay_main(["--spec-dir", report["spec_dir"].replace("spec-v1", "spec-v2"), "--ledger", report["ledger"]])
    assert code == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out
