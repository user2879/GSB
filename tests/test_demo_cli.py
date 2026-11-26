from __future__ import annotations

import json
from pathlib import Path

from mdet_render.cli import main


def test_validate_accepts_full_sample_command_stream(sample_root: Path, capsys) -> None:
    code = main([
        "validate", "--config", str(sample_root / "config.json"),
        "--commands", str(sample_root / "commands.json"),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["commands_validated"] == 8


def test_lease_demo_cli(capsys) -> None:
    code = main(["lease-demo"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["expectations"]["expiry_really_reclaims"] is True
    assert payload["expectations"]["generation_mismatch_does_not_advance"] is True


def test_end_to_end_demo(tmp_path: Path, capsys) -> None:
    root = tmp_path / "samples"
    code = main(["demo", "--root", str(root)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["comparison"]["all"] is True
    assert payload["cache_keys_match"] is True
    assert payload["lease_ok"] is True
    report_path = Path(payload["demo_report"])
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["final_incremental_ledger"]["outcomes"] == {"reused": 50}
