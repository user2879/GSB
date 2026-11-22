from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "snapdiag", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_cli_init_snapshot_write_show_and_injection(tmp_path: Path) -> None:
    session = tmp_path / "cli-session"
    initialized = _run(
        "init",
        "--session",
        str(session),
        "--block-size",
        "512",
        "--block-count",
        "8",
        "--threshold",
        "0.5",
    )
    assert initialized.returncode == 0, initialized.stderr
    payload = json.loads(initialized.stdout)
    assert payload["ok"] and payload["block_count"] == 8
    created = _run("create-snapshot", "--session", str(session), "--snapshot-id", "s0")
    assert created.returncode == 0
    written = _run(
        "write-block",
        "--session",
        str(session),
        "--block",
        "1",
        "--hex",
        "ab" * 512,
    )
    assert written.returncode == 0
    shown = _run("show", "--session", str(session), "--target", "s0")
    assert shown.returncode == 0
    view = json.loads(shown.stdout)["view"]
    assert view["snapshots"][0]["dirty_blocks"] == 1
    injected = _run(
        "diagnose-injection",
        "--session",
        str(session),
        "--inject",
        "version-conflict",
    )
    assert injected.returncode == 0
    diagnosis = json.loads(injected.stdout)
    assert diagnosis["diagnosis"]["observed_code"] == "E_VERSION"


def test_cli_geometry_error_has_exit_code_and_first_field(tmp_path: Path) -> None:
    result = _run(
        "init",
        "--session",
        str(tmp_path / "bad"),
        "--block-size",
        "777",
        "--block-count",
        "8",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["code"] == "E_GEOM"
    assert payload["first_violation"]["field"] == "block_size"

