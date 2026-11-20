from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ckptmeta.catalog import scan_catalog
from ckptmeta.formats import decode_manifest
from ckptmeta.samples import generate_fixtures


ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str], *, stdin: str | None = None):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        [sys.executable, "-m", "ckptmeta", *arguments],
        cwd=ROOT,
        env=environment,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("flag", "expected_code", "expected_exit"),
    [
        ("--inject-integrity", "E_INTEGRITY", 12),
        ("--inject-chain", "E_CHAIN", 13),
        ("--inject-upgrade", "E_UPGRADE", 14),
    ],
)
def test_cli_injections_write_logs_and_exit_nonzero(tmp_path: Path, flag: str, expected_code: str, expected_exit: int):
    workspace = tmp_path / flag.removeprefix("--inject-")
    result = _run(["demo", "--workspace", str(workspace), flag])
    assert result.returncode == expected_exit, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["business_code"] == expected_code
    assert payload["injected_input_only"] is True
    log_path = Path(payload["log_file"])
    assert log_path.exists()
    event = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert event["business_code"] == expected_code
    assert "manifest_path" in event and "shard_index" in event and "timestamp" in event


def test_cli_injection_leaves_complete_manifests_decodable(tmp_path: Path):
    workspace = tmp_path / "upgrade"
    result = _run(["demo", "--workspace", str(workspace), "--inject-upgrade"])
    assert result.returncode == 14
    for manifest in scan_catalog(workspace / "crash_catalog"):
        assert decode_manifest(manifest.source_path).version == 3


def test_cli_normalize_expected_manifest_reports_restore_error(tmp_path: Path):
    fixtures = generate_fixtures(tmp_path / "restore")
    expected = tmp_path / "expected-v3.json"
    raw = json.loads(fixtures.direct_target.read_text(encoding="utf-8"))
    raw["shards"][2]["parameter_digest"] = "f" * 64
    expected.write_text(json.dumps(raw), encoding="utf-8")
    result = _run(
        [
            "normalize",
            "--manifest",
            str(fixtures.v1_target),
            "--ledger",
            str(fixtures.ledger),
            "--expected-manifest",
            str(expected),
        ]
    )
    assert result.returncode == 15
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["business_code"] == "E_RESTORE"
    assert payload["shard_index"] == 2


def test_cli_demo_and_session(tmp_path: Path):
    workspace = tmp_path / "demo"
    demo = _run(["demo", "--workspace", str(workspace)])
    assert demo.returncode == 0, demo.stderr
    summary = json.loads(demo.stdout.strip().splitlines()[-1])
    assert summary["summary"]["all_checks_passed"] is True
    assert summary["scenario_3_forced_termination_resume"]["forced_exit_code"] != 0
    assert summary["scenario_3_forced_termination_resume"]["interrupted_run_exit_report"] == "unavailable"

    fixtures = generate_fixtures(tmp_path / "session-fixtures")
    requests = [
        {"command": "scan", "request_id": "scan"},
        {"command": "open", "request_id": "open"},
        {
            "command": "normalize",
            "manifest": str(fixtures.direct_target),
            "request_id": "normalize",
        },
        {
            "command": "snapshot_save",
            "snapshot_db": str(tmp_path / "session.sqlite3"),
            "request_id": "save",
        },
        {
            "command": "snapshot_replay",
            "snapshot_db": str(tmp_path / "session.sqlite3"),
            "request_id": "replay",
        },
        {"command": "quit", "request_id": "quit"},
    ]
    stdin = "\n".join(json.dumps(request) for request in requests) + "\n"
    session = _run(
        [
            "session",
            "--catalog",
            str(fixtures.direct),
            "--config",
            str(fixtures.config),
            "--ledger",
            str(fixtures.ledger),
            "--snapshot-db",
            str(tmp_path / "session.sqlite3"),
        ],
        stdin=stdin,
    )
    assert session.returncode == 0, session.stderr
    lines = [json.loads(line) for line in session.stdout.strip().splitlines()]
    assert [line["request_id"] for line in lines] == ["scan", "open", "normalize", "save", "replay", "quit"]
    assert all(line["ok"] is True for line in lines)
    assert lines[4]["reproduced"] is True
