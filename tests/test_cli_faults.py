from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ossdiag.events import read_jsonl

from conftest import KiB, object_doc, spec_doc, write_spec


ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    spec = write_spec(tmp_path / "spec.json", spec_doc(objects=[object_doc(total_bytes=128 * KiB)]))
    command = [
        sys.executable,
        "-m",
        "ossdiag",
        "run",
        "--spec",
        str(spec),
        "--workdir",
        str(tmp_path / "work"),
        *extra,
    ]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def _assert_business_log(tmp_path: Path, code: str) -> None:
    records = read_jsonl(tmp_path / "work" / "events.jsonl")
    matches = [item for item in records if item.get("code") == code]
    assert matches
    assert all("monotonic_ms" in item for item in matches)
    assert matches[-1].get("object_key") == "object/a.bin"
    assert "shard_index" in matches[-1]


def test_cli_lease_overflow_exit_code_and_log(tmp_path) -> None:
    completed = _run(tmp_path, "--inject-lease-overflow")
    assert completed.returncode == 3, completed.stderr + completed.stdout
    _assert_business_log(tmp_path, "E_LEASE")


def test_cli_digest_mismatch_exit_code_and_log(tmp_path) -> None:
    completed = _run(tmp_path, "--inject-digest-mismatch")
    assert completed.returncode == 3, completed.stderr + completed.stdout
    _assert_business_log(tmp_path, "E_DIGEST")
    summary = json.loads((tmp_path / "work" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is False
    assert any(item["code"] == "E_DIGEST" for item in summary["errors"])


def test_cli_snapshot_conflict_exit_code_and_log(tmp_path) -> None:
    first = _run(tmp_path)
    assert first.returncode == 0, first.stderr + first.stdout
    second = _run(tmp_path, "--inject-snapshot-conflict")
    assert second.returncode == 3, second.stderr + second.stdout
    _assert_business_log(tmp_path, "E_SNAPSHOT")
    summary = json.loads((tmp_path / "work" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["reconcile"]["replaced"] is True
    assert summary["reconcile"]["backup_path"]


def test_cli_spec_error_is_e_spec_and_exit_2(tmp_path) -> None:
    bad = spec_doc(objects=[object_doc(total_bytes=0)])
    spec = write_spec(tmp_path / "bad.json", bad)
    completed = subprocess.run(
        [sys.executable, "-m", "ossdiag", "run", "--spec", str(spec), "--workdir", str(tmp_path / "work")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "E_SPEC" in completed.stderr
    records = read_jsonl(tmp_path / "work" / "events.jsonl")
    assert any(item.get("code") == "E_SPEC" and item.get("object_key") == "object/a.bin" for item in records)
