"""Ledger recovery after forced termination and E_REPLAY verification."""

import subprocess
import sys

from gateway.engine import run
from gateway.errors import ReplayError
from gateway.ledger import Ledger
from gateway.loader import load_declaration
from gateway.replay import recovery_scan, replay_verify

import pytest


def test_crash_recovery_and_replay(sample, tmp_path):
    # Child process really dies after 1 committed batch.
    child = tmp_path / "child.py"
    child.write_text(
        "import sys; sys.path.insert(0, '.')\n"
        "from gateway.engine import run\n"
        "from gateway.ledger import Ledger\n"
        f"run({sample['backends']!r}, {sample['rules']!r}, "
        f"{sample['requests']!r}, Ledger({sample['db']!r}), "
        "global_limit=2, forward_repeat=5, crash_after_commits=1)\n")
    proc = subprocess.run([sys.executable, str(child)],
                          capture_output=True)
    assert proc.returncode == 137

    decl = load_declaration(sample["backends"], sample["rules"],
                            sample["requests"])
    ledger = Ledger(sample["db"])
    version = ledger.current_version()
    scan = recovery_scan(decl, ledger, version)
    assert len(scan["committed"]) == 1
    assert len(scan["unknown"]) >= 1  # unfinished batches marked UNKNOWN
    cursor = scan["cursor"]
    assert cursor["committed_batches"] == 1
    assert len(cursor["content_digest"]) == 16

    # Resume: committed batch is skipped, UNKNOWN ones re-run.
    result = run(sample["backends"], sample["rules"], sample["requests"],
                 ledger, global_limit=2, forward_repeat=5)
    assert result.skipped_committed == scan["committed"]
    report = replay_verify(decl, ledger, result.config_version)
    assert report["status"] == "OK"
    rows = ledger.ledger_rows(result.config_version)
    assert len(rows) == len(result.routed)
    ledger.close()


def test_replay_detects_tampering_e_replay(sample):
    ledger = Ledger(sample["db"])
    result = run(sample["backends"], sample["rules"], sample["requests"],
                 ledger, global_limit=2, forward_repeat=5)
    version = result.config_version
    # Tamper with one committed ledger row.
    ledger.conn.execute(
        "UPDATE ledger SET output_summary = output_summary + 1 "
        "WHERE config_version=? AND request_seq=0", (version,))
    ledger.conn.commit()
    decl = load_declaration(sample["backends"], sample["rules"],
                            sample["requests"])
    with pytest.raises(ReplayError) as exc:
        replay_verify(decl, ledger, version)
    assert exc.value.CODE == "E_REPLAY"
    assert exc.value.request_seq == 0
    assert exc.value.field == "output_summary"
    ledger.close()


def test_ledger_format_version_present(sample):
    ledger = Ledger(sample["db"])
    row = ledger.conn.execute(
        "SELECT value FROM meta WHERE key='format_version'").fetchone()
    ledger.close()
    assert row[0] == "1"
