from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from audit_forensics.engine import AuditEngine
from audit_forensics.rules import RuleConfig
from audit_forensics.storage import make_envelope, atomic_write_json
from audit_forensics.canonical import canonical_json
from helpers import (
    BASE_TS,
    ROOT,
    case_signature,
    codes,
    event,
    import_workspace,
    parse_stdout,
    rules_config,
    run_cli,
    write_rules,
    write_v1,
)


def _clean_setup(tmp_path):
    events = [
        event("e1", trace_id="T-1", seq=1, ts=BASE_TS, severity="high"),
        event(
            "e2",
            trace_id="T-1",
            seq=2,
            ts=BASE_TS + 1_000,
            parent_event_id="e1",
            severity="critical",
        ),
    ]
    rules = rules_config(
        "v1",
        [
            {
                "rule_id": "R-1",
                "conditions": {"actors": ["alice"]},
                "group_by": ["actor"],
            }
        ],
    )
    source = write_v1(tmp_path / "events.jsonl", events)
    rules_path = write_rules(tmp_path / "rules.json", rules)
    return source, rules_path, events, rules


def test_cli_exit_codes_query_export_and_stable_field_order(tmp_path):
    source, rules_path, _, _ = _clean_setup(tmp_path)
    workspace = tmp_path / "workspace"
    imported = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--rules",
        str(rules_path),
        "--input",
        str(source),
    )
    assert imported.returncode == 0, imported.stderr
    report = parse_stdout(imported)
    assert report["status"] == "ok"
    assert report["cases"]

    case_id = report["cases"][0]["case_id"]
    queried = run_cli(
        "query-case", "--workspace", str(workspace), "--case-id", case_id
    )
    assert queried.returncode == 0
    assert parse_stdout(queried)["case"]["case_id"] == case_id

    missing = run_cli(
        "query-case", "--workspace", str(workspace), "--case-id", "CASE-MISSING"
    )
    assert missing.returncode == 1
    assert codes(parse_stdout(missing)) == {"E_CASE_NOT_FOUND"}

    out1 = tmp_path / "report-1.json"
    out2 = tmp_path / "report-2.json"
    exported1 = run_cli(
        "export-report",
        "--workspace",
        str(workspace),
        "--output",
        str(out1),
        "--graph-output",
        str(tmp_path / "graph-1.json"),
    )
    exported2 = run_cli(
        "export-report",
        "--workspace",
        str(workspace),
        "--output",
        str(out2),
        "--graph-output",
        str(tmp_path / "graph-2.json"),
    )
    assert exported1.returncode == 0
    assert exported2.returncode == 0
    assert out1.read_bytes() == out2.read_bytes()

    verified = run_cli("verify-checkpoint", "--workspace", str(workspace))
    assert verified.returncode == 0
    assert parse_stdout(verified)["valid"] is True

    parameter_error = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--workers",
        "0",
        "--input",
        str(source),
    )
    assert parameter_error.returncode == 2


def test_checkpoint_schema_change_forces_rebuild(tmp_path):
    source, rules_path, events, _ = _clean_setup(tmp_path)
    workspace = tmp_path / "workspace"
    engine = AuditEngine(workspace, rules=RuleConfig.from_dict(json.loads(rules_path.read_text())))
    first = engine.import_streams([source])
    assert first.exit_code == 0
    checkpoint_path = workspace / ".audit" / "checkpoint.json"
    envelope = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    envelope["schema_version"] = "audit-checkpoint.v0"
    checkpoint_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")

    invalid = run_cli("verify-checkpoint", "--workspace", str(workspace))
    assert invalid.returncode == 1
    assert "E_STATE_SCHEMA_MISMATCH" in codes(parse_stdout(invalid))

    rebuilt = engine.import_streams([])
    report = rebuilt.payload
    assert report["rebuild"]["reason"] == "checkpoint_schema_changed"
    assert any(
        notice["code"] == "E_STATE_SCHEMA_MISMATCH" for notice in report["notices"]
    )
    assert len(report["events"]) == len(events)
    rewritten = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert rewritten["schema_version"] == "audit-checkpoint.v1"
    assert rewritten["payload"]["generation"] == first.payload["checkpoint"]["generation"] + 1


def test_stale_case_snapshot_is_detected_and_not_loaded(tmp_path):
    source, rules_path, _, _ = _clean_setup(tmp_path)
    workspace = tmp_path / "workspace"
    imported = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--rules",
        str(rules_path),
        "--input",
        str(source),
    )
    assert imported.returncode == 0
    snapshot_path = workspace / ".audit" / "case_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["payload"]["generation"] += 99
    atomic_write_json(
        snapshot_path,
        make_envelope("audit-case-snapshot.v1", snapshot["payload"]),
    )

    verified = run_cli("verify-checkpoint", "--workspace", str(workspace))
    assert verified.returncode == 1
    assert "E_SNAPSHOT_STALE" in codes(parse_stdout(verified))

    queried = run_cli("query-case", "--workspace", str(workspace))
    assert queried.returncode == 0
    assert parse_stdout(queried)["case_count"] >= 1


def test_forced_termination_before_checkpoint_replace_recovers(tmp_path):
    first_batch, rules_path, _, _ = _clean_setup(tmp_path)
    workspace = tmp_path / "workspace"
    initial = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--rules",
        str(rules_path),
        "--input",
        str(first_batch),
    )
    assert initial.returncode == 0
    initial_generation = parse_stdout(initial)["checkpoint"]["generation"]
    second_batch = write_v1(
        tmp_path / "second.jsonl",
        [
            event(
                "e3",
                trace_id="T-1",
                seq=3,
                ts=BASE_TS + 2_000,
                parent_event_id="e2",
                severity="high",
            )
        ],
    )
    ready = tmp_path / "ready-before-checkpoint.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        {
            "AUDIT_FORENSICS_TEST_PAUSE_AT": "checkpoint",
            "AUDIT_FORENSICS_TEST_PAUSE_SECONDS": "30",
            "AUDIT_FORENSICS_TEST_READY_FILE": str(ready),
        }
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "audit_forensics",
            "import",
            "--workspace",
            str(workspace),
            "--rules",
            str(rules_path),
            "--input",
            str(second_batch),
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for(ready)
    child.terminate()
    child.wait(timeout=10)

    verified = run_cli("verify-checkpoint", "--workspace", str(workspace))
    assert verified.returncode == 0
    checkpoint = parse_stdout(verified)["checkpoint"]
    assert checkpoint["generation"] == initial_generation
    queried = parse_stdout(
        run_cli("query-case", "--workspace", str(workspace))
    )
    assert queried["case_count"] == 1
    assert "e3" not in {
        member
        for case in queried["cases"]
        for member in case["member_event_ids"]
    }

    recovered = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--rules",
        str(rules_path),
        "--input",
        str(second_batch),
    )
    assert recovered.returncode == 0
    recovered_report = parse_stdout(recovered)
    assert recovered_report["checkpoint"]["generation"] == initial_generation + 1
    assert "e3" in {
        item["event"]["event_id"] for item in recovered_report["events"]
    }
    assert not list((workspace / ".audit").glob(".*.tmp"))


def test_forced_termination_after_checkpoint_leaves_stale_snapshot(tmp_path):
    first_batch, rules_path, _, _ = _clean_setup(tmp_path)
    workspace = tmp_path / "workspace"
    initial = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--rules",
        str(rules_path),
        "--input",
        str(first_batch),
    )
    assert initial.returncode == 0
    initial_generation = parse_stdout(initial)["checkpoint"]["generation"]
    second_batch = write_v1(
        tmp_path / "second.jsonl",
        [event("e3", trace_id="T-1", seq=3, ts=BASE_TS + 2_000, parent_event_id="e2")],
    )
    ready = tmp_path / "ready-before-snapshot.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        {
            "AUDIT_FORENSICS_TEST_PAUSE_AT": "case_snapshot",
            "AUDIT_FORENSICS_TEST_PAUSE_SECONDS": "30",
            "AUDIT_FORENSICS_TEST_READY_FILE": str(ready),
        }
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "audit_forensics",
            "import",
            "--workspace",
            str(workspace),
            "--rules",
            str(rules_path),
            "--input",
            str(second_batch),
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for(ready)
    child.terminate()
    child.wait(timeout=10)

    verified = run_cli("verify-checkpoint", "--workspace", str(workspace))
    assert verified.returncode == 1
    verify_payload = parse_stdout(verified)
    assert verify_payload["checkpoint"]["generation"] == initial_generation + 1
    assert "E_SNAPSHOT_STALE" in codes(verify_payload)

    queried = run_cli("query-case", "--workspace", str(workspace))
    assert queried.returncode == 0
    members = {
        member
        for case in parse_stdout(queried)["cases"]
        for member in case["member_event_ids"]
    }
    assert "e3" in members


def test_lock_conflict_returns_e_lock(tmp_path):
    source, rules_path, _, _ = _clean_setup(tmp_path)
    workspace = tmp_path / "workspace"
    ready = tmp_path / "lock-ready"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        {
            "AUDIT_FORENSICS_TEST_HOLD_LOCK_SECONDS": "2",
            "AUDIT_FORENSICS_TEST_LOCK_READY_FILE": str(ready),
        }
    )
    holder = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "audit_forensics",
            "import",
            "--workspace",
            str(workspace),
            "--rules",
            str(rules_path),
            "--input",
            str(source),
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for(ready)
    conflict = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--rules",
        str(rules_path),
        "--input",
        str(source),
    )
    assert conflict.returncode == 1
    assert codes(parse_stdout(conflict)) == {"E_LOCK"}
    stdout, stderr = holder.communicate(timeout=15)
    assert holder.returncode == 0, stderr


def test_corrupt_current_checkpoint_recovers_previous_complete_checkpoint(tmp_path):
    first_batch, rules_path, _, _ = _clean_setup(tmp_path)
    workspace = tmp_path / "workspace"
    initial = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--rules",
        str(rules_path),
        "--input",
        str(first_batch),
    )
    assert initial.returncode == 0
    second_batch = write_v1(
        tmp_path / "second.jsonl",
        [event("e3", trace_id="T-1", seq=3, ts=BASE_TS + 2_000, parent_event_id="e2")],
    )
    second = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--rules",
        str(rules_path),
        "--input",
        str(second_batch),
    )
    assert second.returncode == 0
    previous_generation = parse_stdout(initial)["checkpoint"]["generation"]
    checkpoint_path = workspace / ".audit" / "checkpoint.json"
    checkpoint_path.write_text('{"half-written":', encoding="utf-8")

    verified = run_cli("verify-checkpoint", "--workspace", str(workspace))
    assert verified.returncode == 1
    payload = parse_stdout(verified)
    assert payload["valid"] is False
    assert payload["checkpoint"]["recovered_from_previous"] is True
    assert payload["checkpoint"]["generation"] == previous_generation
    assert "E_SNAPSHOT_STALE" in codes(payload)
    assert any(
        notice["code"] == "E_CHECKPOINT_CORRUPT" for notice in payload["notices"]
    )

    healed = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--rules",
        str(rules_path),
    )
    assert healed.returncode == 0
    assert run_cli("verify-checkpoint", "--workspace", str(workspace)).returncode == 0


def _wait_for(path: Path, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")
