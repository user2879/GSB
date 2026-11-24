"""Complete repeatable local demonstration and acceptance report."""

from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .declarations import load_declarations
from .engine import LeaseOrchestrator
from .errors import ProbeLeaseError
from .ledger import Ledger, restore_from_snapshot
from .logging import StructuredLogger
from .replay import compare_splice_ledgers, replay_verify


def run_demo(root: str | os.PathLike[str] = "artifacts/results/demo") -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    repo_root = Path(__file__).resolve().parent.parent
    samples = repo_root / "samples" / "config"
    output = Path(root).resolve()
    if output == Path(output.anchor) or output == repo_root or (output / "pyproject.toml").exists():
        raise ValueError("demo output directory must be a disposable subdirectory")
    _remove_tree(output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "demo.jsonl"
    logger = StructuredLogger(log_path)
    declarations = load_declarations(
        samples / "sessions.json",
        samples / "probes.json",
        samples / "ack_policy.json",
    )
    _wait_ports_free(declarations, timeout_s=5.0)

    report: dict[str, Any] = {
        "code": "OK",
        "declarations": {
            "format_version": declarations.format_version,
            "session_set_version": declarations.session_set_version,
            "probe_parameter_version": declarations.probe_parameter_version,
            "ack_policy_version": declarations.ack_policy_version,
            "content_digest": declarations.content_digest,
        },
        "artifacts": {"root": str(output), "log": str(log_path)},
    }

    report["lease_acceptance"] = _demo_leases(output / "lease", declarations, logger)
    _wait_ports_free(declarations)
    report["inflight_pause"] = _demo_inflight(output / "inflight", declarations, logger)
    _wait_ports_free(declarations)
    report["splice"] = _demo_splice(output / "splice", declarations, logger)
    _wait_ports_free(declarations)
    report["crash_recovery"] = _demo_crash(output / "crash", declarations, logger, repo_root, samples)
    _wait_ports_free(declarations)
    report["replay"] = _demo_replay(output, declarations, logger)
    _wait_ports_free(declarations)
    report["injections"] = _demo_injections(repo_root, samples, logger)
    report["negative_verifications"] = _demo_negative_checks(output, declarations, logger)
    _wait_ports_free(declarations)

    report["duration_monotonic_ns"] = time.monotonic_ns() - started_ns
    report["code"] = "OK"
    report_path = output / "demo_report.json"
    report["artifacts"]["report"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.emit("demo.completed", code="OK", report=report)
    return report


def _demo_leases(
    root: Path, declarations: Any, logger: StructuredLogger
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "ledger.sqlite3"
    orchestrator = LeaseOrchestrator(declarations, ledger_path, logger=logger)
    try:
        orchestrator.start_probes()
        alpha_rounds = []
        for _ in range(3):
            result = orchestrator.execute_round("alpha")
            alpha_rounds.append(
                {
                    "cycle": result.cycle,
                    "ratio_bp": result.ratio_bp,
                    "lease_version": result.lease_version,
                    "lease_renewed": result.lease_renewed,
                    "status": result.status,
                }
            )
        lossy_rounds = []
        for _ in range(2):
            result = orchestrator.execute_round("lossy")
            lossy_rounds.append(
                {
                    "cycle": result.cycle,
                    "ratio_bp": result.ratio_bp,
                    "lease_version": result.lease_version,
                    "status": result.status,
                    "pause_reason": result.pause_reason,
                    "consecutive_unrenewed": orchestrator.query_lease("lossy")[
                        "consecutive_unrenewed"
                    ],
                }
            )
        late = orchestrator.execute_round("late")
        mismatch = orchestrator.execute_round("mismatch")
        child_before_termination = orchestrator.query_lease("child")
        terminated = orchestrator.terminate_session("mismatch", "DEMO_FORCE_TERMINATE")
        return {
            "alpha_rounds": alpha_rounds,
            "alpha_final": orchestrator.query_lease("alpha"),
            "lossy_rounds": lossy_rounds,
            "lossy_final": orchestrator.query_lease("lossy"),
            "late_round": {
                "cycle": late.cycle,
                "outcome": late.probes[0]["outcome"],
                "reason": late.probes[0]["reason"],
                "ratio_bp": late.ratio_bp,
                "status": late.status,
                "pause_reason": late.pause_reason,
            },
            "mismatch_round": {
                "cycle": mismatch.cycle,
                "outcome": mismatch.probes[0]["outcome"],
                "reason": mismatch.probes[0]["reason"],
                "response_seq": mismatch.probes[0]["response_seq"],
                "ratio_bp": mismatch.ratio_bp,
                "status": mismatch.status,
                "pause_reason": mismatch.pause_reason,
            },
            "dependency_child_before_termination": child_before_termination,
            "terminated_mismatch": terminated,
            "ledger": str(ledger_path),
        }
    finally:
        orchestrator.close()


def _demo_inflight(
    root: Path, declarations: Any, logger: StructuredLogger
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "ledger.sqlite3"
    orchestrator = LeaseOrchestrator(declarations, ledger_path, logger=logger)
    try:
        orchestrator.start_probes()
        paused = orchestrator.execute_round("alpha", pause_after_send=True)
        raw_rows = [
            dict(row)
            for row in orchestrator.ledger._conn.execute(
                """
                SELECT cycle, attempt, probe_id, outcome, reason,
                       send_mono_ns, round_id
                FROM datagrams WHERE session_id = 'alpha'
                ORDER BY probe_id
                """
            )
        ]
        cursor_while_paused = orchestrator.query_lease("alpha")["cycle_cursor"]
        orchestrator.resume_session("alpha")
        rerun = orchestrator.execute_round("alpha")
        return {
            "paused_result": paused.to_dict(),
            "cursor_while_paused": cursor_while_paused,
            "dangling_rows": raw_rows,
            "resumed_round": {
                "cycle": rerun.cycle,
                "attempt": rerun.attempt,
                "ratio_bp": rerun.ratio_bp,
                "lease_version": rerun.lease_version,
                "status": rerun.status,
            },
            "committed_round_cycles": [
                record["cycle"] for record in orchestrator.ledger.round_records("alpha")
            ],
            "pause_events": orchestrator.ledger.pause_events("alpha"),
            "ledger": str(ledger_path),
        }
    finally:
        orchestrator.close()


def _demo_splice(
    root: Path, declarations: Any, logger: StructuredLogger
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    baseline_path = root / "baseline.sqlite3"
    spliced_path = root / "spliced.sqlite3"

    baseline = LeaseOrchestrator(declarations, baseline_path, logger=logger)
    try:
        baseline.start_probes()
        baseline.run_cycles("alpha", 4)
    finally:
        baseline.close()
    _wait_ports_free(declarations)

    first_segment = LeaseOrchestrator(declarations, spliced_path, logger=logger)
    try:
        first_segment.start_probes()
        first_segment.run_cycles("alpha", 2)
        first_segment.pause_session("alpha", "DEMO_SPLICE_PAUSE")
    finally:
        first_segment.close()

    second_segment = LeaseOrchestrator(declarations, spliced_path, logger=logger)
    try:
        second_segment.start_probes()
        second_segment.resume_session("alpha")
        second_segment.run_cycles("alpha", 2)
    finally:
        second_segment.close()

    with Ledger(baseline_path, declarations, logger=logger, recover_incomplete=False) as left:
        with Ledger(spliced_path, declarations, logger=logger, recover_incomplete=False) as right:
            splice_report = compare_splice_ledgers(left, right, session_ids=["alpha"])
            spliced_events = right.pause_events("alpha")
            spliced_rounds = [
                {
                    "cycle": record["cycle"],
                    "ratio_bp": record["ratio_bp"],
                    "lease_version": record["lease_version"],
                    "status": record["status_after"],
                }
                for record in right.logical_round_records("alpha")
            ]
    return {
        "comparison": splice_report.to_dict(),
        "spliced_pause_reasons": [
            {
                "kind": event["kind"],
                "reason": event["reason"],
                "cycle": event["cycle"],
            }
            for event in spliced_events
        ],
        "spliced_rounds": spliced_rounds,
        "baseline_ledger": str(baseline_path),
        "spliced_ledger": str(spliced_path),
    }


def _demo_crash(
    root: Path,
    declarations: Any,
    logger: StructuredLogger,
    repo_root: Path,
    samples: Path,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "ledger.sqlite3"
    snapshot_path = root / "before_crash.json"
    cli_log = root / "crash-cli.jsonl"
    command = [
        sys.executable,
        "-m",
        "probelease.cli",
        "run",
        "--sessions",
        str(samples / "sessions.json"),
        "--probes",
        str(samples / "probes.json"),
        "--ack-policy",
        str(samples / "ack_policy.json"),
        "--ledger",
        str(ledger_path),
        "--session",
        "alpha",
        "--cycles",
        "2",
        "--snapshot",
        str(snapshot_path),
        "--snapshot-after-cycle",
        "1",
        "--crash-before-commit-cycle",
        "2",
        "--log",
        str(cli_log),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    _wait_ports_free(declarations)

    recovered = LeaseOrchestrator(declarations, ledger_path, logger=logger)
    try:
        recovery_report = recovered.recovery_report.to_dict()
        committed_rounds_before_restore = [
            {"cycle": record["cycle"], "attempt": record["attempt"]}
            for record in recovered.ledger.round_records("alpha")
        ]
        unknown_rows = [
            dict(row)
            for row in recovered.ledger._conn.execute(
                """
                SELECT cycle, attempt, probe_id, outcome, reason
                FROM datagrams WHERE outcome = 'UNKNOWN' ORDER BY probe_id
                """
            )
        ]
    finally:
        recovered.close()

    restore_report = restore_from_snapshot(
        snapshot_path=snapshot_path,
        db_path=ledger_path,
        declarations=declarations,
        logger=logger,
    )
    _wait_ports_free(declarations)
    resumed = LeaseOrchestrator(declarations, ledger_path, logger=logger)
    try:
        resumed.start_probes()
        rerun = resumed.execute_round("alpha")
        following = resumed.execute_round("alpha")
        final_lease = resumed.query_lease("alpha")
        rounds_after_restore = [
            {"cycle": record["cycle"], "attempt": record["attempt"]}
            for record in resumed.ledger.round_records("alpha")
        ]
    finally:
        resumed.close()
    return {
        "crash_exit_code": completed.returncode,
        "crash_stdout": completed.stdout,
        "crash_stderr": completed.stderr,
        "forced_termination_exit_report": None,
        "exit_report_available": False,
        "recovery_report": recovery_report,
        "committed_rounds_before_restore": committed_rounds_before_restore,
        "unknown_rows_after_recovery": unknown_rows,
        "restore_report": restore_report.to_dict(),
        "rerun": {"cycle": rerun.cycle, "attempt": rerun.attempt},
        "following": {"cycle": following.cycle, "attempt": following.attempt},
        "rounds_after_restore_and_rerun": rounds_after_restore,
        "final_lease": final_lease,
        "snapshot": str(snapshot_path),
        "ledger": str(ledger_path),
    }


def _demo_replay(
    output: Path, declarations: Any, logger: StructuredLogger
) -> dict[str, Any]:
    ledger_path = output / "crash" / "ledger.sqlite3"
    report = replay_verify(
        ledger_path, declarations, session_id="alpha", logger=logger
    )
    return report.to_dict()


def _demo_injections(
    repo_root: Path, samples: Path, logger: StructuredLogger
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    for injection in ("corrupt-session", "ack-version", "dependency-cycle"):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "probelease.cli",
                "validate",
                "--sessions",
                str(samples / "sessions.json"),
                "--probes",
                str(samples / "probes.json"),
                "--ack-policy",
                str(samples / "ack_policy.json"),
                "--inject",
                injection,
            ],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {
                "code": "E_INTERNAL",
                "message": completed.stdout or completed.stderr,
            }
        result[injection] = {
            "exit_code": completed.returncode,
            "code": payload.get("code"),
            "field_path": payload.get("field_path"),
            "message": payload.get("message"),
        }
        logger.emit(
            "demo.injection", inject=injection, code=payload.get("code"), exit=completed.returncode
        )
    return result


def _demo_negative_checks(
    output: Path, declarations: Any, logger: StructuredLogger
) -> dict[str, Any]:
    source = output / "crash" / "ledger.sqlite3"
    replay_bad = output / "crash" / "negative_replay.sqlite3"
    splice_bad = output / "splice" / "negative_spliced.sqlite3"
    shutil.copy2(source, replay_bad)
    shutil.copy2(output / "splice" / "spliced.sqlite3", splice_bad)
    _tamper_ratio(replay_bad, cycle=2, value=1)
    _tamper_ratio(splice_bad, cycle=3, value=1)
    try:
        replay_verify(
            replay_bad,
            declarations,
            session_id="alpha",
            logger=logger,
        )
    except ProbeLeaseError as exc:
        replay_failure = {
            "code": exc.code,
            "field_path": exc.field_path,
            "details": exc.details,
        }
    else:
        replay_failure = {"code": "UNEXPECTED_OK"}
    _wait_ports_free(declarations)
    with Ledger(
        output / "splice" / "baseline.sqlite3",
        declarations,
        logger=logger,
        recover_incomplete=False,
    ) as left:
        with Ledger(
            splice_bad,
            declarations,
            logger=logger,
            recover_incomplete=False,
        ) as right:
            try:
                compare_splice_ledgers(left, right, session_ids=["alpha"])
            except ProbeLeaseError as exc:
                splice_failure = {
                    "code": exc.code,
                    "field_path": exc.field_path,
                    "details": exc.details,
                }
            else:
                splice_failure = {"code": "UNEXPECTED_OK"}
    return {"E_REPLAY": replay_failure, "E_SPLICE": splice_failure}


def _tamper_ratio(path: Path, *, cycle: int, value: int) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "UPDATE rounds SET ratio_bp = ? WHERE session_id = 'alpha' AND cycle = ?",
            (value, cycle),
        )
        conn.commit()
    finally:
        conn.close()


def _wait_ports_free(declarations: Any, timeout_s: float = 4.0) -> None:
    deadline = time.monotonic() + timeout_s
    for session in declarations.sessions:
        while True:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.bind(("127.0.0.1", session.port))
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"UDP port {session.port} was not released within {timeout_s}s"
                    )
                time.sleep(0.02)
            finally:
                probe.close()


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return

    def onerror(function: Any, target: str, _exc_info: Any) -> None:
        try:
            os.chmod(target, 0o666)
        except OSError:
            pass
        function(target)

    shutil.rmtree(path, onerror=onerror)
