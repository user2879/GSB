"""Repeatable end-to-end acceptance run; writes JSON and Markdown evidence."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from receipt_diag.engine import DiagnosticsEngine
from receipt_diag.errors import DiagnosticError
from receipt_diag.invalidation import compute_invalidation
from receipt_diag.ledger import Ledger
from receipt_diag.planning import plan_batches
from receipt_diag.samplegen import basic_sample, changed_sample
from receipt_diag.specs import load_specs

OUTPUT = ROOT / "artifacts" / "verification"


def free_ports(count: int = 2) -> list[int]:
    holders: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            holder.bind(("127.0.0.1", 0))
            holder.listen(1)
            holders.append(holder)
            ports.append(holder.getsockname()[1])
    finally:
        for holder in holders:
            holder.close()
    return ports


def materialize(
    path: Path,
    documents: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    ports: list[int],
) -> Path:
    request_doc, worker_doc, rules_doc = documents
    for worker, port in zip(worker_doc["workers"], ports):
        worker["port"] = port
    path.mkdir(parents=True, exist_ok=True)
    for name, document in zip(
        ("request_stream.json", "worker_spec.json", "merge_rules.json"),
        (request_doc, worker_doc, rules_doc),
    ):
        (path / name).write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return path


def wait_for_ports_closed(ports: list[int], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        any_open = False
        for port in ports:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(0.1)
            try:
                connection.connect(("127.0.0.1", port))
                any_open = True
            except OSError:
                pass
            finally:
                connection.close()
        if not any_open:
            return
        time.sleep(0.05)


def run_unit_tests() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-v"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    log_path = OUTPUT / "test.log"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    tail = (result.stdout + result.stderr).strip().splitlines()[-1:]
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "summary": tail[0] if tail else "",
        "log_path": str(log_path),
    }


def main() -> int:
    started = datetime.now(timezone.utc)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema": "receipt-diag-verification-v1",
        "started_at": started.isoformat(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "ok": False,
    }
    try:
        tests = run_unit_tests()
        report["unit_tests"] = tests

        ports = free_ports(2)
        basic_docs = basic_sample(port_base=43171)
        changed_docs = changed_sample(basic_docs)
        sample_basic = materialize(OUTPUT / "samples" / "basic", basic_docs, ports)
        sample_changed = materialize(OUTPUT / "samples" / "changed", changed_docs, ports)
        spec_basic = load_specs(sample_basic)
        spec_changed = load_specs(sample_changed)
        plan_basic = plan_batches(spec_basic)
        plan_changed = plan_batches(spec_changed)
        report["plan"] = {
            "batch_count": len(plan_basic.batches),
            "batch_ids": [batch.batch_id for batch in plan_basic.sorted_batches()],
            "batches": [batch.to_dict() for batch in plan_basic.sorted_batches()],
            "fingerprint_basic": plan_basic.fingerprint,
            "fingerprint_changed": plan_changed.fingerprint,
        }

        format_reports: dict[str, dict[str, Any]] = {}
        format_maps: dict[str, dict[str, tuple[str, str, str]]] = {}
        for fmt in ("r1", "r2", "r3"):
            ledger = Ledger(OUTPUT / f"format-{fmt}.db")
            try:
                engine = DiagnosticsEngine(spec_basic, OUTPUT / f"format-{fmt}-artifacts", receipt_format=fmt)
                run_report = engine.run(ledger)
                replay_report = engine.replay(ledger)
                format_reports[fmt] = {
                    "completed_batches": run_report["completed_batches"],
                    "ledger_content_digest": run_report["ledger_content_digest"],
                    "replay_verified_batches": len(replay_report["verified_batches"]),
                }
                format_maps[fmt] = {
                    batch["batch_id"]: (
                        batch["output_digest"],
                        batch["receipt_digest"],
                        batch["param_digest"],
                    )
                    for batch in run_report["batches"]
                }
            finally:
                ledger.close()
        report["receipt_formats"] = {
            "runs": format_reports,
            "all_digests_equal": format_maps["r1"] == format_maps["r2"] == format_maps["r3"],
        }

        main_ledger = Ledger(OUTPUT / "main.db")
        main_engine = DiagnosticsEngine(spec_basic, OUTPUT / "main-artifacts")
        try:
            initial_run = main_engine.run(main_ledger)
            before = main_ledger.batches_as_dict()
            invalidation = compute_invalidation(
                spec_basic, spec_changed, plan_basic, plan_changed
            )
            changed_engine = DiagnosticsEngine(spec_changed, OUTPUT / "main-artifacts")
            changed_run = changed_engine.apply_invalidation(
                main_ledger, spec_basic, plan_basic, invalidation
            )
            after = main_ledger.batches_as_dict()
            unaffected_unchanged = all(
                before[batch_id] == after[batch_id]
                for batch_id in invalidation.unaffected_batch_ids
            )
            replay = changed_engine.replay(main_ledger)
            report["invalidation"] = {
                "rerun_batch_count": len(invalidation.rerun_entries),
                "removed_batch_count": len(invalidation.removed_entries),
                "unaffected_batch_count": len(invalidation.unaffected_batch_ids),
                "invalid_request_ids": list(invalidation.invalid_request_ids),
                "unaffected_rows_byte_identical": unaffected_unchanged,
                "rerun_versions": {
                    entry.batch_id: after[entry.batch_id]["batch_version"]
                    for entry in invalidation.rerun_entries
                },
                "unaffected_versions": {
                    batch_id: after[batch_id]["batch_version"]
                    for batch_id in invalidation.unaffected_batch_ids
                },
                "replay_verified_batches": len(replay["verified_batches"]),
                "report": invalidation.to_dict(),
            }

            mismatch_batch = changed_engine.plan.sorted_batches()[0]
            main_ledger.connection.execute(
                "UPDATE batches SET output_digest='injected-replay-mismatch' WHERE batch_id=?",
                (mismatch_batch.batch_id,),
            )
            main_ledger.connection.commit()
            main_ledger.recompute_content_digest()
            try:
                changed_engine.replay(main_ledger)
                replay_error = None
            except DiagnosticError as exc:
                replay_error = exc.as_dict()
            report["replay_mismatch"] = replay_error
        finally:
            main_ledger.close()

        snapshot_a_ledger = Ledger(OUTPUT / "snapshot-a.db")
        try:
            snapshot_a_engine = DiagnosticsEngine(
                spec_basic, OUTPUT / "snapshot-a-artifacts"
            )
            snapshot_a_engine.run(snapshot_a_ledger)
        finally:
            snapshot_a_ledger.close()
        snapshot_b_ledger = Ledger(OUTPUT / "snapshot-b.db")
        try:
            snapshot_b_engine = DiagnosticsEngine(spec_changed, OUTPUT / "snapshot-b-artifacts")
            snapshot_b_engine.run(snapshot_b_ledger)
            snapshot_path = OUTPUT / "snapshot-a.json"
            with Ledger(OUTPUT / "snapshot-a.db") as ledger_a:
                snapshot = ledger_a.export_snapshot(snapshot_path)
            replaced = snapshot_b_ledger.restore_snapshot(snapshot)
            noop = snapshot_b_ledger.restore_snapshot(snapshot)
            report["snapshot"] = {
                "source_digest": snapshot["content_digest"],
                "restored_digest": snapshot_b_ledger.content_digest,
                "replaced_batch_ids": list(replaced),
                "noop_replaced_batch_ids": list(noop),
                "batch_count": len(snapshot["batches"]),
            }
        finally:
            snapshot_b_ledger.close()

        injection_results: dict[str, Any] = {}
        for injection in (
            "receipt_corruption",
            "dependency_dangling",
            "ledger_digest_mismatch",
        ):
            ledger = Ledger(OUTPUT / f"inject-{injection}.db")
            engine = DiagnosticsEngine(spec_basic, OUTPUT / f"inject-{injection}-artifacts")
            try:
                engine.run(ledger, injection=injection)
                injection_results[injection] = {"ok": True, "unexpected_success": True}
            except DiagnosticError as exc:
                injection_results[injection] = exc.as_dict()
            finally:
                ledger.close()
        report["injections"] = injection_results

        crash_ledger_path = OUTPUT / "crash.db"
        crash_artifacts = OUTPUT / "crash-artifacts"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        crash = subprocess.run(
            [
                sys.executable,
                "-m",
                "receipt_diag.cli",
                "run",
                "--sample-dir",
                str(sample_basic),
                "--db",
                str(crash_ledger_path),
                "--artifacts",
                str(crash_artifacts),
                "--crash-after-commits",
                "1",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        wait_for_ports_closed(ports)
        exit_report_after_crash = (crash_artifacts / "run_report.json").exists()
        crash_ledger = Ledger(crash_ledger_path)
        try:
            recovery_engine = DiagnosticsEngine(spec_basic, crash_artifacts)
            before_recovery = recovery_engine.recovery_status(crash_ledger)
            recovered = recovery_engine.recover(crash_ledger)
            recovered_replay = recovery_engine.replay(crash_ledger)
        finally:
            crash_ledger.close()
        report["crash_recovery"] = {
            "crash_exit_code": crash.returncode,
            "exit_report_existed_after_crash": exit_report_after_crash,
            "before_recovery": before_recovery,
            "recovered": recovered["recovery"],
            "completed_batches": recovered["completed_batches"],
            "replay_verified_batches": len(recovered_replay["verified_batches"]),
        }

        checks = {
            "unit_tests_passed": tests["ok"],
            "three_formats_equal": report["receipt_formats"]["all_digests_equal"],
            "invalidation_subset_only": (
                report["invalidation"]["rerun_batch_count"] == 3
                and report["invalidation"]["unaffected_batch_count"] == 2
                and report["invalidation"]["unaffected_rows_byte_identical"]
            ),
            "replay_mismatch_is_e_replay": (
                report["replay_mismatch"] is not None
                and report["replay_mismatch"].get("code") == "E_REPLAY"
            ),
            "snapshot_whole_replace": (
                report["snapshot"]["source_digest"] == report["snapshot"]["restored_digest"]
                and len(report["snapshot"]["replaced_batch_ids"]) == 5
                and report["snapshot"]["noop_replaced_batch_ids"] == []
            ),
            "injection_codes": (
                injection_results["receipt_corruption"].get("code") == "E_FIELD"
                and injection_results["dependency_dangling"].get("code") == "E_FIELD"
                and injection_results["ledger_digest_mismatch"].get("code") == "E_LEDGER"
            ),
            "crash_recovery_complete": (
                report["crash_recovery"]["crash_exit_code"] == 137
                and not report["crash_recovery"]["exit_report_existed_after_crash"]
                and report["crash_recovery"]["recovered"]["recovered"]
                and report["crash_recovery"]["completed_batches"] == 5
                and len(report["crash_recovery"]["recovered"]["replay_verified_batches"]) == 5
                and report["crash_recovery"]["replay_verified_batches"] == 5
            ),
        }
        report["checks"] = checks
        report["ok"] = all(checks.values())
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        (OUTPUT / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_markdown(report)
        print(json.dumps({"ok": report["ok"], "report": str(OUTPUT / "report.json"), "checks": checks}, indent=2))
        return 0 if report["ok"] else 1
    except Exception as exc:
        report["error"] = repr(exc)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        (OUTPUT / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_markdown(report)
        raise


def write_markdown(report: dict[str, Any]) -> None:
    destination = ROOT / "docs" / "VERIFICATION.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    checks = report.get("checks", {})
    lines = [
        "# Verification Record",
        "",
        f"- Generated (UTC): `{report.get('completed_at', report.get('started_at'))}`",
        f"- Python: `{report.get('python')}`",
        f"- Platform: `{report.get('platform')}`",
        f"- Overall: **{'PASS' if report.get('ok') else 'FAIL'}**",
        "",
    ]
    if "error" in report:
        lines.extend([f"Unexpected error: `{report['error']}`", ""])
    if checks:
        lines.extend(["## Checks", "", "| Check | Result |", "| --- | --- |"])
        for name, value in checks.items():
            lines.append(f"| `{name}` | {'PASS' if value else 'FAIL'} |")
        lines.append("")
    if "unit_tests" in report:
        lines.extend(
            [
                "## Unit Tests",
                "",
                f"- Exit code: `{report['unit_tests']['returncode']}`",
                f"- Summary: `{report['unit_tests']['summary']}`",
                f"- Log: `{report['unit_tests']['log_path']}`",
                "",
            ]
        )
    if "plan" in report:
        lines.extend(
            [
                "## Plan",
                "",
                f"- Batches: `{report['plan']['batch_count']}`",
                f"- Plan fingerprint: `{report['plan']['fingerprint_basic']}`",
                "",
            ]
        )
    if "receipt_formats" in report:
        lines.extend(
            [
                "## Receipt Formats",
                "",
                f"- r1/r2/r3 digest equality: `{report['receipt_formats']['all_digests_equal']}`",
            ]
        )
        for fmt, run in report["receipt_formats"]["runs"].items():
            lines.append(
                f"- `{fmt}`: batches `{run['completed_batches']}`, ledger digest `{run['ledger_content_digest']}`"
            )
        lines.append("")
    if "invalidation" in report:
        invalidation = report["invalidation"]
        lines.extend(
            [
                "## Invalidation",
                "",
                f"- Rerun batches: `{invalidation['rerun_batch_count']}`",
                f"- Unaffected batches: `{invalidation['unaffected_batch_count']}`",
                f"- Removed batches: `{invalidation['removed_batch_count']}`",
                f"- Unaffected rows byte-identical: `{invalidation['unaffected_rows_byte_identical']}`",
                f"- Replay verified batches: `{invalidation['replay_verified_batches']}`",
                "",
            ]
        )
    if "snapshot" in report:
        snapshot = report["snapshot"]
        lines.extend(
            [
                "## Snapshot",
                "",
                f"- Source digest: `{snapshot['source_digest']}`",
                f"- Restored digest: `{snapshot['restored_digest']}`",
                f"- Replaced batch IDs: `{len(snapshot['replaced_batch_ids'])}`",
                f"- Repeated restore replaced IDs: `{len(snapshot['noop_replaced_batch_ids'])}`",
                "",
            ]
        )
    if "injections" in report:
        lines.extend(["## Injections", "", "| Injection | Code | Path |", "| --- | --- | --- |"])
        for name, value in report["injections"].items():
            lines.append(f"| `{name}` | `{value.get('code')}` | `{value.get('path')}` |")
        lines.append("")
    if "replay_mismatch" in report and report["replay_mismatch"] is not None:
        mismatch = report["replay_mismatch"]
        lines.extend(
            [
                "## Replay Mismatch",
                "",
                f"- Code: `{mismatch.get('code')}`",
                f"- First location: `{mismatch.get('path')}`",
                "",
            ]
        )
    if "crash_recovery" in report:
        crash = report["crash_recovery"]
        lines.extend(
            [
                "## Crash Recovery",
                "",
                f"- Forced exit code: `{crash['crash_exit_code']}`",
                f"- Exit report existed after crash: `{crash['exit_report_existed_after_crash']}`",
                f"- Committed before recovery: `{crash['before_recovery']['completed_count']}`",
                f"- Unknown before recovery: `{len(crash['before_recovery']['unknown_batch_ids'])}`",
                f"- Recovered batches rerun: `{len(crash['recovered']['rerun_batch_ids'])}`",
                f"- Internal recovery replay verified batches: `{len(crash['recovered']['replay_verified_batches'])}`",
                f"- Replay verified batches: `{crash['replay_verified_batches']}`",
                "",
            ]
        )
    destination.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
