from __future__ import annotations

import argparse
import json
import platform
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from .errors import (
    E_RULE,
    E_SNAPSHOT,
    E_UPSTREAM,
    MeshError,
    RuleError,
    SnapshotError,
    UpstreamError,
)
from .ledger import Ledger
from .logging import JsonlLogger
from .rules import RequestSpec, RuleSet, load_requests, load_rules
from .scheduler import Scheduler
from .snapshot import (
    SnapshotStore,
    build_snapshot,
    diff_snapshots,
    load_snapshot,
)
from .transport import AttemptResponse, HttpTransport, Transport
from .upstream import start_upstreams
from .util import (
    atomic_write_json,
    mono_ms,
    sha256_file,
    utc_now_iso,
)
from .verify import verify_snapshot


EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_E_RULE = 2
EXIT_E_UPSTREAM = 3
EXIT_E_SNAPSHOT = 4
EXIT_E_VERIFY = 5


INJECTION_ALIASES = {
    "E_RULE": "E_RULE",
    "rule_corruption": "E_RULE",
    "rule_file_corruption": "E_RULE",
    "E_UPSTREAM": "E_UPSTREAM",
    "upstream_unreachable": "E_UPSTREAM",
    "E_SNAPSHOT": "E_SNAPSHOT",
    "snapshot_write_failure": "E_SNAPSHOT",
}


class InjectedUnreachableTransport:
    def request(
        self, instance: Any, request: RequestSpec, attempt_no: int
    ) -> AttemptResponse:
        del instance, request, attempt_no
        return AttemptResponse(
            False,
            E_UPSTREAM,
            "injected upstream unreachable: connection intentionally refused",
            None,
        )


def _default_round_id(prefix: str = "round") -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def _write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    if not path:
        return
    atomic_write_json(path, report)


def _print(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _apply_rule_mutation(
    source: Path,
    target: Path,
    delay_ms: int,
) -> threading.Timer | None:
    if delay_ms < 0:
        return None

    def mutate() -> None:
        time.sleep(delay_ms / 1000.0)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        except OSError:
            pass

    timer = threading.Timer(0.001, mutate)
    timer.daemon = True
    timer.start()
    return timer


def _run_command(args: argparse.Namespace) -> int:
    log_path = Path(args.log)
    report_path = Path(args.report) if args.report else None
    started = mono_ms()
    logger: JsonlLogger | None = None
    upstream_runtime = None
    mutation_timer = None
    try:
        logger = JsonlLogger(log_path)
        logger.emit(
            "run_invoked",
            rules=args.rules,
            requests=args.requests,
            db=args.db,
            snapshots=args.snapshots,
            injection=args.inject,
            start_upstreams=args.start_upstreams,
        )
    except OSError as exc:
        report = {
            "schema_version": 1,
            "command": "run",
            "status": "error",
            "code": "E_IO",
            "message": f"unable to open structured log: {exc}",
        }
        _print(report)
        return EXIT_INTERNAL

    injection = INJECTION_ALIASES.get(args.inject) if args.inject else None
    if args.inject and injection is None:
        report = {
            "schema_version": 1,
            "command": "run",
            "status": "error",
            "code": E_RULE,
            "field_path": "$.inject",
            "message": f"unknown injection kind {args.inject!r}",
        }
        logger.error(RuleError(report["message"], "$.inject"), round_id=None)
        _write_report(report_path, report)
        _print(report)
        logger.close()
        return EXIT_E_RULE

    round_id = args.round_id or _default_round_id()
    try:
        if injection == "E_RULE":
            rule_path = Path(args.rules)
            try:
                raw_rules = rule_path.read_bytes()
            except OSError as exc:
                raise RuleError(
                    "unable to read rules file for injection",
                    "$",
                    {"reason": str(exc)},
                ) from exc
            del raw_rules
            rules = load_rules(rule_path, raw_override=b'{"schema_version": 1, "services": [')
        else:
            rules = load_rules(args.rules)
        requests = load_requests(args.requests, rules)
    except MeshError as exc:
        logger.error(exc, round_id=round_id, phase="rules")
        report = {
            "schema_version": 1,
            "command": "run",
            "status": "error",
            "code": exc.code,
            "field_path": exc.field_path,
            "message": exc.message,
            "context": dict(exc.context),
            "round_id": round_id,
            "log_path": str(log_path),
        }
        _write_report(report_path, report)
        _print(report)
        logger.close()
        return EXIT_E_RULE

    if args.mutate_rules_to and args.mutate_after_ms is not None:
        mutation_timer = _apply_rule_mutation(
            Path(args.mutate_rules_to),
            Path(args.rules),
            int(args.mutate_after_ms),
        )

    try:
        if args.start_upstreams and injection != "E_UPSTREAM":
            upstream_runtime = start_upstreams(args.upstream_config)
            logger.emit(
                "upstreams_started",
                round_id=round_id,
                ports=upstream_runtime.ports,
                config=args.upstream_config,
            )
        elif injection == "E_UPSTREAM":
            logger.emit(
                "fault_injected",
                round_id=round_id,
                code=E_UPSTREAM,
                kind="upstream_unreachable",
            )
        transport: Transport = (
            InjectedUnreachableTransport()
            if injection == "E_UPSTREAM"
            else HttpTransport()
        )
        input_sha256 = sha256_file(args.requests)
        with Ledger(args.db) as ledger:
            scheduler = Scheduler(
                rules,
                requests,
                transport,
                ledger,
                logger,
                rules_path=args.rules,
                crash_after_ms=args.crash_after_ms,
            )
            result = scheduler.run(round_id, input_sha256=input_sha256)
            snapshot = build_snapshot(
                round_id,
                ledger,
                rules,
                rules_changed_during_round=result.rules_changed_during_round,
                observed_rule_version=result.observed_rule_version,
            )
            verification = verify_snapshot(snapshot)
            snapshot_path: Path | None = None
            try:
                store = SnapshotStore(
                    args.snapshots,
                    injected_failure=injection == "E_SNAPSHOT",
                )
                if injection == "E_SNAPSHOT":
                    logger.emit(
                        "fault_injected",
                        round_id=round_id,
                        code=E_SNAPSHOT,
                        kind="snapshot_write_failure",
                    )
                snapshot_path = store.publish(snapshot, ledger)
                logger.emit(
                    "snapshot_published",
                    round_id=round_id,
                    snapshot_id=snapshot["snapshot_id"],
                    rule_version=snapshot["rule_version"],
                    path=str(snapshot_path),
                )
            except SnapshotError as exc:
                logger.error(exc, round_id=round_id, phase="snapshot")
                report = {
                    "schema_version": 1,
                    "command": "run",
                    "status": "error",
                    "code": E_SNAPSHOT,
                    "field_path": exc.field_path,
                    "message": exc.message,
                    "context": dict(exc.context),
                    "round_id": round_id,
                    "rule_version": rules.rules_version,
                    "snapshot_built": True,
                    "snapshot_persisted": False,
                    "verification": verification,
                    "log_path": str(log_path),
                    "db_path": str(args.db),
                    "duration_ms": round(mono_ms() - started, 3),
                }
                if report_path:
                    atomic_write_json(report_path, report)
                _print(report)
                return EXIT_E_SNAPSHOT

        business_code = E_UPSTREAM if injection == "E_UPSTREAM" else None
        if verification["status"] != "pass":
            business_code = "E_VERIFY"
        status = "error" if business_code else "ok"
        report = {
            "schema_version": 1,
            "command": "run",
            "status": status,
            "code": business_code,
            "round_id": round_id,
            "rule_version": rules.rules_version,
            "observed_rule_version": result.observed_rule_version,
            "rules_changed_during_round": result.rules_changed_during_round,
            "request_count": len(requests),
            "global_concurrency": {
                "limit": result.budget.global_limit,
                "peak": result.budget.global_peak,
            },
            "summary": snapshot["summary"],
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_path": str(snapshot_path),
            "latest_snapshot_path": str(Path(args.snapshots) / "latest.json"),
            "verification": verification,
            "db_path": str(args.db),
            "log_path": str(log_path),
            "duration_ms": round(mono_ms() - started, 3),
        }
        logger.emit(
            "run_finished",
            round_id=round_id,
            status=status,
            code=business_code,
            verification_status=verification["status"],
            snapshot_id=snapshot["snapshot_id"],
            duration_ms=report["duration_ms"],
        )
        _write_report(report_path, report)
        _print(report)
        if injection == "E_UPSTREAM":
            return EXIT_E_UPSTREAM
        if business_code == "E_VERIFY":
            return EXIT_E_VERIFY
        return EXIT_OK
    except MeshError as exc:
        logger.error(exc, round_id=round_id, phase="execution")
        report = {
            "schema_version": 1,
            "command": "run",
            "status": "error",
            "code": exc.code,
            "field_path": exc.field_path,
            "message": exc.message,
            "context": dict(exc.context),
            "round_id": round_id,
            "log_path": str(log_path),
        }
        _write_report(report_path, report)
        _print(report)
        if exc.code == E_RULE:
            return EXIT_E_RULE
        if exc.code == E_UPSTREAM:
            return EXIT_E_UPSTREAM
        if exc.code == E_SNAPSHOT:
            return EXIT_E_SNAPSHOT
        return EXIT_INTERNAL
    except Exception as exc:
        logger.error(exc, round_id=round_id, phase="internal", traceback=traceback.format_exc())
        report = {
            "schema_version": 1,
            "command": "run",
            "status": "error",
            "code": "E_INTERNAL",
            "message": str(exc),
            "round_id": round_id,
            "log_path": str(log_path),
            "traceback": traceback.format_exc(),
        }
        _write_report(report_path, report)
        _print(report)
        return EXIT_INTERNAL
    finally:
        if mutation_timer is not None:
            mutation_timer.cancel()
        if upstream_runtime is not None:
            upstream_runtime.stop()
            try:
                logger.emit("upstreams_stopped", ports=upstream_runtime.ports)
            except Exception:
                pass
        logger.close()


def _verify_command(args: argparse.Namespace) -> int:
    try:
        snapshot = load_snapshot(args.snapshot)
        result = verify_snapshot(snapshot)
    except MeshError as exc:
        _print(exc.to_dict())
        return EXIT_E_VERIFY
    _write_report(args.report, result)
    _print(result)
    return EXIT_OK if result["status"] == "pass" else EXIT_E_VERIFY


def _recover_command(args: argparse.Namespace) -> int:
    logger = JsonlLogger(args.log)
    try:
        with Ledger(args.db) as ledger:
            report = ledger.recovery_report()
        report["log_path"] = str(args.log)
        report["db_path"] = str(args.db)
        logger.emit(
            "recovery_inspected",
            interrupted_rounds=len(report["interrupted_rounds"]),
            recovered_items=len(report["recovered_items"]),
            unknown_items=len(report["unknown_items"]),
            unknown_attempts=len(report["unknown_attempts"]),
            exit_report_supported=report["exit_report_supported"],
        )
    except Exception as exc:
        logger.error(exc, phase="recovery")
        _print(
            {
                "schema_version": 1,
                "command": "recover",
                "status": "error",
                "code": "E_RECOVERY",
                "message": str(exc),
            }
        )
        return EXIT_INTERNAL
    finally:
        logger.close()
    _write_report(args.report, report)
    _print(report)
    return EXIT_OK


def _diff_command(args: argparse.Namespace) -> int:
    try:
        old = load_snapshot(args.old)
        new = load_snapshot(args.new)
        report = diff_snapshots(old, new)
    except MeshError as exc:
        _print(exc.to_dict())
        return EXIT_INTERNAL
    _write_report(args.report, report)
    _print(report)
    return EXIT_OK


def _view_command(args: argparse.Namespace) -> int:
    from .viewmodel import build_view_model, render_view_text

    try:
        snapshot = load_snapshot(args.snapshot)
        view = build_view_model(snapshot)
    except MeshError as exc:
        _print(exc.to_dict())
        return EXIT_INTERNAL
    if args.format == "text":
        print(render_view_text(view))
    else:
        _print(view)
    if args.report:
        _write_report(args.report, view)
    return EXIT_OK


def _gui_command(args: argparse.Namespace) -> int:
    try:
        from .gui import launch

        return launch(args.snapshot)
    except Exception as exc:
        _print(
            {
                "schema_version": 1,
                "command": "gui",
                "status": "error",
                "code": "E_UI",
                "message": str(exc),
            }
        )
        return EXIT_INTERNAL


def _doctor_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "check": "python",
            "status": "pass" if sys.version_info >= (3, 11) else "fail",
            "actual": platform.python_version(),
            "required": ">=3.11",
        }
    )
    for name in ("numpy", "tkinter"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", getattr(module, "TkVersion", "present"))
            checks.append(
                {"check": name, "status": "pass", "actual": str(version)}
            )
        except Exception as exc:
            checks.append(
                {
                    "check": name,
                    "status": "warn",
                    "actual": f"unavailable: {exc}",
                    "note": "required for the Windows GUI environment",
                }
            )
    probe = workspace / f".write-probe-{uuid.uuid4().hex}"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        write_status = "pass"
        write_detail = "workspace is writable"
    except OSError as exc:
        write_status = "fail"
        write_detail = str(exc)
    checks.append(
        {
            "check": "workspace_write",
            "status": write_status,
            "actual": write_detail,
            "path": str(workspace),
        }
    )
    checks.append(
        {
            "check": "sqlite",
            "status": "pass",
            "actual": sqlite3.sqlite_version,
            "required": ">=3.35",
        }
    )
    checks.append(
        {
            "check": "platform",
            "status": "info",
            "actual": platform.platform(),
            "note": "target execution environment is local Windows; harness may use WSL",
        }
    )
    overall = "fail" if any(item["status"] == "fail" for item in checks) else "ok"
    report = {
        "schema_version": 1,
        "command": "doctor",
        "status": overall,
        "checks": checks,
    }
    _write_report(args.report, report)
    _print(report)
    return EXIT_OK if overall == "ok" else EXIT_INTERNAL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meshgov",
        description="Deterministic local service-mesh traffic governance workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one governed request round")
    run.add_argument("--rules", default="samples/rules.json")
    run.add_argument("--requests", default="samples/requests.json")
    run.add_argument("--db", default="artifacts/mesh.db")
    run.add_argument("--log", default="artifacts/logs/events.jsonl")
    run.add_argument("--snapshots", default="artifacts/snapshots")
    run.add_argument("--report", default="artifacts/reports/run.json")
    run.add_argument("--round-id")
    run.add_argument("--upstream-config", default="samples/upstreams.json")
    run.add_argument(
        "--start-upstreams",
        action="store_true",
        help="start deterministic loopback upstreams in this process",
    )
    run.add_argument(
        "--inject",
        choices=sorted(INJECTION_ALIASES),
        help="inject E_RULE, E_UPSTREAM, or E_SNAPSHOT",
    )
    run.add_argument(
        "--mutate-rules-to",
        help="test hook: copy this file over --rules while the round is running",
    )
    run.add_argument("--mutate-after-ms", type=int)
    run.add_argument(
        "--crash-after-ms",
        type=int,
        help="test hook: force-terminate the process after the given delay",
    )
    run.set_defaults(func=_run_command)

    verify = subparsers.add_parser("verify", help="verify a result snapshot")
    verify.add_argument("--snapshot", default="artifacts/snapshots/latest.json")
    verify.add_argument("--report", default="artifacts/reports/verify.json")
    verify.set_defaults(func=_verify_command)

    recover = subparsers.add_parser("recover", help="inspect persisted state after termination")
    recover.add_argument("--db", default="artifacts/mesh.db")
    recover.add_argument("--log", default="artifacts/logs/events.jsonl")
    recover.add_argument("--report", default="artifacts/reports/recovery.json")
    recover.set_defaults(func=_recover_command)

    diff = subparsers.add_parser("diff", help="diff two immutable snapshots")
    diff.add_argument("--old", required=True)
    diff.add_argument("--new", default="artifacts/snapshots/latest.json")
    diff.add_argument("--report", default="artifacts/reports/snapshot-diff.json")
    diff.set_defaults(func=_diff_command)

    view = subparsers.add_parser("view", help="render the stable view model headlessly")
    view.add_argument("--snapshot", default="artifacts/snapshots/latest.json")
    view.add_argument("--format", choices=("json", "text"), default="json")
    view.add_argument("--report")
    view.set_defaults(func=_view_command)

    gui = subparsers.add_parser("gui", help="launch the Tkinter workbench")
    gui.add_argument("--snapshot", default="artifacts/snapshots/latest.json")
    gui.set_defaults(func=_gui_command)

    doctor = subparsers.add_parser("doctor", help="check local runtime prerequisites")
    doctor.add_argument("--workspace", default=".")
    doctor.add_argument("--report", default="artifacts/reports/doctor.json")
    doctor.set_defaults(func=_doctor_command)

    acceptance = subparsers.add_parser(
        "acceptance", help="run the full deterministic acceptance suite"
    )
    acceptance.add_argument("--workspace", default=".")
    acceptance.add_argument("--report", default="artifacts/reports/acceptance.json")
    acceptance.add_argument("--skip-tests", action="store_true")
    acceptance.set_defaults(
        func=lambda args: __import__("meshgov.acceptance", fromlist=["main"]).main(
            [
                "--workspace",
                args.workspace,
                "--report",
                args.report,
                *(["--skip-tests"] if args.skip_tests else []),
            ]
        )
    )

    upstream = subparsers.add_parser("upstream", help="run deterministic loopback upstreams")
    upstream.add_argument("--config", default="samples/upstreams.json")
    upstream.set_defaults(
        func=lambda args: __import__("meshgov.upstream", fromlist=["main"]).main(
            ["--config", args.config]
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
