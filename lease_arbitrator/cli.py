from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .errors import ArbitrationError
from .sample import write_sample
from .ui import build_view_model, render_text_snapshot, run_tk
from .verify import verify_ledger


ERROR_EXIT_CODES = {
    "E_SPEC": 2,
    "E_PORT": 3,
    "E_HANDSHAKE": 4,
    "E_FIELD": 5,
    "E_LEASE": 6,
    "E_VERIFY": 7,
    "E_UNKNOWN": 8,
    "E_UI": 9,
}


def _print_json(value: Any, *, stderr: bool = False) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr if stderr else sys.stdout)


def _state_db(args: argparse.Namespace) -> Path:
    if getattr(args, "db", None):
        return Path(args.db)
    return Path(args.state_dir) / "ledger.sqlite"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lease_arbitrator", description="Cross-node artifact cache lease arbitrator")
    sub = parser.add_subparsers(dest="command", required=True)

    init_sample = sub.add_parser("init-sample", help="write deterministic local sample declarations")
    init_sample.add_argument("--dir", default="samples/demo")
    init_sample.add_argument("--force", action="store_true")

    run = sub.add_parser("run", help="run events against local node subprocesses")
    run.add_argument("--sample", default="samples/demo")
    run.add_argument("--state-dir", default=".runtime/demo")
    run.add_argument("--crash-batch", type=int, default=None, help=argparse.SUPPRESS)
    run.add_argument("--crash-attempt", type=int, default=1, help=argparse.SUPPRESS)

    recover = sub.add_parser("recover", help="mark interrupted inflight batches unknown")
    recover.add_argument("--sample", default="samples/demo")
    recover.add_argument("--state-dir", default=".runtime/demo")

    verify = sub.add_parser("verify", help="independently verify ledger against declarations")
    verify.add_argument("--sample", default="samples/demo")
    verify.add_argument("--state-dir", default=".runtime/demo")
    verify.add_argument("--db", default=None)

    ui = sub.add_parser("ui", help="open Tkinter diagnostics or print a read-only text snapshot")
    ui.add_argument("--state-dir", default=".runtime/demo")
    ui.add_argument("--db", default=None)
    ui.add_argument("--text", action="store_true", help="print stable text snapshot without opening Tk")

    doctor = sub.add_parser("doctor", help="check required local Python modules")
    doctor.add_argument("--require-tk", action="store_true")
    return parser


def doctor_report(require_tk: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {"python": sys.version.split()[0], "modules": {}}
    for module_name in ("sqlite3", "socket", "numpy"):
        try:
            module = __import__(module_name)
            report["modules"][module_name] = {"available": True, "version": getattr(module, "__version__", sqlite3.sqlite_version if module_name == "sqlite3" else None)}
        except Exception as exc:
            report["modules"][module_name] = {"available": False, "error": str(exc)}
    try:
        import tkinter

        report["modules"]["tkinter"] = {"available": True, "version": getattr(tkinter, "TkVersion", None)}
    except Exception as exc:
        report["modules"]["tkinter"] = {"available": False, "error": str(exc)}
    report["ok"] = all(report["modules"][name]["available"] for name in ("sqlite3", "socket", "numpy"))
    if require_tk:
        report["ok"] = report["ok"] and report["modules"]["tkinter"]["available"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init-sample":
            path = write_sample(args.dir, force=args.force)
            _print_json({"ok": True, "sample_dir": str(path), "files": ["nodes.json", "pipeline.json", "events.jsonl", "expected-core-digests.json"]})
            return 0

        if args.command == "doctor":
            report = doctor_report(require_tk=args.require_tk)
            _print_json(report)
            return 0 if report["ok"] else 1

        if args.command == "run":
            from .runner import SampleRunner

            with SampleRunner(
                args.sample,
                args.state_dir,
                crash_before_seq=args.crash_batch,
                crash_attempt=args.crash_attempt,
            ) as runner:
                report = runner.run()
            _print_json(report)
            return 0

        if args.command == "recover":
            from .runner import SampleRunner

            with SampleRunner(args.sample, args.state_dir) as runner:
                markers = runner.recover_inflight()
                open_unknown = [dict(row) for row in runner.ledger.open_unknown_batches()]
            _print_json({"ok": True, "recovered_markers": markers, "open_unknown_batches": open_unknown, "next_action": "rerun the same run command"})
            return 0

        if args.command == "verify":
            report = verify_ledger(args.sample, _state_db(args))
            _print_json(report)
            return 0

        if args.command == "ui":
            db_path = _state_db(args)
            if args.text:
                print(render_text_snapshot(build_view_model(db_path)))
                return 0
            run_tk(db_path)
            return 0

        parser.error(f"unknown command {args.command}")
        return 2
    except ArbitrationError as exc:
        payload = exc.to_dict()
        if exc.code == "E_UNKNOWN":
            payload["recovery"] = "run `recover`, then rerun the same event batch"
        _print_json(payload, stderr=True)
        return ERROR_EXIT_CODES.get(exc.code, 1)
    except FileNotFoundError as exc:
        _print_json({"code": "E_SPEC", "message": f"file not found: {exc}"}, stderr=True)
        return 2
    except Exception as exc:
        _print_json({"code": "E_INTERNAL", "type": type(exc).__name__, "message": str(exc)}, stderr=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
