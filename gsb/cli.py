"""Command line interface for the diagnostic pipeline."""

from __future__ import annotations

import argparse
import json
import socket
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .pipeline import recover_pipeline, run_pipeline
from .sample import build_sample_spec


def collect_doctor(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve()
    checks: dict[str, Any] = {}
    checks["python_version"] = sys.version.split()[0]
    checks["python_3_10_or_newer"] = sys.version_info >= (3, 10)
    try:
        import numpy as np

        checks["numpy"] = True
        checks["numpy_version"] = np.__version__
    except Exception as exc:  # noqa: BLE001
        checks["numpy"] = False
        checks["numpy_error"] = str(exc)

    try:
        import tkinter as tk

        checks["tkinter"] = True
        checks["tkinter_version"] = getattr(tk, "TkVersion", None)
    except Exception as exc:  # noqa: BLE001
        checks["tkinter"] = False
        checks["tkinter_note"] = f"optional GUI dependency unavailable: {exc}"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        checks["loopback_bind"] = True
        checks["loopback_probe_port"] = sock.getsockname()[1]

    with tempfile.TemporaryDirectory(dir=root if root.exists() else None) as tmp:
        db = Path(tmp) / "write.sqlite3"
        con = sqlite3.connect(db)
        con.execute("create table t(x)")
        con.execute("insert into t values (1)")
        con.commit()
        con.close()
        checks["sqlite_writable"] = True

    checks["workdir"] = str(root)
    checks["workdir_writable"] = root.exists() and __import__("os").access(root, __import__("os").W_OK)
    checks["tempdir_writable"] = Path(tempfile.gettempdir()).exists()
    checks["no_containers"] = not Path("/.dockerenv").exists()
    checks["transport"] = "tcp/127.0.0.1"
    checks["ok"] = bool(
        checks["python_3_10_or_newer"]
        and checks.get("numpy")
        and checks["loopback_bind"]
        and checks["sqlite_writable"]
        and checks["workdir_writable"]
    )
    return {"ok": checks["ok"], "version": __version__, "checks": checks}


def doctor() -> int:
    result = collect_doctor()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gsb-pipeline")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor")
    sub.add_parser("print-sample")

    sample = sub.add_parser("write-sample")
    sample.add_argument("--output", default="samples/diagnostic_spec.json")

    run = sub.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--session-id", default=None)
    run.add_argument("--crash-after-commit", type=int, default=None)
    run.add_argument("--no-scheduler", action="store_true")

    recover = sub.add_parser("recover")
    recover.add_argument("--output-dir", required=True)
    recover.add_argument("--session-id", required=True)
    recover.add_argument("--spec", default=None)

    ui = sub.add_parser("ui")
    ui.add_argument("--report", required=True)

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "print-sample":
        print(json.dumps(build_sample_spec(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "write-sample":
        _write_json(args.output, build_sample_spec())
        print(args.output)
        return 0
    if args.command == "run":
        report = run_pipeline(
            spec_path=args.spec,
            output_dir=args.output_dir,
            session_id=args.session_id,
            crash_after_commit=args.crash_after_commit,
            use_scheduler=not args.no_scheduler,
        )
        print(json.dumps({"ok": True, "report_path": report.get("report_path"), "summary": report["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "recover":
        result = recover_pipeline(output_dir=args.output_dir, session_id=args.session_id, spec_path=args.spec)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["recovery"]["status"] in {"consistent", "requires_rerun"} else 2
    if args.command == "ui":
        from .ui import launch

        launch(args.report)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
