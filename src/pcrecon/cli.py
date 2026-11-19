"""Command-line diagnostics entry point.

Subcommands:
    generate      create the deterministic sample under <workspace>/data
    check-gl      verify the offscreen OpenGL context / FBO
    run           execute the fixed edit sequence and emit deterministic reports
    crash-demo    simulate a hard kill before commit, then recover orphans
    hash-verify   re-run the pipeline and compare report hashes
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import config
from .application import Application
from .generator import generate_sample
from .hashing import digest_file, read_json


def _default_workspace() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def cmd_generate(args) -> int:
    out_dir = os.path.join(args.workspace, "data")
    result = generate_sample(out_dir, seed=args.seed)
    print(json.dumps({"status": "ok", "files": result["files"], "stats": result["manifest"]["stats"]},
                     sort_keys=True, indent=2))
    return 0


def cmd_check_gl(args) -> int:
    from .errors import GLContextError
    from .render_gl import verify_offscreen_context

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        info = verify_offscreen_context()
    except GLContextError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True))
        return 2
    print(json.dumps(info, sort_keys=True, indent=2))
    return 0


def cmd_run(args) -> int:
    app = Application(args.workspace, verify=not args.no_verify)
    summary = app.run_edit_sequence()
    if args.undo:
        restored = app.undo_last()
        summary["undo_restored_version"] = restored.version
    exit_report = app.write_clean_exit_report()
    printable = {
        "status": "ok",
        "incremental_report": summary["incremental_report"],
        "frame_report": summary["frame_report"],
        "incremental_report_hash": summary["incremental_report_hash"],
        "frame_report_hash": summary["frame_report_hash"],
        "all_consistent": summary["incremental"]["all_consistent"],
        "frame_count": summary["frames"]["frame_count"],
        "degraded_frames": summary["frames"]["degraded_frame_count"],
        "clean_exit": exit_report["clean_exit"],
    }
    print(json.dumps(printable, sort_keys=True, indent=2))
    return 0


def cmd_crash_demo(args) -> int:
    """Create an uncommitted tmp cache, kill without exit report, then recover."""
    from .ledger import Ledger

    ledger = Ledger(args.workspace)
    orphan_dir = os.path.join(ledger.cache_root, "tmp-simulated-crash")
    os.makedirs(orphan_dir, exist_ok=True)
    with open(os.path.join(orphan_dir, "scratch.bin"), "wb") as fh:
        fh.write(b"uncommitted")
    ledger.remove_exit_report()
    recovered = ledger.recover()
    still_exists = os.path.exists(orphan_dir)
    print(json.dumps({
        "status": "ok",
        "recovered_orphans": recovered["recovered_orphans"],
        "orphan_reaped": not still_exists,
        "exit_report_present": recovered["exit_report_present"],
        "head_version": recovered["head_version"],
    }, sort_keys=True, indent=2))
    return 0 if (recovered["recovered_orphans"] and not still_exists) else 1


def cmd_hash_verify(args) -> int:
    inc = os.path.join(args.workspace, "reports", "incremental_report.json")
    frame = os.path.join(args.workspace, "reports", "frame_report.json")
    before = {inc: digest_file(inc), frame: digest_file(frame)}
    app = Application(args.workspace, verify=True)
    app.run_edit_sequence()
    app.write_clean_exit_report()
    after = {inc: digest_file(inc), frame: digest_file(frame)}
    ok = before == after
    print(json.dumps({"status": "ok" if ok else "mismatch", "before": before, "after": after},
                     sort_keys=True, indent=2))
    return 0 if ok else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcrecon")
    parser.add_argument("--workspace", default=_default_workspace())
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--seed", type=int, default=config.SEED)
    p_gen.set_defaults(func=cmd_generate)

    p_gl = sub.add_parser("check-gl")
    p_gl.set_defaults(func=cmd_check_gl)

    p_run = sub.add_parser("run")
    p_run.add_argument("--no-verify", action="store_true")
    p_run.add_argument("--undo", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_crash = sub.add_parser("crash-demo")
    p_crash.set_defaults(func=cmd_crash_demo)

    p_hash = sub.add_parser("hash-verify")
    p_hash.set_defaults(func=cmd_hash_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
