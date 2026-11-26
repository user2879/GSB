from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .environment import check_environment
from .injection import run_injection_suite
from .studio import Studio


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vertical-workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="generate deterministic samples and workspace files")
    init_parser.add_argument("--root", default=".")

    run_parser = sub.add_parser("run", help="run one incremental/full pass")
    run_parser.add_argument("--root", default=".")

    inject_parser = sub.add_parser("inject", help="run fault injection isolation suite")
    inject_parser.add_argument("--root", default=".")

    verify_parser = sub.add_parser("verify", help="run acceptance verification and write reports")
    verify_parser.add_argument("--root", default=".")

    sub.add_parser("check-environment", help="check Python, NumPy, Tk, loopback TCP and multiprocessing")
    sub.add_parser("gui", help="launch Tkinter diagnostic workbench")

    args = parser.parse_args(argv)
    if args.command == "init":
        studio = Studio.create(Path(args.root))
        result = studio.engine.run(1)
        _print(
            {
                "initialized": str(Path(args.root).resolve()),
                "samples": len(result["samples"]),
                "keypoints": len(result["keypoints"]),
                "failures": len(result["failures"]),
            }
        )
        return 0
    if args.command == "run":
        studio = Studio(Path(args.root))
        result = studio.engine.run(max(1, studio.global_version))
        _print(
            {
                "samples": len(result["samples"]),
                "keypoints": len(result["keypoints"]),
                "conflicts": len(result["conflicts"]),
                "failures": result["failures"],
                "pending_queue": result["pending_queue"],
                "business_code_counts": result["business_code_counts"],
                "snapshot_symlink": str(Path(args.root) / "snapshots" / "current"),
            }
        )
        return 0
    if args.command == "inject":
        _print(run_injection_suite(args.root))
        return 0
    if args.command == "verify":
        from .acceptance import run_acceptance

        report = run_acceptance(args.root)
        _print({"report": str(Path(args.root) / "reports" / "acceptance_report.json"), "ok": report["ok"]})
        return 0 if report["ok"] else 1
    if args.command == "check-environment":
        report = check_environment()
        _print(report)
        return 0 if report["python"]["ok"] and report["numpy"]["ok"] else 1
    if args.command == "gui":
        from .gui import main as gui_main

        return gui_main()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
