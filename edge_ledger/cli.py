from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .engine import build_workspace
from .errors import SimulatedCrash
from .report import write_reports
from .samplegen import generate_workspace
from .verify import run_verification, write_verification
from .workspace import apply_command_file


def _summary(result) -> dict:
    return {
        "canonical_digest": result.canonical_digest,
        "global_version": result.result["global_version"],
        "snapshot_generation": result.snapshot_generation,
        "recovery": result.recovery,
        "runtime_ms": round(result.runtime_seconds * 1000, 3),
        "sample_count": result.result["stats"]["total"]["input_samples"],
        "rendered_count": result.result["stats"]["total"]["rendered_samples"],
        "removed_duplicates": result.result["stats"]["total"]["removed_duplicates"],
        "render_points_consumed": result.result["stats"]["total"]["render_points_consumed"],
        "pending_count": len(result.result["pending_queue"]),
        "failure_count": len(result.result["failures"]),
        "code_counts": result.result["stats"]["code_counts"],
        "max_observed_concurrency": result.probe_max_concurrency,
    }


def _root(value: str) -> Path:
    return Path(value).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edge-ledger", description=__doc__)
    parser.add_argument("--root", default=".", help="workspace root (default: cwd)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--json", action="store_true")

    build = subparsers.add_parser("build", help="run an incremental or full build")
    build.add_argument("--root", dest="sub_root", default=None)
    build.add_argument("--state-dir", type=Path, default=None)
    build.add_argument("--full", action="store_true")
    build.add_argument("--no-report", action="store_true")
    build.add_argument("--json", action="store_true")
    build.add_argument("--crash-after-stage", default=None)
    build.add_argument("--crash-mode", choices=["raise", "exit"], default="raise")

    apply = subparsers.add_parser("apply", help="apply a command file, then build")
    apply.add_argument("--root", dest="sub_root", default=None)
    apply.add_argument("--commands", required=True, type=Path)
    apply.add_argument("--no-report", action="store_true")
    apply.add_argument("--json", action="store_true")

    report = subparsers.add_parser("report", help="build and write reports")
    report.add_argument("--root", dest="sub_root", default=None)
    report.add_argument("--json", action="store_true")

    verify = subparsers.add_parser("verify", help="run the full acceptance sequence")
    verify.add_argument("--root", dest="sub_root", default=None)
    verify.add_argument("--output", type=Path, default=None)
    verify.add_argument("--json", action="store_true")

    gui = subparsers.add_parser("gui", help="launch the Tk diagnostic interface")
    gui.add_argument("--root", dest="sub_root", default=None)

    args = parser.parse_args(argv)
    root = _root(args.sub_root or args.root)
    if args.command == "generate":
        summary = generate_workspace(root)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command in {"build", "apply"}:
        if args.command == "apply":
            _, failures = apply_command_file(root, args.commands)
            if failures:
                print(
                    json.dumps(
                        {"ok": False, "failures": [failure.as_dict() for failure in failures]},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                return 2
        try:
            result = build_workspace(
                root,
                state_dir=getattr(args, "state_dir", None),
                full=getattr(args, "full", False),
                crash_after_stage=getattr(args, "crash_after_stage", None),
                crash_mode=getattr(args, "crash_mode", "raise"),
            )
        except SimulatedCrash as exc:
            print(f"simulated crash: {exc}", file=sys.stderr)
            return 3
        if not args.no_report:
            paths = write_reports(result)
            if args.json:
                payload = _summary(result)
                payload["reports"] = {key: str(value) for key, value in paths.items()}
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(json.dumps(_summary(result), ensure_ascii=False, sort_keys=True))
        else:
            print(json.dumps(_summary(result), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "report":
        result = build_workspace(root)
        paths = write_reports(result)
        print(
            json.dumps(
                {
                    "canonical_digest": result.canonical_digest,
                    "reports": {key: str(value) for key, value in paths.items()},
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "verify":
        verification = run_verification(root)
        output = args.output or (root / "artifacts" / "verification.json")
        write_verification(verification, output)
        if args.json:
            print(json.dumps(verification.as_dict(), ensure_ascii=False, sort_keys=True))
        else:
            print(
                json.dumps(
                    {
                        "passed": verification.passed,
                        "output": str(output),
                        "checks": len(verification.report["checks"]),
                        "edit_steps": len(verification.report["edit_steps"]),
                        "final_digest": verification.report.get("final_digest"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return 0 if verification.passed else 1
    if args.command == "gui":
        from .gui import launch_gui

        launch_gui(root)
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())

