"""Command-line entry points for generation, runs, comparisons, and edits."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

from .edits import EditError, apply_edit_commands
from .engine import Workbench, load_config
from .generate import generate_samples, generate_variants, write_default_config, write_edit_commands
from .util import atomic_write_json


def _run(args: argparse.Namespace) -> int:
    stop_event = threading.Event()

    def request_stop(signum, frame):  # pragma: no cover - exercised by process signals
        stop_event.set()

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            try:
                signal.signal(getattr(signal, signal_name), request_stop)
            except (ValueError, OSError):
                pass

    config = load_config(Path(args.config), output_override=Path(args.output) if args.output else None)
    workbench = Workbench(config, serial=bool(args.serial), stop_event=stop_event)
    report = workbench.run(stop_after_pages=args.stop_after_pages)
    snapshot_summary = {
        key: report["snapshot"][key]
        for key in ("version", "config_hash", "completed_pages", "pending_pages", "consumed_budget")
    }
    print(
        json.dumps(
            {
                "report": str(Path(config["_output_dir"]) / "report.json"),
                "summary": report["summary"],
                "snapshot": snapshot_summary,
                "result_digest": report["result_digest"],
                "stopped": report["stopped"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _generate(args: argparse.Namespace) -> int:
    generate_samples(Path(args.samples))
    generate_variants(Path(args.variants))
    write_default_config(Path(args.config))
    write_edit_commands(Path(args.commands))
    print(
        json.dumps(
            {
                "samples": str(Path(args.samples).resolve()),
                "variants": str(Path(args.variants).resolve()),
                "config": str(Path(args.config).resolve()),
                "commands": str(Path(args.commands).resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    exact_equal = left == right
    digest_equal = left.get("result_digest") == right.get("result_digest")
    pages_equal = left.get("pages") == right.get("pages")
    failures_equal = left.get("failures") == right.get("failures")
    budget_equal = left.get("budget") == right.get("budget")
    result = {
        "exact_equal": exact_equal,
        "result_digest_equal": digest_equal,
        "pages_equal": pages_equal,
        "failures_equal": failures_equal,
        "budget_equal": budget_equal,
        "left_digest": left.get("result_digest"),
        "right_digest": right.get("result_digest"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.out:
        atomic_write_json(Path(args.out), result)
    if args.strict:
        return 0 if exact_equal else 1
    return 0 if digest_equal and pages_equal and failures_equal and budget_equal else 1


def _apply_edits(args: argparse.Namespace) -> int:
    try:
        result = apply_edit_commands(Path(args.config), Path(args.commands))
    except EditError as exc:
        print(f"edit command failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _selfcheck(args: argparse.Namespace) -> int:
    base_output = Path(args.output)
    serial_output = base_output / "serial"
    parallel_output = base_output / "parallel"
    for path in (serial_output, parallel_output):
        if path.exists():
            shutil.rmtree(path)
    serial_config = load_config(Path(args.config), output_override=serial_output)
    parallel_config = load_config(Path(args.config), output_override=parallel_output)
    serial_report = Workbench(serial_config, serial=True).run()
    parallel_report = Workbench(parallel_config, serial=False).run()
    exact_equal = serial_report == parallel_report
    verification = {
        "exact_equal": exact_equal,
        "result_digest_equal": serial_report["result_digest"] == parallel_report["result_digest"],
        "serial": serial_report["summary"],
        "parallel": parallel_report["summary"],
        "result_digest": serial_report["result_digest"],
        "serial_report": str(serial_output / "report.json"),
        "parallel_report": str(parallel_output / "report.json"),
    }
    base_output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(base_output / "selfcheck.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    passed = (
        exact_equal
        and serial_report["summary"]["failures"] == 0
        and serial_report["summary"]["pages_ok"] == serial_report["summary"]["pages_valid"]
        and serial_report["summary"]["verification_passed_pages"] == serial_report["summary"]["pages_valid"]
    )
    return 0 if passed else 1


def _validate_report(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    failures = report.get("failures", [])
    ok = (
        not failures
        and summary.get("pages_ok") == summary.get("pages_valid")
        and summary.get("verification_passed_pages") == summary.get("pages_valid")
    )
    print(json.dumps({"ok": ok, "summary": summary, "failures": failures}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chart-workbench", description="Deterministic chart calibration workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a calibration pass")
    run_parser.add_argument("--config", default="config/calibration.json")
    run_parser.add_argument("--output")
    run_parser.add_argument("--serial", action="store_true", help="reproducible single-worker baseline")
    run_parser.add_argument("--stop-after-pages", type=int, default=None, help=argparse.SUPPRESS)
    run_parser.set_defaults(func=_run)

    generate_parser = subparsers.add_parser("generate", help="regenerate deterministic local samples")
    generate_parser.add_argument("--samples", default="samples")
    generate_parser.add_argument("--variants", default="variants")
    generate_parser.add_argument("--config", default="config/calibration.json")
    generate_parser.add_argument("--commands", default="config/edit_commands.json")
    generate_parser.set_defaults(func=_generate)

    compare_parser = subparsers.add_parser("compare", help="compare two diagnostic reports")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--strict", action="store_true")
    compare_parser.add_argument("--out")
    compare_parser.set_defaults(func=_compare)

    edits_parser = subparsers.add_parser("apply-edits", help="apply an edit-command JSON stream")
    edits_parser.add_argument("--config", default="config/calibration.json")
    edits_parser.add_argument("--commands", default="config/edit_commands.json")
    edits_parser.set_defaults(func=_apply_edits)

    selfcheck_parser = subparsers.add_parser("selfcheck", help="run serial/parallel baseline consistency check")
    selfcheck_parser.add_argument("--config", default="config/calibration.json")
    selfcheck_parser.add_argument("--output", default="out/selfcheck")
    selfcheck_parser.set_defaults(func=_selfcheck)

    validate_parser = subparsers.add_parser("validate-report", help="check a report against acceptance invariants")
    validate_parser.add_argument("--report", default="out/report.json")
    validate_parser.set_defaults(func=_validate_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
