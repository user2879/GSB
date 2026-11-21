"""Command-line entry points for the local arbitration workbench."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from .config import parse_config
from .generator import generate_samples
from .numeric import read_json
from .versions import convert_file
from .workbench import Workbench


def _print_json(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _root(args: argparse.Namespace) -> Path:
    return Path(args.root).resolve()


def _workbench(args: argparse.Namespace) -> Workbench:
    return Workbench(
        root=_root(args),
        rules_path=args.rules,
        config_path=args.config,
        evidence_path=args.evidence,
        exception_injections_path=args.exceptions,
        output_dir=args.output_dir,
        persist_edits=True,
    )


def _install_signal_handlers(workbench: Workbench) -> dict:
    state = {"signals": []}

    def handler(signum, _frame):
        state["signals"].append(int(signum))
        workbench.request_stop()

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            try:
                signal.signal(getattr(signal, signal_name), handler)
            except ValueError:
                pass
    return state


def command_generate(args: argparse.Namespace) -> int:
    manifest = generate_samples(_root(args), seed=args.seed, generate_version_samples=not args.no_versions)
    _print_json({"status": "ok", "manifest": manifest})
    return 0


def command_run(args: argparse.Namespace) -> int:
    workbench = _workbench(args)
    signals = _install_signal_handlers(workbench)
    result = workbench.run(use_cache=not args.fresh, mode="full" if args.fresh else "incremental")
    payload = {
        "status": "stopped" if result.interrupted else "ok",
        "signals": signals["signals"],
        "startup": workbench.startup_status,
        "summary": result.summary(),
        "snapshot_version": result.global_version,
        "recovery": workbench.recovery_info,
        "outputs": result.output_info.get("paths", {}),
        "diagnostic_path": str(workbench.root / args.output_dir / "diagnostic_report.json"),
    }
    _print_json(payload)
    return 0 if not result.interrupted or signals["signals"] else 0


def command_status(args: argparse.Namespace) -> int:
    workbench = _workbench(args)
    _print_json({"status": "ok", **workbench.status()})
    return 0


def command_edit(args: argparse.Namespace) -> int:
    workbench = _workbench(args)
    signals = _install_signal_handlers(workbench)
    if args.edit_file:
        edit = read_json(args.edit_file)
    else:
        edit = json.loads(args.edit)
    if args.no_run:
        affected = workbench.apply_edit(edit)
        _print_json({"status": "edited", "affected_tickets": affected})
        return 0
    outcome = workbench.apply_edit_and_run(
        edit,
        budget_ms=args.budget_ms if args.budget_ms is not None else args.budget,
    )
    outcome["signals"] = signals["signals"]
    _print_json(outcome)
    return 0


def command_convert(args: argparse.Namespace) -> int:
    config = parse_config(read_json(args.config_file))
    report = convert_file(
        args.source,
        args.source_version,
        args.target,
        args.target_version,
        config,
        source_sidecar_path=args.source_sidecar,
        target_sidecar_path=args.target_sidecar,
        temp_dir=_root(args) / config.temp_dir,
    )
    _print_json({"status": "ok", "report": report})
    return 0


def command_compare(args: argparse.Namespace) -> int:
    workbench = _workbench(args)
    result = workbench.run(use_cache=not args.fresh, mode="incremental")
    comparison = workbench.compare_incremental_with_full(result)
    _print_json(
        {
            "status": "ok",
            "summary": result.summary(),
            "concurrency": result.diagnostics.get("concurrency", {}),
            "comparison": comparison,
        }
    )
    return 0 if comparison["equal"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arbitration-workbench", description=__doc__)
    parser.add_argument("--root", default=".", help="workbench root directory")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--seed", type=int, default=20261121)
    generate.add_argument("--no-versions", action="store_true")
    generate.set_defaults(func=command_generate)

    run = sub.add_parser("run", help="run the arbitration pipeline")
    run.add_argument("--rules", default="samples/rules.json")
    run.add_argument("--config", default="samples/config.json")
    run.add_argument("--evidence", default="samples/evidence.jsonl")
    run.add_argument("--exceptions", default="samples/exception_injections.json")
    run.add_argument("--output-dir", default="outputs")
    run.add_argument("--fresh", action="store_true", help="ignore wake snapshots/cache")
    run.set_defaults(func=command_run)

    status = sub.add_parser("status", help="inspect snapshot and latest summary")
    status.add_argument("--rules", default="samples/rules.json")
    status.add_argument("--config", default="samples/config.json")
    status.add_argument("--evidence", default="samples/evidence.jsonl")
    status.add_argument("--exceptions", default="samples/exception_injections.json")
    status.add_argument("--output-dir", default="outputs")
    status.set_defaults(func=command_status)

    edit = sub.add_parser("edit", help="apply one edit command and run within budget")
    edit.add_argument("--rules", default="samples/rules.json")
    edit.add_argument("--config", default="samples/config.json")
    edit.add_argument("--evidence", default="samples/evidence.jsonl")
    edit.add_argument("--exceptions", default="samples/exception_injections.json")
    edit.add_argument("--output-dir", default="outputs")
    source = edit.add_mutually_exclusive_group(required=True)
    source.add_argument("--edit", help="edit JSON object")
    source.add_argument("--edit-file", help="path to edit JSON object")
    edit.add_argument("--budget-ms", type=int, default=None)
    edit.add_argument("--budget", type=int, default=None)
    edit.add_argument("--no-run", action="store_true")
    edit.set_defaults(func=command_edit)

    convert = sub.add_parser("convert", help="convert one exchange version to another")
    convert.add_argument("--config-file", default="samples/config.json")
    convert.add_argument("--source", required=True)
    convert.add_argument("--source-version", choices=("v1", "v2", "v3"), required=True)
    convert.add_argument("--source-sidecar")
    convert.add_argument("--target", required=True)
    convert.add_argument("--target-version", choices=("v1", "v2", "v3"), required=True)
    convert.add_argument("--target-sidecar")
    convert.set_defaults(func=command_convert)

    compare = sub.add_parser("compare", help="run incremental arbitration and compare with a blank full rebuild")
    compare.add_argument("--rules", default="samples/rules.json")
    compare.add_argument("--config", default="samples/config.json")
    compare.add_argument("--evidence", default="samples/evidence.jsonl")
    compare.add_argument("--exceptions", default="samples/exception_injections.json")
    compare.add_argument("--output-dir", default="outputs")
    compare.add_argument("--fresh", action="store_true")
    compare.set_defaults(func=command_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Accept --root before or after the subcommand for a friendlier demo UX.
    if "--root" in argv:
        index = argv.index("--root")
        if index > 0 and index + 1 < len(argv):
            root_value = argv[index + 1]
            del argv[index : index + 2]
            argv = ["--root", root_value] + argv
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        payload = {
            "status": "error",
            "code": getattr(exc, "code", "E_WORKBENCH_FAILED"),
            "message": str(exc),
        }
        details = getattr(exc, "details", None)
        if details:
            payload["details"] = details
        _print_json(payload)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
