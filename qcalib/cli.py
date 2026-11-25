"""Command line entry point for the calibration pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import commands as commands_mod
from . import loaders, receipts
from .errors import FailureList
from .hashing import stable_json
from .pipeline import PipelineInputs
from .runner import diagnostic_report, run_pipeline
from .snapshot import ForcedAbort, SnapshotStore


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def _load_all(args, failures: FailureList) -> PipelineInputs:
    layers = loaders.load_weight_package(args.weights, failures)
    calibration = loaders.load_calibration(args.calibration, layers, failures)
    config = loaders.load_config(args.config)
    if args.commands:
        command_list = loaders.load_command_stream(args.commands, failures)
        config, layers = commands_mod.apply_commands(
            config, layers, command_list, failures)
    receipt_entries = receipts.parse_receipt_dir(config.receipts_dir, failures)
    return PipelineInputs(layers=layers, calibration=calibration,
                          config=config, receipt_entries=receipt_entries,
                          input_failures=failures.to_list())


def _cmd_run(args) -> int:
    failures = FailureList()
    inputs = _load_all(args, failures)
    store = SnapshotStore(args.state, fault_point=args.fault)
    previous_cache = None
    ledger_path = Path(args.output)
    if ledger_path.exists() and not args.fresh:
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        previous_cache = previous.get("cache")
    try:
        result = run_pipeline(inputs, store, previous_cache=previous_cache)
    except ForcedAbort as exc:
        print(f"forced abort: {exc}", file=sys.stderr)
        return 2
    result.setdefault("failures", [])
    _write(ledger_path, result)
    if args.report:
        _write(Path(args.report), diagnostic_report(result))
    print(json.dumps({"layers": len(result["layers"]),
                      "failures": len(result["failures"]),
                      "output": str(ledger_path)}, ensure_ascii=False))
    return 0


def _cmd_recover(args) -> int:
    failures = FailureList()
    inputs = _load_all(args, failures)
    store = SnapshotStore(args.state)
    from .pipeline import config_fingerprint
    config_hash = config_fingerprint(inputs.config)
    content_versions = {layer.name: layer.content_version
                        for layer in inputs.layers}
    payload = store.recoverable_stages(config_hash, content_versions)
    recovered = None
    if "ledger_aggregation" in payload:
        document = store.load_stage("ledger_aggregation")
        recovered = document["payload"] if document else None
    if recovered is None:
        print(json.dumps({"recovered": False,
                          "recoverable_stages": payload}))
        return 1
    _write(Path(args.output), recovered)
    print(json.dumps({"recovered": True, "output": args.output,
                      "recoverable_stages": payload}))
    return 0


def _cmd_diff(args) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    ignore = set(args.ignore or [])
    left = {k: v for k, v in left.items() if k not in ignore}
    right = {k: v for k, v in right.items() if k not in ignore}
    equal = stable_json(left) == stable_json(right)
    print(json.dumps({"equal": equal, "ignored": sorted(ignore)}))
    return 0 if equal else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qcalib")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the calibration pipeline")
    run.add_argument("--weights", required=True)
    run.add_argument("--calibration", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--commands")
    run.add_argument("--state", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--report")
    run.add_argument("--fresh", action="store_true")
    run.add_argument("--fault")
    run.set_defaults(func=_cmd_run)

    recover = sub.add_parser("recover", help="recover ledger from snapshots")
    recover.add_argument("--weights", required=True)
    recover.add_argument("--calibration", required=True)
    recover.add_argument("--config", required=True)
    recover.add_argument("--commands")
    recover.add_argument("--state", required=True)
    recover.add_argument("--output", required=True)
    recover.set_defaults(func=_cmd_recover)

    diff = sub.add_parser("diff", help="field-level ledger comparison")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.add_argument("--ignore", nargs="*")
    diff.set_defaults(func=_cmd_diff)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
