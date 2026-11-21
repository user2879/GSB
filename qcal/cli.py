"""Command-line entry point for the quantization component and example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .acceptance import run_acceptance
from .calibration import (
    downgrade_v2_to_v1,
    parse_calibration_file,
    reverse_downgrade,
    upgrade_v1_to_v2,
    write_downgrade_artifacts,
)
from .engine import ForcedInterrupt, QcalEngine, RunPaths, snapshot_summary
from .io_utils import atomic_write_json


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", default="samples", help="sample bundle directory")
    parser.add_argument("--config", default=None, help="override config path")


def command_run(args: argparse.Namespace) -> int:
    paths = RunPaths.discover(args.samples, config=args.config)
    engine = QcalEngine(
        paths,
        output_dir=args.output,
        state_dir=args.state,
        fresh=args.fresh,
        injection_path=args.injections,
        force_interrupt_after_stage=args.interrupt_after,
        budget_override=args.budget,
        workers_override=args.workers,
    )
    try:
        report = engine.run()
    except ForcedInterrupt as exc:
        summary = snapshot_summary(args.state)
        print(json.dumps({"status": "interrupted", "exit_code": exc.exit_code, "stage": exc.stage, **summary}, indent=2))
        return exc.exit_code
    print(
        json.dumps(
            {
                "result_digest": report["result_digest"],
                "failures": report["failure_count"],
                "pending": report["pending_count"],
                "report": str(Path(args.output) / "run_report.json"),
            },
            indent=2,
        )
    )
    return 0


def command_acceptance(args: argparse.Namespace) -> int:
    report = run_acceptance(args.samples, args.artifacts)
    print(json.dumps(report["verification"], indent=2))
    return 0 if all(report["verification"].values()) else 1


def command_convert(args: argparse.Namespace) -> int:
    paths = RunPaths.discover(args.samples, config=args.config)
    weight_layers = set()
    with open(paths.weights, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                weight_layers.add(str(json.loads(line)["layer"]))
    parsed_v1 = parse_calibration_file(paths.calibration_v1, "v1", weight_layers)
    parsed_v2 = parse_calibration_file(paths.calibration_v2, "v2", weight_layers)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.direction in ("v1-to-v2", "both", "roundtrip"):
        upgraded = upgrade_v1_to_v2(parsed_v1.records)
        atomic_write_json(
            output / "v1_to_v2.json",
            {"exchange_version": "v2", "layers": [upgraded[layer].to_dict() for layer in sorted(upgraded)]},
        )
    if args.direction in ("v2-to-v1", "both", "roundtrip"):
        downgraded, sidecar, thousandths = downgrade_v2_to_v1(parsed_v2.records)
        text_path, sidecar_path = write_downgrade_artifacts(output, downgraded, sidecar, thousandths)
        restored, failures, status = reverse_downgrade(downgraded, sidecar_path)
        atomic_write_json(
            output / "reverse_status.json",
            {
                "status": status,
                "restored": {layer: record.to_dict() for layer, record in sorted(restored.items())},
                "failures": [failure.to_dict() for failure in failures],
                "v1_text": text_path.name,
                "sidecar": sidecar_path.name,
            },
        )
    print(json.dumps({"status": "ok", "output": str(output), "direction": args.direction}, indent=2))
    return 0


def command_status(args: argparse.Namespace) -> int:
    print(json.dumps(snapshot_summary(args.state), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic quantization calibration and distillation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run or resume the staged pipeline")
    _add_common(run_parser)
    run_parser.add_argument("--output", default="artifacts/run")
    run_parser.add_argument("--state", default="artifacts/state")
    run_parser.add_argument("--fresh", action="store_true")
    run_parser.add_argument("--injections", default=None)
    run_parser.add_argument("--interrupt-after", default=None)
    run_parser.add_argument("--budget", type=int, default=None)
    run_parser.add_argument("--workers", type=int, default=None)
    run_parser.set_defaults(func=command_run)

    acceptance_parser = subparsers.add_parser("acceptance", help="run the local acceptance suite")
    _add_common(acceptance_parser)
    acceptance_parser.add_argument("--artifacts", default="artifacts")
    acceptance_parser.set_defaults(func=command_acceptance)

    convert_parser = subparsers.add_parser("convert", help="exercise v1/v2 conversion and sidecar")
    _add_common(convert_parser)
    convert_parser.add_argument("--direction", choices=["v1-to-v2", "v2-to-v1", "both", "roundtrip"], default="both")
    convert_parser.add_argument("--output", default="artifacts/conversion")
    convert_parser.set_defaults(func=command_convert)

    status_parser = subparsers.add_parser("status", help="show resumable snapshot status")
    status_parser.add_argument("--state", default="artifacts/state")
    status_parser.set_defaults(func=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
