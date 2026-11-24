from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, canonical_json
from .engine import run_engine
from .errors import MMWindowError
from .generate import DEFAULT_SEED, generate_samples


ROOT = Path(__file__).resolve().parents[1]


def _safe_checkpoint(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("refusing to use a filesystem root as checkpoint directory")
    return resolved


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    windows = report["windows"]
    return {
        "ok": not any(
            failure["code"]
            in {"E_EVENT_INVALID", "E_WINDOW_OVERFLOW", "E_EVENT_BUDGET_PENDING"}
            for failure in report["failures"]
        ),
        "window_count": len(windows),
        "closed_window_count": sum(1 for window in windows if window.get("closed")),
        "empty_window_count": len(report["empty_windows"]),
        "empty_windows": report["empty_windows"],
        "resettlement_count": len(report["resettlements"]),
        "failure_count": len(report["failures"]),
        "failure_codes": sorted({failure["code"] for failure in report["failures"]}),
        "watermark_ms": report["watermark"]["final_watermark_ms"],
        "max_seen_timestamp_ms": report["watermark"]["max_seen_timestamp_ms"],
        "processed_arrival_count": report["meta"]["processed_arrival_count"],
        "arrival_count": report["meta"]["arrival_count"],
        "pending_count": len(report["pending"]),
        "checkpoint": report["checkpoint"],
        "diagnostics": report["diagnostics"],
        "cache_stats": report["cache"]["stats"],
        "quantization_level": report["meta"]["quantization_level"],
        "fractional_bits": report["meta"]["fractional_bits"],
    }


def _cmd_generate(args: argparse.Namespace) -> int:
    summary = generate_samples(args.samples, seed=args.seed, clean=not args.keep)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    sample_dir = Path(args.samples)
    config_path = Path(args.config) if args.config else sample_dir / "window_config.json"
    plan_path = Path(args.plan) if args.plan else sample_dir / "injection_plan.json"
    checkpoint = _safe_checkpoint(Path(args.checkpoint))
    report = run_engine(
        sample_dir,
        config_path,
        plan_path,
        checkpoint,
        batch_event_budget=args.batch_budget,
        max_batches=args.max_batches,
        reset=args.reset,
        frames_jsonl=args.frames_jsonl,
        text_jsonl=args.text_jsonl,
        sensor_jsonl=args.sensor_jsonl,
    )
    if args.output:
        atomic_write_json(Path(args.output), report)
    if args.print_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif not args.quiet:
        print(json.dumps(_summary(report), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mmwin",
        description="Deterministic multimodal event-window aggregation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--samples", "--sample-dir", dest="samples", default=str(ROOT / "samples"))
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument("--keep", action="store_true")
    generate.set_defaults(func=_cmd_generate)

    run = subparsers.add_parser("run", help="run incremental aggregation")
    run.add_argument("--samples", "--sample-dir", dest="samples", default=str(ROOT / "samples"))
    run.add_argument("--config", "--window-config", dest="config", default=None)
    run.add_argument("--plan", "--injection-plan", dest="plan", default=None)
    run.add_argument(
        "--checkpoint",
        "--checkpoint-dir",
        dest="checkpoint",
        default=str(ROOT / ".mmwin-checkpoint"),
    )
    run.add_argument("--output", default=None)
    run.add_argument("--batch-budget", "--batch-size", dest="batch_budget", type=int, default=None)
    run.add_argument("--max-batches", type=int, default=None)
    run.add_argument("--reset", action="store_true", help="delete checkpoint and full replay")
    run.add_argument("--frames-jsonl", default=None)
    run.add_argument("--text-jsonl", default=None)
    run.add_argument("--sensor-jsonl", default=None)
    run.add_argument("--print-json", action="store_true", help="print the complete report")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (MMWindowError, ValueError, OSError) as exc:
        code = getattr(exc, "code", "E_RUNTIME")
        print(canonical_json({"code": code, "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
