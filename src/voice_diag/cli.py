from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .errors import DiagnosticError
from .pipeline import DiagnosticPipeline, PipelineOptions
from .receipts import (
    compare_normalized_receipts,
    load_receipt_a1,
    load_receipt_a2,
    to_a1,
    to_a2,
)
from .synthesize import generate_demo
from .util import atomic_write_json
from .viewer import launch_viewer


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="path to diagnostic config JSON")
    parser.add_argument("--manifest", required=True, help="path to session manifest JSON")
    parser.add_argument("--out", required=True, help="snapshot/output directory")
    parser.add_argument("--max-batches", type=int, default=128, help="maximum point-budget batches")
    parser.add_argument("--max-workers", type=int, default=None, help="cap; cannot exceed config value")
    parser.add_argument(
        "--arrival-limit",
        type=int,
        default=None,
        help="process only fragments with arrival_index below this rolling-arrival cursor",
    )
    parser.add_argument(
        "--crash-at",
        default=None,
        choices=("after_parse", "after_endpoint", "after_batch", "all"),
        help="test-only forced process termination hook",
    )


def _command_run(args: argparse.Namespace) -> int:
    pipeline = DiagnosticPipeline(
        args.config,
        args.manifest,
        args.out,
        options=PipelineOptions(
            max_workers=args.max_workers,
            max_batches=max(1, args.max_batches),
            crash_at=args.crash_at,
            arrival_limit=args.arrival_limit,
        ),
    )
    try:
        result = pipeline.run()
    except DiagnosticError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    if not result.get("complete"):
        return 2
    return 0


def _command_generate(args: argparse.Namespace) -> int:
    paths = generate_demo(args.out, seed=args.seed, overwrite=not args.no_overwrite)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


def _command_demo(args: argparse.Namespace) -> int:
    sample = Path(args.out) / "sample"
    diagnostics = Path(args.out) / "diagnostics"
    generate_demo(sample, seed=args.seed, overwrite=True)
    pipeline = DiagnosticPipeline(
        sample / "config.json",
        sample / "manifest.json",
        diagnostics,
        options=PipelineOptions(max_workers=args.max_workers, max_batches=args.max_batches),
    )
    result = pipeline.run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("complete") else 2


def _command_receipts(args: argparse.Namespace) -> int:
    try:
        a1 = load_receipt_a1(args.a1) if args.a1 else None
        a2 = load_receipt_a2(args.a2) if args.a2 else None
    except DiagnosticError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    payload: dict[str, Any] = {}
    if a1 is not None and a2 is not None:
        payload["comparison"] = compare_normalized_receipts(a1, a2)
    if a1 is not None:
        payload["a1"] = a1.as_dict()
        if args.out_prefix:
            conversion = to_a2(a1, vocabulary=json.loads(args.vocabulary_json) if args.vocabulary_json else None)
            Path(f"{args.out_prefix}.a2.json").write_text(conversion.text, encoding="utf-8")
            atomic_write_json(Path(f"{args.out_prefix}.a2-sidecar.json"), conversion.sidecar)
    if a2 is not None:
        payload["a2"] = a2.as_dict()
        if args.out_prefix:
            conversion = to_a1(a2, vocabulary=json.loads(args.vocabulary_json) if args.vocabulary_json else None)
            Path(f"{args.out_prefix}.a1").write_text(conversion.text, encoding="utf-8")
            atomic_write_json(Path(f"{args.out_prefix}.a1-sidecar.json"), conversion.sidecar)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _command_view(args: argparse.Namespace) -> int:
    try:
        launch_viewer(args.out)
    except Exception as exc:  # GUI/display failures should not print a traceback
        print(json.dumps({"code": "E_VIEWER", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-diag",
        description="Deterministic local diagnostics for overlapping multi-speaker command sessions",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate", help="create deterministic local sample data")
    generate_parser.add_argument("--out", required=True)
    generate_parser.add_argument("--seed", type=int, default=20261127)
    generate_parser.add_argument("--no-overwrite", action="store_true")
    generate_parser.set_defaults(func=_command_generate)

    run_parser = subparsers.add_parser("run", help="run or resume diagnostics")
    _add_run_arguments(run_parser)
    run_parser.set_defaults(func=_command_run)

    demo_parser = subparsers.add_parser("demo", help="generate and run the built-in sample")
    demo_parser.add_argument("--out", required=True)
    demo_parser.add_argument("--seed", type=int, default=20261127)
    demo_parser.add_argument("--max-batches", type=int, default=128)
    demo_parser.add_argument("--max-workers", type=int, default=None)
    demo_parser.set_defaults(func=_command_demo)

    receipts_parser = subparsers.add_parser("receipts", help="parse and compare a1/a2 evidence")
    receipts_parser.add_argument("--a1")
    receipts_parser.add_argument("--a2")
    receipts_parser.add_argument("--out-prefix")
    receipts_parser.add_argument("--vocabulary-json", help='JSON object such as {"1":"开灯"}')
    receipts_parser.set_defaults(func=_command_receipts)

    view_parser = subparsers.add_parser("view", help="open a Tkinter review window for committed outputs")
    view_parser.add_argument("--out", required=True, help="diagnostic output directory")
    view_parser.set_defaults(func=_command_view)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
