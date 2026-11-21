"""Command line interface for generation, runs, recovery and acceptance tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .demo import run_acceptance_demo
from .engine import RunOptions, run_until_complete
from .errors import SegmentationError
from .injections import materialize_injections
from .patches import apply_patch_batch
from .runtime_control import STOP_EVENT, install_signal_handlers, restore_signal_handlers
from .samplegen import generate_sample_workspace
from .snapshot import SnapshotStore
from .verify import compare_states, load_state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m remote_seg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--out", type=Path, default=Path("sample_data"))
    generate.add_argument("--width", type=int, default=48)
    generate.add_argument("--height", type=int, default=48)
    generate.add_argument("--rows", type=int, default=2)
    generate.add_argument("--cols", type=int, default=3)
    generate.add_argument("--overlap", type=int, default=8)
    generate.add_argument("--seed", type=int, default=20261122)
    generate.add_argument("--patch-batches", type=int, default=2)

    run = subparsers.add_parser("run", help="run one or more incremental budget batches")
    run.add_argument("--index", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--fusion-versions", type=Path, required=True)
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--report", type=Path)
    run.add_argument("--full", action="store_true", dest="full_rebuild")
    run.add_argument("--strict-recovery", action="store_true")
    run.add_argument("--drain", action="store_true", help="repeat budget batches until settled")
    run.add_argument("--max-batches", type=int, default=100)
    run.add_argument("--injection-manifest", type=Path)

    recover = subparsers.add_parser("recover", help="inspect the only recoverable snapshot point")
    recover.add_argument("--state-dir", type=Path, required=True)
    recover.add_argument("--config", type=Path)
    recover.add_argument("--index", type=Path)
    recover.add_argument("--strict", action="store_true")

    patch = subparsers.add_parser("patch", help="apply one deterministic patch batch")
    patch.add_argument("--index", type=Path, required=True)
    patch.add_argument("--plan", type=Path, required=True)
    patch.add_argument("--batch", type=int, required=True)

    inject = subparsers.add_parser("inject", help="materialize an injection profile on a workspace copy")
    inject.add_argument("--source", type=Path, required=True)
    inject.add_argument("--out", type=Path, required=True)
    inject.add_argument("--plan", type=Path, required=True)
    inject.add_argument("--profile", required=True)
    inject.add_argument("--index-name", default="index.json")
    inject.add_argument("--config-name", default="segmentation.json")

    compare = subparsers.add_parser("compare", help="compare two snapshot state directories")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)

    demo = subparsers.add_parser("demo", help="run the full acceptance edit sequence")
    demo.add_argument("--out", type=Path, default=Path("runtime/demo"))
    demo.add_argument("--sample-dir", type=Path, default=Path("sample_data"))
    demo.add_argument("--config", type=Path, default=Path("configs/segmentation.json"))
    demo.add_argument(
        "--sequence", type=Path, default=Path("configs/edit_sequence.json")
    )
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _run_command(args: argparse.Namespace) -> int:
    previous_handlers = install_signal_handlers()
    try:
        options = RunOptions(
            index_path=args.index,
            config_path=args.config,
            fusion_versions_path=args.fusion_versions,
            state_dir=args.state_dir,
            report_path=args.report,
            full_rebuild=args.full_rebuild,
            strict_recovery=args.strict_recovery,
            injection_manifest_path=args.injection_manifest,
            stop_event=STOP_EVENT,
            max_batches=args.max_batches if args.drain else 1,
        )
        report = run_until_complete(options)
        _print(report)
        if report["status"] == "graceful_stop":
            return 0
        if report["status"] == "failed":
            return 1
        if report["status"] == "budget_pending":
            return 3
        return 0
    finally:
        restore_signal_handlers(previous_handlers)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            result = generate_sample_workspace(
                args.out,
                width=args.width,
                height=args.height,
                rows=args.rows,
                cols=args.cols,
                overlap=args.overlap,
                seed=args.seed,
                patch_batches=args.patch_batches,
            )
            _print({key: value for key, value in result.items() if key not in {"index", "fusion_versions", "patch_plan"}})
            return 0
        if args.command == "run":
            return _run_command(args)
        if args.command == "recover":
            store = SnapshotStore(args.state_dir)
            expected_config = None
            expected_index = None
            if args.strict and args.config:
                from .inputs import load_config

                expected_config = load_config(args.config).hash
            if args.strict and args.index:
                from .inputs import load_index

                expected_index = load_index(args.index).index_hash
            state = store.load(
                expected_config_hash=expected_config,
                expected_index_hash=expected_index,
            )
            if state is None:
                _print({"status": "empty", "codes": []})
                return 4
            _print({"status": "recovered", "recovery_point": state.recovery_point})
            return 0
        if args.command == "patch":
            records = apply_patch_batch(args.index, args.plan, args.batch)
            _print(
                {
                    "status": "ok",
                    "records": [
                        {
                            "batch": record.batch,
                            "tile_id": record.tile_id,
                            "rect": list(record.rect),
                            "before_fingerprint": record.before_fingerprint,
                            "after_fingerprint": record.after_fingerprint,
                        }
                        for record in records
                    ],
                }
            )
            return 0
        if args.command == "inject":
            manifest = materialize_injections(
                args.source,
                args.out,
                args.plan,
                profile=args.profile,
                index_name=args.index_name,
                config_name=args.config_name,
            )
            _print(manifest)
            return 0
        if args.command == "compare":
            result = compare_states(load_state(args.left), load_state(args.right))
            _print(result)
            return 0 if result["equal"] else 1
        if args.command == "demo":
            result = run_acceptance_demo(
                args.out,
                sample_dir=args.sample_dir,
                config_path=args.config,
                sequence_path=args.sequence,
            )
            _print(result)
            return 0 if result["status"] == "pass" else 1
    except SegmentationError as exc:
        _print({"status": "failed", "code": exc.code, "message": exc.message, "details": exc.details})
        return 1
    except FileNotFoundError as exc:
        _print({"status": "failed", "code": "E_NOT_FOUND", "message": str(exc)})
        return 1
    return 2
