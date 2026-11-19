"""Command line interface for the offline disaster-recovery decision bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_text
from .engine import PATHS
from .errors import BenchError
from .injection import (
    FAULT_BAD_RATIO,
    FAULT_CORRUPT_SCENE,
    FAULT_MISSING_SNAPSHOT_VERSION,
    StructuredLogger,
    exercise_fault,
    strip_snapshot_version,
)
from .ledger import Ledger, read_snapshot
from .loader import load_scene
from .planner import select_path
from .recovery import replay_from_ledger
from .samplegen import write_samples
from .session import assert_paths_identical, crash_and_recover, run_and_persist


def _print(payload: Any) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))


def _bench_error(exc: BenchError) -> int:
    _print(exc.to_dict())
    return 2


def cmd_validate(args: argparse.Namespace) -> int:
    scene = load_scene(args.scene)
    _print({"ok": True, "scene_id": scene.scene_id, "scale": scene.scale()})
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    scene = load_scene(args.scene)
    result = run_and_persist(
        scene,
        args.db,
        path_name=args.path,
        snapshot_dir=args.snapshot_dir,
    )
    _print(result)
    return 0


def cmd_consistency(args: argparse.Namespace) -> int:
    scene = load_scene(args.scene)
    report = assert_paths_identical(scene)
    payload = {
        "ok": True,
        "scene_id": scene.scene_id,
        "paths": list(PATHS),
        "rounds": report["rounds"],
        "canonical_bytes_per_round": report["canonical_bytes_per_round"],
        "change_ratio": report["change_ratio"],
        "planner": report["planner"],
    }
    _print(payload)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    scene = load_scene(args.scene)
    report = assert_paths_identical(scene)
    plan = select_path(scene.scale(), report["change_ratio"])
    _print(plan)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    scene = load_scene(args.scene)
    ledger = Ledger(args.db)
    try:
        report = replay_from_ledger(scene, ledger, args.path)
    finally:
        ledger.close()
    _print(
        {
            "verified_rounds": report.verified_rounds,
            "rerun_rounds": report.rerun_rounds,
            "digests": report.digests,
        }
    )
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    scene = load_scene(args.scene)
    payload = read_snapshot(args.snapshot_dir, scene.scene_id)
    _print(payload)
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log)
    context: dict[str, Any] = {"ratio": args.ratio}
    if args.kind == FAULT_CORRUPT_SCENE:
        scene_path = Path(args.scene)
        context["scene_dict"] = json.loads(scene_path.read_text(encoding="utf-8"))
    if args.kind == FAULT_MISSING_SNAPSHOT_VERSION:
        scene = load_scene(args.scene)
        ledger = Ledger(args.db)
        try:
            ledger.initialize_session(scene, args.path)
            from .ledger import write_snapshot

            snapshot_file = write_snapshot(args.snapshot_dir, scene, ledger)
        finally:
            ledger.close()
        strip_snapshot_version(snapshot_file)
        context["snapshot_dir"] = args.snapshot_dir
        context["scene_id"] = scene.scene_id
    try:
        result = exercise_fault(args.kind, context, logger)
    except BenchError as exc:
        return _bench_error(exc)
    _print(result.to_dict())
    return 0 if result.ok else 2


def cmd_crash_demo(args: argparse.Namespace) -> int:
    scene = load_scene(args.scene)
    result = crash_and_recover(
        scene,
        args.db,
        crash_round=args.crash_round,
        log_path=args.log,
        snapshot_dir=args.snapshot_dir,
        path_name=args.path,
    )
    _print(result)
    return 0


def cmd_check_env(args: argparse.Namespace) -> int:
    import platform
    import sqlite3

    try:
        import numpy

        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    try:
        import tkinter  # noqa: F401

        tkinter_ok = True
    except Exception:
        tkinter_ok = False
    _print(
        {
            "python": platform.python_version(),
            "sqlite3": sqlite3.sqlite_version,
            "numpy": numpy_version,
            "tkinter_importable": tkinter_ok,
        }
    )
    return 0


def cmd_gen_samples(args: argparse.Namespace) -> int:
    paths = write_samples(args.output_dir)
    _print({key: str(value) for key, value in paths.items()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dr-bench",
        description="Offline disaster-recovery switch decision experiment bench",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-env", help="verify python/numpy/sqlite/tkinter")
    p.set_defaults(func=cmd_check_env)

    p = sub.add_parser("gen-samples", help="generate deterministic sample scenes")
    p.add_argument("--output-dir", default="samples")
    p.set_defaults(func=cmd_gen_samples)

    p = sub.add_parser("validate", help="strong-validate a scene file")
    p.add_argument("scene")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="run one path and persist the ledger")
    p.add_argument("scene")
    p.add_argument("--db", required=True)
    p.add_argument("--path", choices=PATHS, default=PATHS[1])
    p.add_argument("--snapshot-dir")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("consistency", help="byte-compare the three paths")
    p.add_argument("scene")
    p.set_defaults(func=cmd_consistency)

    p = sub.add_parser("plan", help="show adaptive path selection")
    p.add_argument("scene")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("replay", help="recover and replay against the ledger")
    p.add_argument("scene")
    p.add_argument("--db", required=True)
    p.add_argument("--path", choices=PATHS, default=PATHS[1])
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("snapshot", help="read and verify a snapshot descriptor")
    p.add_argument("scene")
    p.add_argument("--snapshot-dir", required=True)
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("inject", help="inject one of the three fault classes")
    p.add_argument(
        "--kind",
        choices=(
            FAULT_CORRUPT_SCENE,
            FAULT_BAD_RATIO,
            FAULT_MISSING_SNAPSHOT_VERSION,
        ),
        required=True,
    )
    p.add_argument("--scene")
    p.add_argument("--db")
    p.add_argument("--snapshot-dir")
    p.add_argument("--path", choices=PATHS, default=PATHS[1])
    p.add_argument("--ratio", type=float, default=1.5)
    p.add_argument("--log", required=True)
    p.set_defaults(func=cmd_inject)

    p = sub.add_parser("crash-demo", help="simulate forced termination and recover")
    p.add_argument("scene")
    p.add_argument("--db", required=True)
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--crash-round", type=int, required=True)
    p.add_argument("--path", choices=PATHS, default=PATHS[1])
    p.set_defaults(func=cmd_crash_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BenchError as exc:
        return _bench_error(exc)


if __name__ == "__main__":
    sys.exit(main())
