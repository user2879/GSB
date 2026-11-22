"""Command line interface for sample generation, runs, and recovery checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.generate_samples import DEFAULT_SEED, generate

from .faults import run_fault_suite
from .runner import run_workbench_demo
from .snapshot import SnapshotStore


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def command_generate(args: argparse.Namespace) -> int:
    manifest = generate(Path(args.out), args.seed)
    _print(manifest)
    return 0


def command_run(args: argparse.Namespace) -> int:
    snapshot_root = Path(args.snapshots)
    if args.clean_snapshots:
        SnapshotStore.remove(snapshot_root)
        crash_root = snapshot_root.parent / (snapshot_root.name + "-crash")
        SnapshotStore.remove(crash_root)
    summary = run_workbench_demo(
        data_dir=args.data,
        injections_file=args.injections,
        snapshot_root=snapshot_root,
        artifact_dir=args.out,
        workers=args.workers,
        with_faults=not args.no_faults,
    )
    _print(summary)
    return 0 if summary.get("status") == "PASS" else 2


def command_faults(args: argparse.Namespace) -> int:
    report = run_fault_suite(args.injections)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _print(report["summary"])
    return 0 if report["summary"]["all_detected"] else 2


def command_recover(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.snapshots)
    inspection = store.inspect(
        expected_version=args.version,
        expected_config_hash=args.config_hash,
    )
    _print(inspection)
    return 0 if inspection.get("reusable") else 1


def command_demo(args: argparse.Namespace) -> int:
    manifest = generate(Path(args.data), args.seed)
    print(
        json.dumps(
            {"generated": manifest["counts"], "seed": manifest["seed"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    namespace = argparse.Namespace(
        data=str(Path(args.data) / "base"),
        injections=str(Path(args.data) / "injected" / "injections.json"),
        snapshots=args.snapshots,
        out=args.out,
        workers=args.workers,
        no_faults=False,
        clean_snapshots=True,
    )
    return command_run(namespace)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic e-commerce query intent diagnosis workbench"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate deterministic samples")
    generate_parser.add_argument("--out", default="data")
    generate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate_parser.set_defaults(handler=command_generate)

    run_parser = subparsers.add_parser("run", help="run incremental verification")
    run_parser.add_argument("--data", default="data/base")
    run_parser.add_argument("--injections", default="data/injected/injections.json")
    run_parser.add_argument("--snapshots", default=".snapshots")
    run_parser.add_argument("--out", default="artifacts")
    run_parser.add_argument("--workers", type=int, default=None)
    run_parser.add_argument("--no-faults", action="store_true")
    run_parser.add_argument(
        "--keep-snapshots",
        dest="clean_snapshots",
        action="store_false",
        help="do not clear existing snapshots before the run",
    )
    run_parser.set_defaults(clean_snapshots=True, handler=command_run)

    faults_parser = subparsers.add_parser("faults", help="run isolated fault suite")
    faults_parser.add_argument("--injections", default="data/injected/injections.json")
    faults_parser.add_argument("--out", default="artifacts/fault_report.json")
    faults_parser.set_defaults(handler=command_faults)

    recover_parser = subparsers.add_parser("recover", help="check snapshot recovery status")
    recover_parser.add_argument("--snapshots", default=".snapshots")
    recover_parser.add_argument("--version", type=int, required=True)
    recover_parser.add_argument("--config-hash", required=True)
    recover_parser.set_defaults(handler=command_recover)

    demo_parser = subparsers.add_parser("demo", help="generate samples and run verification")
    demo_parser.add_argument("--data", default="data")
    demo_parser.add_argument("--snapshots", default=".snapshots")
    demo_parser.add_argument("--out", default="artifacts")
    demo_parser.add_argument("--workers", type=int, default=None)
    demo_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    demo_parser.set_defaults(handler=command_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
