"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import ErrorCode, StudioError
from .injections import run_fault_matrix
from .ledger import verify_ledger_files
from .persistence import AtomicStateStore
from .render import OffscreenViewport, result_to_ppm
from .sample import SampleConfig, edit_sequence, load_sample_bundle, write_sample_bundle
from .snapshots import SnapshotStore, scene_from_engine
from .validation import run_validation


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pointcloud-studio")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate the deterministic 120k sample bundle")
    generate.add_argument("--out", required=True)

    validate = subparsers.add_parser("validate", help="run end-to-end acceptance validation")
    validate.add_argument("--sample", default="artifacts/sample")
    validate.add_argument("--workspace", default="artifacts/run_workspace")
    validate.add_argument("--report", default="artifacts/validation_report.json")
    validate.add_argument("--reuse-sample", action="store_true")
    validate.add_argument("--operations", type=int, default=None)

    verify = subparsers.add_parser("verify", help="independently recompute the current committed ledger")
    verify.add_argument("--workspace", required=True)

    verify_ledger = subparsers.add_parser(
        "verify-ledger",
        help="recompute a ledger from binary point/normal files and ledger parameters only",
    )
    verify_ledger.add_argument("--points", required=True)
    verify_ledger.add_argument("--normals", required=True)
    verify_ledger.add_argument("--ledger", required=True)

    render = subparsers.add_parser("render", help="render the current snapshot or report E_GL_CONTEXT")
    render.add_argument("--workspace", required=True)
    render.add_argument("--width", type=int, default=960)
    render.add_argument("--height", type=int, default=640)
    render.add_argument("--output-ppm", default=None)

    faults = subparsers.add_parser("inject-faults", help="run the serialized business-error matrix")
    faults.add_argument("--sample", default="artifacts/sample")
    faults.add_argument("--workspace", default="artifacts/run_workspace")
    faults.add_argument("--out", default="artifacts/fault_matrix")

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            manifest = write_sample_bundle(args.out, SampleConfig())
            _print(
                {
                    "ok": True,
                    "out": str(Path(args.out)),
                    "point_count": manifest["point_count"],
                    "bounds": manifest["bounds"],
                    "confidence": manifest["normal_confidence"],
                    "files": manifest["files"],
                }
            )
            return 0
        if args.command == "validate":
            report = run_validation(
                sample_dir=args.sample,
                workspace_dir=args.workspace,
                report_path=args.report,
                regenerate_sample=not args.reuse_sample,
                operations_limit=args.operations,
            )
            _print(
                {
                    "ok": report["passed"],
                    "report": str(Path(args.report)),
                    "checks": report["checks"],
                    "final_committed_version": report["final_committed_version"],
                }
            )
            return 0 if report["passed"] else 1
        if args.command == "verify":
            store = AtomicStateStore(args.workspace)
            result = store.independent_verify_current()
            _print(result)
            return 0
        if args.command == "verify-ledger":
            result = verify_ledger_files(args.points, args.normals, args.ledger)
            _print(result)
            return 0
        if args.command == "render":
            loaded = AtomicStateStore(args.workspace).load_current()
            snapshots = SnapshotStore()
            scene = snapshots.publish(scene_from_engine(loaded.engine))
            result = OffscreenViewport(snapshots).render(
                width=args.width,
                height=args.height,
                expected_version=scene.version,
            )
            if result.ok and args.output_ppm:
                result_to_ppm(result, args.output_ppm)
            _print(result.to_dict())
            return 0 if result.ok or result.code == ErrorCode.E_GL_CONTEXT.value else 1
        if args.command == "inject-faults":
            report = run_fault_matrix(
                sample_dir=args.sample,
                workspace=args.workspace,
                output_dir=args.out,
            )
            _print(report)
            return 0 if report["all_matched"] else 1
        raise AssertionError("unreachable command")
    except StudioError as exc:
        _print(exc.to_dict())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
