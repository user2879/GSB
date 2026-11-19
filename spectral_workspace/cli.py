"""Command line interface for the spectral material workspace."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Sequence

from .canonical import atomic_write_json, canonical_bytes, read_json, sha256_obj
from .demo import run_crash_worker, run_demo, workspace_report
from .errors import DomainError
from .model import parse_scene
from .persistence import PersistenceStore
from .reporting import build_report
from .synth import (
    DEFAULT_SEED,
    demo_operations,
    generate_sample_scene,
    write_sample_scene,
)
from .workspace import open_workspace


def _emit(document: Any) -> None:
    sys.stdout.buffer.write(canonical_bytes(document) + b"\n")
    sys.stdout.buffer.flush()


def _load_json(path: str) -> Any:
    try:
        return read_json(path)
    except Exception as error:  # noqa: BLE001
        raise DomainError(
            "E_INPUT_JSON",
            "input JSON could not be read",
            pointer=str(path),
            details={"error": str(error)},
        ) from error


def _scene_summary(scene) -> dict[str, Any]:
    return {
        "ok": True,
        "scene_sha256": sha256_obj(scene.to_dict()),
        "material_count": len(scene.materials),
        "light_count": len(scene.lights),
        "material_ids": sorted(scene.materials),
        "light_ids": sorted(scene.lights),
        "observer": scene.config.observer,
        "camera_white_point_xyz": list(scene.config.camera_white_point_xyz),
        "adaptation_method": scene.config.adaptation_method,
        "target_color_space": scene.config.target_color_space,
        "sampling": {
            "start_nm": 380,
            "end_nm": 780,
            "step_nm": 5,
            "count": 81,
        },
    }


def cmd_generate(args: argparse.Namespace) -> int:
    sample = write_sample_scene(args.out, seed=args.seed, replace=not args.no_replace)
    summary = {
        "ok": True,
        "output": args.out,
        "seed": args.seed,
        "scene_sha256": sha256_obj(sample),
        "material_count": len(sample["materials"]),
        "light_count": len(sample["lights"]),
    }
    if args.operations:
        operations = demo_operations(sample, seed=args.seed)
        atomic_write_json(args.operations, operations, pretty=True)
        summary["operations_output"] = args.operations
        summary["operations_sha256"] = sha256_obj(operations)
    _emit(summary)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    payload = _load_json(args.input)
    scene = parse_scene(payload, source=args.input)
    _emit(_scene_summary(scene))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    report = run_demo(
        args.out,
        args.report,
        seed=args.seed,
        force=args.force,
    )
    _emit(
        {
            "ok": True,
            "report": args.report,
            "seed": args.seed,
            "revision": report["workspace"]["revision"],
            "snapshot_version": report["workspace"]["snapshot_version"],
            "stage_count": len(report["stages"]),
            "material_count": len(report["colors"]),
            "forced_recovery_consistent": report["recovery_comparison"]["status"]
            == "ok",
            "incremental_vs_full_verification": report["verification"]["status"],
            "injection_error_codes": report["acceptance"]["injection_error_codes"],
        }
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = workspace_report(args.workspace)
    atomic_write_json(args.report, report, pretty=True)
    _emit(
        {
            "ok": True,
            "report": args.report,
            "revision": report["workspace"]["revision"],
            "snapshot_version": report["workspace"]["snapshot_version"],
            "state_sha256": report["workspace"]["state_sha256"],
            "prior_exit_state": report["recovery"]["prior_exit_state"],
        }
    )
    return 0


def _load_operations(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    document = _load_json(path)
    if not isinstance(document, list):
        raise DomainError(
            "E_OPERATION_INVALID",
            "operations input must be a JSON array",
            pointer=str(path),
        )
    return document


def cmd_apply(args: argparse.Namespace) -> int:
    operations = _load_operations(args.operations)
    engine = open_workspace(args.workspace)
    applied: list[dict[str, Any]] = []
    try:
        if engine.revision == 0:
            if operations and operations[0].get("op") == "initialize":
                pending = operations
            else:
                if args.input:
                    initial_scene = _load_json(args.input)
                else:
                    initial_scene = generate_sample_scene(args.seed)
                pending = [
                    {
                        "version": 1,
                        "op": "initialize",
                        "payload": {"scene": initial_scene},
                    }
                ] + operations
        else:
            pending = [
                operation
                for operation in operations
                if int(operation.get("version", 0)) > engine.revision
            ]
        for operation in pending:
            result = engine.apply(operation)
            applied.append(
                {
                    "revision": result["revision"],
                    "operation": operation["op"],
                    "snapshot_version": result["snapshot_version"],
                    "state_sha256": result["state_sha256"],
                    "recomputed_count": result["delta"]["recomputed_count"],
                    "reused_count": result["delta"]["reused_count"],
                }
            )
        report = build_report(engine)
        atomic_write_json(args.report, report, pretty=True)
        summary = {
            "ok": True,
            "workspace": args.workspace,
            "report": args.report,
            "revision": engine.revision,
            "snapshot_version": engine.snapshot_version,
            "state_sha256": engine.state_hash if engine.revision else None,
            "applied": applied,
        }
    finally:
        engine.close()
    _emit(summary)
    return 0


def cmd_restore_snapshot(args: argparse.Namespace) -> int:
    store = PersistenceStore(args.workspace)
    result = store.import_snapshot_bundle(args.snapshot)
    _emit({"ok": True, **result})
    return 0


def cmd_crash_worker(args: argparse.Namespace) -> int:
    run_crash_worker(args.workspace, args.operations, args.ready_file)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectral-workspace",
        description="Deterministic spectral material and colour management workspace",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic sample data")
    generate.add_argument("--out", default="samples/sample_scene.json")
    generate.add_argument("--operations", default="samples/demo_operations.json")
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument("--no-replace", action="store_true")
    generate.set_defaults(func=cmd_generate)

    validate = subparsers.add_parser("validate", help="validate a scene JSON file")
    validate.add_argument("--input", required=True)
    validate.set_defaults(func=cmd_validate)

    demo = subparsers.add_parser(
        "demo", help="run fixed edits, fault injection and forced-termination recovery"
    )
    demo.add_argument("--out", default="runtime/demo")
    demo.add_argument("--report", default="reports/demo_report.json")
    demo.add_argument("--seed", type=int, default=DEFAULT_SEED)
    demo.add_argument("--force", action="store_true")
    demo.set_defaults(func=cmd_demo)

    report = subparsers.add_parser("report", help="reopen a workspace and write a report")
    report.add_argument("--workspace", required=True)
    report.add_argument("--report", required=True)
    report.set_defaults(func=cmd_report)

    apply = subparsers.add_parser("apply", help="apply versioned operations to a workspace")
    apply.add_argument("--workspace", required=True)
    apply.add_argument("--operations")
    apply.add_argument("--input")
    apply.add_argument("--seed", type=int, default=DEFAULT_SEED)
    apply.add_argument("--report", default="reports/apply_report.json")
    apply.set_defaults(func=cmd_apply)

    restore = subparsers.add_parser(
        "restore-snapshot", help="atomically replace CURRENT from a snapshot bundle"
    )
    restore.add_argument("--workspace", required=True)
    restore.add_argument("--snapshot", required=True)
    restore.set_defaults(func=cmd_restore_snapshot)

    worker = subparsers.add_parser("crash-worker", help=argparse.SUPPRESS)
    worker.add_argument("--workspace", required=True)
    worker.add_argument("--operations", required=True)
    worker.add_argument("--ready-file", required=True)
    worker.set_defaults(func=cmd_crash_worker)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DomainError as error:
        sys.stdout.buffer.write(canonical_bytes(error.to_dict()) + b"\n")
        sys.stdout.buffer.flush()
        return 2
    except FileExistsError as error:
        _emit(
            {
                "ok": False,
                "error": {
                    "code": "E_OUTPUT_EXISTS",
                    "message": str(error),
                },
            }
        )
        return 2
    except Exception as error:  # noqa: BLE001 - CLI must return machine readable failure
        _emit(
            {
                "ok": False,
                "error": {
                    "code": "E_INTERNAL",
                    "message": str(error),
                    "type": type(error).__name__,
                },
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
