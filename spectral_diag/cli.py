"""Command-line entry points for the spectral diagnostics application."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, canonical_json
from .closure import ConventionSession
from .config import RunConfig
from .errors import EXIT_CODES, DiagnosticError, ErrorCode
from .injection import prepare_scene
from .matrix import solve_adaptation
from .pipeline import execute_pipeline
from .storage import Workspace, atomic_write_bytes


def _config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-json", help="path or inline JSON for a complete RunConfig")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--observer", default="cie1931_2", choices=("cie1931_2", "cie1964_10"))
    parser.add_argument("--normalization", default="y100_reflector", choices=("y100_reflector", "unity"))
    parser.add_argument("--reference-density", type=float, default=0.5, dest="reference_density_nm")
    parser.add_argument("--quadrature", default="trapezoid", choices=("trapezoid",))
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--material-count", type=int, default=8)
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--quant-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(0.0, 150.0))
    parser.add_argument("--cond-threshold", type=float, default=10000.0)
    parser.add_argument("--attribution-density", type=float, default=5.0, dest="attribution_density_nm")
    parser.add_argument("--drop-count", type=int, default=1, dest="attribution_drop_count")
    parser.add_argument("--injection-json", action="append", default=[], help="injection object or JSON file")


def _load_json_argument(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith(("{", "[")):
        return json.loads(stripped)
    path = Path(value)
    if path.exists() and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _config_from_args(args: argparse.Namespace, extra_injections: list[dict[str, Any]] | None = None) -> RunConfig:
    if args.config_json:
        data = _load_json_argument(args.config_json)
        if not isinstance(data, dict):
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="config JSON must be an object",
                details={"type": type(data).__name__},
            )
        config = RunConfig.from_dict(data)
    else:
        injections: list[dict[str, Any]] = []
        for raw in args.injection_json:
            value = _load_json_argument(raw)
            if isinstance(value, list):
                injections.extend(value)
            elif isinstance(value, dict) and "kind" in value:
                injections.append(value)
            elif isinstance(value, dict) and "injections" in value:
                injections.extend(value["injections"])
            else:
                raise DiagnosticError(
                    code=ErrorCode.CONFIG_INVALID,
                    message="injection JSON must be an injection object, list, or object with injections",
                    details={"value": value},
                )
        config = RunConfig(
            seed=args.seed,
            observer=args.observer,
            normalization=args.normalization,
            reference_density_nm=args.reference_density_nm,
            quadrature=args.quadrature,
            sample_count=args.sample_count,
            material_count=args.material_count,
            quant_bits=args.quant_bits,
            quant_range=tuple(args.quant_range),
            cond_threshold=args.cond_threshold,
            attribution_density_nm=args.attribution_density_nm,
            attribution_drop_count=args.attribution_drop_count,
            injections=tuple(injections),
        )
    if extra_injections:
        config = RunConfig.from_dict(
            {
                **config.to_dict(),
                "injections": list(config.injections) + list(extra_injections),
            }
        )
    return config


def _default_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}-{os.getpid()}"


def _print_json(value: Any) -> None:
    print(canonical_json(value))


def _run(args: argparse.Namespace, extra_injections: list[dict[str, Any]] | None = None) -> int:
    config = _config_from_args(args, extra_injections)
    workspace = Workspace(args.workspace).ensure()
    run_id = args.run_id or _default_run_id("run")
    workspace.begin(run_id)
    result = execute_pipeline(config)
    result["run_id"] = run_id
    commit = workspace.stage_and_commit(
        run_id,
        result,
        crash_after_stage=bool(args.crash_after_stage),
        exit_fn=os._exit,
    )
    summary = {
        "run_id": run_id,
        "status": result.get("status"),
        "error_code": None if not result.get("error") else result["error"].get("code"),
        "first_affected": result.get("first_affected"),
        "matrix_condition_number": None
        if not result.get("matrix")
        else result["matrix"].get("condition_number"),
        "matrix_residual_rmse": None
        if not result.get("matrix")
        else result["matrix"].get("residual_rmse"),
        "attribution_reconciled": None
        if not result.get("attribution")
        else result["attribution"].get("additivity", {}).get("reconciled"),
        "result_sha256": result.get("result_sha256"),
        "commit": commit,
    }
    _print_json(summary)
    if result.get("status") == "rejected" and result.get("error"):
        return EXIT_CODES.get(ErrorCode(result["error"]["code"]), 1)
    return 0


def _generate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    scene = prepare_scene(config)
    output = Path(args.workspace) / "generated-scene.json"
    atomic_write_bytes(output, canonical_bytes(scene.payload))
    _print_json(
        {
            "status": "generated",
            "scene_sha256": scene.sha256,
            "output": str(output),
            "reference_truth": scene.payload["reference_truth"],
        }
    )
    return 0


def _solve(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    scene = prepare_scene(config)
    solution = solve_adaptation(
        scene.source_xyz,
        scene.target_xyz,
        cond_threshold=config.cond_threshold,
        allow_prune=True,
    )
    _print_json(
        {
            "status": solution.status,
            "scene_sha256": scene.sha256,
            "solution": solution.to_dict(),
        }
    )
    return 0 if solution.accepted else 10


def _recover(args: argparse.Namespace) -> int:
    report = Workspace(args.workspace).recover()
    _print_json(report)
    return 0


def _closure(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    session = ConventionSession(config)
    first = session.run()
    same = session.run()
    switched = session.run(observer=args.other_observer, normalization=args.other_normalization)
    restored = session.run(observer=config.observer, normalization=config.normalization)
    _print_json(
        {
            "initial_recomputed": first["report"]["recomputed"],
            "same_convention_recomputed": same["report"]["recomputed"],
            "switched_recomputed": switched["report"]["recomputed"],
            "restored_recomputed": restored["report"]["recomputed"],
            "restored_cache_hits": restored["report"]["cache_hits"],
            "initial_hashes": first["report"]["hashes"],
            "restored_hashes": restored["report"]["hashes"],
            "hashes_restored": first["report"]["hashes"] == restored["report"]["hashes"],
            "closure_reconciled": restored["report"]["closure_reconciled"],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectral-diag", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="generate, solve, attribute, and commit one run")
    run_parser.add_argument("--workspace", required=True)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--crash-after-stage", action="store_true")
    _config_arguments(run_parser)
    run_parser.set_defaults(handler=_run)

    generate_parser = subparsers.add_parser("generate", help="generate a deterministic scene only")
    generate_parser.add_argument("--workspace", required=True)
    _config_arguments(generate_parser)
    generate_parser.set_defaults(handler=_generate)

    solve_parser = subparsers.add_parser("solve", help="generate and solve a matrix only")
    solve_parser.add_argument("--workspace", required=True)
    _config_arguments(solve_parser)
    solve_parser.set_defaults(handler=_solve)

    inject_parser = subparsers.add_parser("inject", help="run with a parameter-level anomaly")
    inject_parser.add_argument("--workspace", required=True)
    inject_parser.add_argument("--run-id")
    inject_parser.add_argument("--kind", required=True)
    inject_parser.add_argument("--params", default="{}")
    inject_parser.add_argument("--crash-after-stage", action="store_true")
    _config_arguments(inject_parser)
    inject_parser.set_defaults(handler=lambda args: _run(
        args,
        [{"kind": args.kind, "params": _load_json_argument(args.params)}],
    ))

    recover_parser = subparsers.add_parser("recover", help="recover the latest committed stable state")
    recover_parser.add_argument("--workspace", required=True)
    recover_parser.set_defaults(handler=_recover)

    closure_parser = subparsers.add_parser("closure", help="exercise observer/normalisation closure")
    closure_parser.add_argument("--workspace", default=".")
    closure_parser.add_argument("--other-observer", default="cie1964_10", choices=("cie1931_2", "cie1964_10"))
    closure_parser.add_argument("--other-normalization", default="y100_reflector", choices=("y100_reflector", "unity"))
    _config_arguments(closure_parser)
    closure_parser.set_defaults(handler=_closure)

    demo_parser = subparsers.add_parser("demo", help="run the fixed acceptance sequence")
    demo_parser.add_argument("--workspace", default="demo_workspace")
    demo_parser.add_argument("--reset", action="store_true")
    demo_parser.set_defaults(handler=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import run_fixed_sequence

            report = run_fixed_sequence(Path(args.workspace), reset=bool(args.reset))
            _print_json(
                {
                    "status": report["status"],
                    "all_passed": report["all_passed"],
                    "workspace": report["workspace"],
                    "report_path": report["report_path"],
                }
            )
            return 0 if report["all_passed"] else 1
        return int(args.handler(args))
    except DiagnosticError as error:
        print(json.dumps(error.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        print(json.dumps({"code": "E_INTERRUPTED", "message": "interrupted before normal close"}), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
