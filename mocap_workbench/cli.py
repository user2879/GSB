"""Command-line entry point with stable JSON output and business exit codes."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .demo import demo_stdout, run_demo
from .errors import BusinessError, ErrorCode, EXIT_CODES
from .generator import GenerationConfig, generate_scene, load_scene, save_scene
from .pipeline import run_all_strategies, save_solution, solve_scene
from .serialization import canonical_json_bytes, write_json
from .strategies import BudgetConfig


def _budget(args: argparse.Namespace) -> BudgetConfig:
    return BudgetConfig(
        memory_mb=float(args.memory_mb),
        time_budget_ms=float(args.time_ms),
        window_size=int(args.window_size),
        window_overlap=int(args.window_overlap),
        sparse_anchor_step=int(args.anchor_step),
        smooth_lambda=float(args.smooth_lambda),
    )


def _add_budget_options(parser: argparse.ArgumentParser) -> None:
    defaults = BudgetConfig()
    parser.add_argument("--memory-mb", type=float, default=defaults.memory_mb)
    parser.add_argument("--time-ms", type=float, default=defaults.time_budget_ms)
    parser.add_argument("--window-size", type=int, default=defaults.window_size)
    parser.add_argument("--window-overlap", type=int, default=defaults.window_overlap)
    parser.add_argument("--anchor-step", type=int, default=defaults.sparse_anchor_step)
    parser.add_argument("--smooth-lambda", type=float, default=defaults.smooth_lambda)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mocap-workbench", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="generate, solve and cross-check all three data scales")
    demo.add_argument("--out", default="workdir")
    demo.add_argument("--sizes", nargs="+", type=int, default=[300, 3000, 30000])
    demo.add_argument("--seed", type=int, default=20261119)
    demo.add_argument("--skip-determinism-check", action="store_true")
    _add_budget_options(demo)

    generate = subparsers.add_parser("generate", help="produce a deterministic scene")
    generate.add_argument("--frames", type=int, required=True)
    generate.add_argument("--seed", type=int, default=20261119)
    generate.add_argument("--out", required=True)

    solve = subparsers.add_parser("solve", help="solve a saved scene")
    solve.add_argument("--scene", required=True)
    solve.add_argument("--out", required=True)
    solve.add_argument("--strategy", choices=["auto", "full_segment", "windowed", "sparse_anchor"], default="auto")
    solve.add_argument("--strict-retarget", action="store_true")
    _add_budget_options(solve)

    consistency = subparsers.add_parser("consistency", help="compare every budget-feasible strategy")
    consistency.add_argument("--scene", required=True)
    consistency.add_argument("--out", required=True)
    consistency.add_argument("--position-tolerance-m", type=float, default=0.05)
    consistency.add_argument("--residual-tolerance", type=float, default=0.02)
    _add_budget_options(consistency)

    errors = subparsers.add_parser("errors", help="exercise all declared business error paths")
    errors.add_argument("--seed", type=int, default=20261119)
    errors.add_argument("--out", default="workdir/error_checks.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            report = run_demo(
                args.out,
                sizes=tuple(args.sizes),
                seed=args.seed,
                budget=_budget(args),
                determinism_check=not args.skip_determinism_check,
            )
            demo_stdout(report)
            return 0
        if args.command == "generate":
            scene = generate_scene(GenerationConfig(frames=args.frames, seed=args.seed))
            manifest = save_scene(scene, args.out)
            payload = {
                "ok": True,
                "out": args.out,
                "frames": args.frames,
                "seed": args.seed,
                "scene_digest": manifest["scene_digest"],
            }
            sys.stdout.buffer.write(canonical_json_bytes(payload))
            return 0
        if args.command == "solve":
            scene = load_scene(args.scene)
            result = solve_scene(scene, budget=_budget(args), strategy=args.strategy, strict_retarget=args.strict_retarget)
            report = save_solution(result, args.out)
            summary = {
                "ok": True,
                "out": args.out,
                "strategy": result.decision.to_report(),
                "pose_sequence_digest": result.pose_sequence_digest,
                "constraints": result.report()["constraints"],
                "frame_hashes_path": f"{args.out}/frame_hashes.json",
            }
            sys.stdout.buffer.write(canonical_json_bytes(summary))
            return 0
        if args.command == "consistency":
            scene = load_scene(args.scene)
            _, comparison = run_all_strategies(
                scene,
                budget=_budget(args),
                position_tolerance_m=args.position_tolerance_m,
                residual_tolerance=args.residual_tolerance,
            )
            write_json(args.out, comparison)
            sys.stdout.buffer.write(canonical_json_bytes(comparison))
            return 0
        if args.command == "errors":
            from .demo import _error_checks

            checks = _error_checks(args.seed)
            payload = {"ok": True, "error_code_checks": checks}
            write_json(args.out, payload)
            sys.stdout.buffer.write(canonical_json_bytes(payload))
            return 0
        parser.error("unknown command")
    except BusinessError as exc:
        sys.stdout.buffer.write(canonical_json_bytes(exc.as_dict()))
        sys.stdout.buffer.flush()
        return EXIT_CODES[exc.code]
    except OSError as exc:
        error = BusinessError(ErrorCode.E_IO, str(exc))
        sys.stdout.buffer.write(canonical_json_bytes(error.as_dict()))
        sys.stdout.buffer.flush()
        return EXIT_CODES[error.code]
    except Exception as exc:  # pragma: no cover - explicit debug path
        if "--debug" in sys.argv:
            traceback.print_exc()
        error = BusinessError(ErrorCode.E_INPUT, f"unexpected failure: {exc}")
        sys.stdout.buffer.write(canonical_json_bytes(error.as_dict()))
        sys.stdout.buffer.flush()
        return EXIT_CODES[error.code]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
