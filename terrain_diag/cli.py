from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from .anomalies import run_error_code_injections
from .canonical import canonical_dumps, sha256_file, write_json
from .diagnostics import find_first_divergence
from .errors import ErrorCode, WorkbenchError
from .heightfield import DEFAULT_SEED
from .quadtree import total_leaf_bytes
from .runner import RunSettings, default_budgets, prepare_assets, run_workbench
from .workspace import MINIMUM_FREE_BYTES, Workspace


def _common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, choices=(257, 1025), default=1025)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=DEFAULT_SEED)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--track", default="canonical", choices=("canonical", "yaw_offset", "height_offset"))
    parser.add_argument("--projection-budget-px", type=float, default=1.5)
    parser.add_argument("--depth-error-budget", type=float, default=0.02)
    parser.add_argument("--depth-bias", type=float, default=1.0e-5)
    parser.add_argument("--base-world-expand", type=float, default=0.05)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--memory-budget-bytes", type=int)
    parser.add_argument("--vram-budget-bytes", type=int)
    parser.add_argument("--allow-low-disk", action="store_true")


def _settings_from_args(args: argparse.Namespace) -> RunSettings:
    return RunSettings(
        size=args.size,
        seed=args.seed,
        frame_count=args.frames,
        track_variant=args.track,
        projection_error_budget_px=args.projection_budget_px,
        depth_error_budget=args.depth_error_budget,
        depth_bias=args.depth_bias,
        base_world_expand=args.base_world_expand,
        budget_scale=args.budget_scale,
        memory_budget_bytes=args.memory_budget_bytes,
        vram_budget_bytes=args.vram_budget_bytes,
        minimum_free_bytes=MINIMUM_FREE_BYTES,
        enforce_disk=not args.allow_low_disk,
    )


def _file_map(root: Path) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        result[relative] = (path.stat().st_size, sha256_file(path))
    return result


def _compare_run_directories(left: Path, right: Path) -> dict[str, Any]:
    left_files = _file_map(left)
    right_files = _file_map(right)
    only_left = sorted(set(left_files) - set(right_files))
    only_right = sorted(set(right_files) - set(left_files))
    differing = sorted(
        name for name in set(left_files) & set(right_files) if left_files[name] != right_files[name]
    )
    identical = not only_left and not only_right and not differing
    if not identical:
        raise WorkbenchError(
            ErrorCode.E_TRACK_DIVERGE,
            "two deterministic demo runs are not byte-identical",
            {
                "left": str(left),
                "right": str(right),
                "only_left": only_left,
                "only_right": only_right,
                "differing": differing,
            },
        )
    return {
        "identical": True,
        "file_count": len(left_files),
        "total_bytes": sum(item[0] for item in left_files.values()),
        "tree_sha256": sha256_file(left / "meta" / "run_manifest.json"),
    }


def command_generate(args: argparse.Namespace) -> int:
    workspace = Workspace.create(
        args.output,
        minimum_free_bytes=MINIMUM_FREE_BYTES,
        enforce_disk=not args.allow_low_disk,
    )
    results: list[dict[str, Any]] = []
    for size in args.sizes:
        settings = RunSettings(
            size=size,
            seed=args.seed,
            frame_count=1,
            minimum_free_bytes=MINIMUM_FREE_BYTES,
            enforce_disk=not args.allow_low_disk,
        )
        assets = prepare_assets(workspace, settings)
        memory_bytes, vram_bytes = total_leaf_bytes(assets.quadtree)
        budgets = default_budgets(assets.quadtree)
        results.append(
            {
                "size": size,
                "terrain_sha256": assets.terrain_sha256,
                "quadtree_sha256": assets.quadtree.content_sha256,
                "leaf_blocks": len(assets.quadtree.leaves),
                "leaf_memory_bytes": memory_bytes,
                "leaf_vram_bytes": vram_bytes,
                "default_streaming_budgets": budgets.to_dict(),
            }
        )
    print(canonical_dumps({"workspace_layout": str(workspace.root), "generated": results}))
    return 0


def command_run(args: argparse.Namespace) -> int:
    result = run_workbench(args.output, _settings_from_args(args))
    print(
        canonical_dumps(
            {
                "workspace": str(result.workspace.root),
                "report": str(result.report_path),
                "report_sha256": result.report_sha256,
                "run_manifest": str(result.run_manifest_path),
                "run_manifest_sha256": result.run_manifest_sha256,
                "summary": result.summary,
            }
        )
    )
    return 0


def command_compare(args: argparse.Namespace) -> int:
    divergence = find_first_divergence(args.left, args.right, raise_error=False)
    if args.output:
        write_json(args.output, divergence)
    print(canonical_dumps(divergence))
    if divergence.get("diverged"):
        return 2
    return 0


def command_anomalies(args: argparse.Namespace) -> int:
    report = run_error_code_injections(args.workspace, args.output, seed=args.seed)
    print(canonical_dumps(report))
    return 0 if report["all_passed"] else 1


def _demo_check_degradations(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
) -> dict[str, Any]:
    actions = ("LOWER_HIZ", "EXPAND_BBOX", "REDUCE_RESIDENT")
    observed = {
        action: {
            "run_a": sum(run_a[size].summary["degradation_counts"][action] for size in run_a),
            "run_b": sum(run_b[size].summary["degradation_counts"][action] for size in run_b),
        }
        for action in actions
    }
    exercised = {
        action: observed[action]["run_a"] > 0 and observed[action]["run_b"] > 0
        for action in actions
    }
    if not all(exercised.values()):
        raise WorkbenchError(
            ErrorCode.E_BUDGET_DENIED,
            "demo did not exercise every fixed-priority budget degradation path",
            {"observed": observed, "exercised": exercised},
        )
    return {"observed_counts": observed, "all_paths_exercised": True}


def command_demo(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.force:
        if any(output.iterdir()):
            raise WorkbenchError(
                ErrorCode.E_BUDGET_DENIED,
                "demo output directory is not empty; pass --force to replace run directories",
                {"output": str(output)},
            )
    output.mkdir(parents=True, exist_ok=True)
    for name in ("run_a", "run_b", "anomalies"):
        target = output / name
        if target.exists():
            shutil.rmtree(target)

    sizes = tuple(args.sizes)
    run_a: dict[int, Any] = {}
    run_b: dict[int, Any] = {}
    for size in sizes:
        settings = RunSettings(
            size=size,
            seed=args.seed,
            frame_count=args.frames,
            projection_error_budget_px=args.projection_budget_px,
            depth_error_budget=args.depth_error_budget,
            depth_bias=args.depth_bias,
            base_world_expand=args.base_world_expand,
            budget_scale=args.budget_scale,
            minimum_free_bytes=MINIMUM_FREE_BYTES,
            enforce_disk=not args.allow_low_disk,
        )
        run_a[size] = run_workbench(output / "run_a", settings)
        run_b[size] = run_workbench(output / "run_b", settings)

    directory_comparison = _compare_run_directories(output / "run_a", output / "run_b")
    report_comparisons: dict[str, Any] = {}
    for size in sizes:
        comparison = find_first_divergence(
            run_a[size].report_path,
            run_b[size].report_path,
            raise_error=False,
        )
        report_comparisons[str(size)] = comparison
        if comparison.get("diverged"):
            raise WorkbenchError(
                ErrorCode.E_TRACK_DIVERGE,
                "same-input report comparison found a divergence",
                {"size": size, "comparison": comparison},
            )

    anomaly_report = run_error_code_injections(
        output / "run_a",
        output / "anomalies" / "anomaly_report.json",
    )
    if not anomaly_report["all_passed"]:
        raise WorkbenchError(
            ErrorCode.E_TRACK_DIVERGE,
            "one or more business error-code injections failed",
            {"anomaly_report": anomaly_report},
        )
    degradation_check = _demo_check_degradations(run_a, run_b)
    repro = {
        "schema_version": 1,
        "command": "python -m terrain_diag demo",
        "seed": int(args.seed),
        "sizes": list(sizes),
        "frames_per_size": args.frames,
        "directory_comparison": directory_comparison,
        "report_comparisons": report_comparisons,
        "degradation_check": degradation_check,
        "anomaly_checks": anomaly_report,
        "run_manifest_sha256": directory_comparison["tree_sha256"],
        "runs": {
            "run_a": {
                str(size): {
                    "report_sha256": run_a[size].report_sha256,
                    "summary": run_a[size].summary,
                }
                for size in sizes
            },
            "run_b": {
                str(size): {
                    "report_sha256": run_b[size].report_sha256,
                    "summary": run_b[size].summary,
                }
                for size in sizes
            },
        },
        "reproducible": True,
    }
    repro_path = output / "repro_summary.json"
    repro_sha = write_json(repro_path, repro)
    print(
        canonical_dumps(
            {
                "output": str(output),
                "reproducible": True,
                "repro_summary": str(repro_path),
                "repro_summary_sha256": repro_sha,
                "run_tree_sha256": directory_comparison["tree_sha256"],
                "run_manifest_sha256": directory_comparison["tree_sha256"],
                "runs": {
                    name: {
                        str(size): {
                            "report_sha256": value[size].report_sha256,
                        }
                        for size in sizes
                    }
                    for name, value in (("run_a", run_a), ("run_b", run_b))
                },
                "degradation_check": degradation_check,
                "anomaly_all_passed": anomaly_report["all_passed"],
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terrain-diag",
        description="Deterministic terrain Hi-Z occlusion and streaming diagnostics workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the complete two-run reproducibility demonstration")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--sizes", nargs="+", type=int, choices=(257, 1025), default=[257, 1025])
    demo.add_argument("--seed", type=lambda value: int(value, 0), default=DEFAULT_SEED)
    demo.add_argument("--frames", type=int, default=24)
    demo.add_argument("--projection-budget-px", type=float, default=1.5)
    demo.add_argument("--depth-error-budget", type=float, default=0.02)
    demo.add_argument("--depth-bias", type=float, default=1.0e-5)
    demo.add_argument("--base-world-expand", type=float, default=0.05)
    demo.add_argument("--budget-scale", type=float, default=1.0)
    demo.add_argument("--force", action="store_true")
    demo.add_argument("--allow-low-disk", action="store_true")
    demo.set_defaults(handler=command_demo)

    generate = subparsers.add_parser("generate", help="generate deterministic terrain and quadtree manifests")
    generate.add_argument("--output", default="artifacts/generated")
    generate.add_argument("--sizes", nargs="+", type=int, choices=(257, 1025), default=[257, 1025])
    generate.add_argument("--seed", type=lambda value: int(value, 0), default=DEFAULT_SEED)
    generate.add_argument("--allow-low-disk", action="store_true")
    generate.set_defaults(handler=command_generate)

    run = subparsers.add_parser("run", help="run one deterministic trajectory in one workspace")
    run.add_argument("--output", default="artifacts/run")
    _common_run_arguments(run)
    run.set_defaults(handler=command_run)

    compare = subparsers.add_parser("compare", help="locate the first per-block culling divergence")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--output")
    compare.set_defaults(handler=command_compare)

    anomalies = subparsers.add_parser("anomalies", help="verify all business error codes")
    anomalies.add_argument("--workspace", required=True)
    anomalies.add_argument("--output")
    anomalies.add_argument("--seed", type=lambda value: int(value, 0))
    anomalies.set_defaults(handler=command_anomalies)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except WorkbenchError as exc:
        print(canonical_dumps(exc.to_dict()), file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(
            json.dumps(
                {"code": "E_TRACK_DIVERGE", "message": str(exc), "context": {"type": type(exc).__name__}},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
