from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import RunConfig, Workbench
from .errors import WorkbenchError
from .loader import load_plan
from .util import atomic_write_json, ensure_clean_directory, read_json, stable_hash


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", required=True, help="deterministic sample directory")
    parser.add_argument("--output", required=True, help="workspace output directory")
    parser.add_argument("--config", help="optional config override file")
    parser.add_argument("--serial", action="store_true", help="disable the thread pool")
    parser.add_argument(
        "--budget",
        type=int,
        help="override recompute budget for this run (0 is allowed)",
    )


def _run_one(args: argparse.Namespace, *, step: int, mode: str) -> dict[str, Any]:
    if args.budget is not None and args.budget < 0:
        raise ValueError("--budget must be non-negative")
    run_config = RunConfig(
        samples_dir=Path(args.samples),
        output_dir=Path(args.output),
        step=step,
        mode=mode,
        serial=args.serial,
        budget_override=args.budget,
        config_override=Path(args.config) if args.config else None,
        simulate_stale=getattr(args, "inject_stale", False),
    )
    return Workbench(run_config).run()


def command_run(args: argparse.Namespace) -> int:
    report = _run_one(args, step=args.step, mode="incremental")
    _print_summary(report, Path(args.output))
    return 0


def command_roll(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    samples = Path(args.samples).resolve()
    plan = load_plan(samples)
    all_steps = [step["step"] for step in plan["steps"]]
    if all_steps and 0 not in all_steps:
        all_steps = [0] + all_steps
    if args.through_step is not None:
        all_steps = [step for step in all_steps if step <= args.through_step]
    if not all_steps:
        all_steps = [0]
    manifest_path = output / "live" / "manifest.json"
    applied: list[int] = []
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        applied = list(manifest.get("applied_steps", []))
    pending_steps = [step for step in all_steps if step not in applied]
    if not pending_steps and applied:
        report = read_json(output / "report.json")
        _print_summary(report, output)
        return 0
    report = None
    for step in pending_steps:
        report = _run_one(args, step=step, mode="incremental")
    if report is None:
        report = read_json(output / "report.json")
    _print_summary(report, output)
    return 0


def command_rebuild(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    workspace = Path.cwd().resolve()
    if not args.keep_output:
        ensure_clean_directory(output, workspace)
    report = _run_one(args, step=args.step, mode="full")
    comparison = {
        "mode": "full",
        "canonical_hash": report["canonical_hash"],
        "report": str(output / "report.json"),
    }
    atomic_write_json(output / "rebuild_summary.json", comparison)
    _print_summary(report, output)
    return 0


def _first_differences(left: Any, right: Any, path: str = "$", limit: int = 40) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    def walk(a: Any, b: Any, location: str) -> None:
        if len(differences) >= limit:
            return
        if type(a) is not type(b):
            differences.append(
                {"path": location, "left_type": type(a).__name__, "right_type": type(b).__name__}
            )
            return
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a:
                    differences.append({"path": f"{location}.{key}", "reason": "missing_left"})
                elif key not in b:
                    differences.append({"path": f"{location}.{key}", "reason": "missing_right"})
                else:
                    walk(a[key], b[key], f"{location}.{key}")
                if len(differences) >= limit:
                    return
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(
                    {
                        "path": location,
                        "reason": "length",
                        "left_length": len(a),
                        "right_length": len(b),
                    }
                )
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                walk(left_item, right_item, f"{location}[{index}]")
                if len(differences) >= limit:
                    return
        elif a != b:
            differences.append({"path": location, "left": a, "right": b})

    walk(left, right, path)
    return differences


def command_compare(args: argparse.Namespace) -> int:
    left = read_json(args.left)
    right = read_json(args.right)
    left_hash = stable_hash(left)
    right_hash = stable_hash(right)
    differences = _first_differences(left, right)
    result = {
        "equal": left == right,
        "left_hash": left_hash,
        "right_hash": right_hash,
        "left": str(Path(args.left).resolve()),
        "right": str(Path(args.right).resolve()),
        "differences": differences,
        "difference_count": len(differences),
    }
    if args.json_out:
        atomic_write_json(args.json_out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["equal"] else 1


def _print_summary(report: dict[str, Any], output: Path) -> None:
    summary = report["summary"]
    print(
        json.dumps(
            {
                "mode": report["run"]["mode"],
                "global_version": report["run"]["global_version"],
                "requests": summary["request_count"],
                "unique_candidates": summary["unique_candidate_count"],
                "pending": summary["pending_count"],
                "failures": summary["failure_count"],
                "cache_invalidated": summary["invalidated_cache_entries"],
                "cache_reused": summary["reused_cache_entries"],
                "canonical_hash": report.get("canonical_hash"),
                "report": str(output / "report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recall-workbench",
        description="Deterministic recall shard merge, fixed-point fusion and diversity rerank workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one incremental plan step")
    _add_common(run_parser)
    run_parser.add_argument("--step", type=int, required=True, help="materialize plan through this step")
    run_parser.add_argument("--inject-stale", action="store_true")
    run_parser.set_defaults(func=command_run)

    roll_parser = subparsers.add_parser("roll", help="run a rolling sequence incrementally")
    _add_common(roll_parser)
    roll_parser.add_argument("--through-step", type=int)
    roll_parser.add_argument("--inject-stale", action="store_true")
    roll_parser.set_defaults(func=command_roll)

    rebuild_parser = subparsers.add_parser("rebuild", help="full rebuild from a blank output directory")
    _add_common(rebuild_parser)
    rebuild_parser.add_argument("--step", type=int, required=True)
    rebuild_parser.add_argument(
        "--keep-output",
        action="store_true",
        help="do not clear the output directory first",
    )
    rebuild_parser.add_argument("--inject-stale", action="store_true")
    rebuild_parser.set_defaults(func=command_rebuild)

    compare_parser = subparsers.add_parser(
        "compare", help="field-by-field compare two canonical reports"
    )
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--json-out")
    compare_parser.set_defaults(func=command_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        output = Path(getattr(args, "output", ".")).resolve()
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / "error_report.json", {"ok": False, **exc.as_dict()})
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 3 if exc.code == "E_SNAPSHOT_STALE" else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        output = Path(getattr(args, "output", ".")).resolve()
        output.mkdir(parents=True, exist_ok=True)
        error = {"ok": False, "code": "E_RUNTIME", "message": str(exc)}
        atomic_write_json(output / "error_report.json", error)
        print(json.dumps(error, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
