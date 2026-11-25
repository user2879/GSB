"""Command-line entry points for deterministic demo and diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .commands import apply_commands, load_commands
from .config import config_hash, default_config, load_config, save_config, validate_config
from .engine import reconstruct
from .errors import E_PARAM_INVALID, E_VIEW_UNKNOWN, E_VOXEL_ALIGN
from .fixed import digest
from .gui import gui_capability, launch_gui
from .report import build_report, console_report, write_report
from .snapshot import load_snapshot, write_snapshot
from .synth import initialize_project


def _clean_demo(project_dir: Path) -> None:
    if project_dir.exists():
        for child in project_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    project_dir.mkdir(parents=True, exist_ok=True)


def init_project(project_dir: str | Path, *, force: bool = False) -> int:
    root = Path(project_dir)
    if (root / "config.json").exists() and not force:
        print(f"project already exists: {root} (use --force to regenerate)", file=sys.stderr)
        return 2
    if force:
        _clean_demo(root)
    initialize_project(root, config=default_config())
    print(f"initialized deterministic sample at {root}")
    return 0


def run_once(
    project_dir: str | Path,
    *,
    commands_path: str | Path | None = None,
    fresh: bool = False,
    persist_snapshot: bool = True,
    write_latest_report: bool = True,
) -> dict[str, Any]:
    root = Path(project_dir)
    config = load_config(root)
    inherited_errors: list[dict[str, Any]] = []
    if commands_path is not None:
        config, command_errors = apply_commands(root, config, load_commands(commands_path))
        inherited_errors.extend(command_errors)
        config = load_config(root)
    snapshot = load_snapshot(root, config_hash(config))
    inherited_errors.extend(snapshot["errors"])
    previous_state = {} if fresh else snapshot["state"]
    previous_manifest = None if fresh else snapshot["manifest"]
    result = reconstruct(root, config, previous_state=previous_state, inherited_errors=inherited_errors)
    if persist_snapshot:
        manifest = write_snapshot(root, result, previous_manifest=previous_manifest)
        result["manifest"] = manifest
    if write_latest_report:
        report = build_report(result, project_dir=root)
        write_report(report, root / "reports" / "latest.json")
    return result


def canonical_result(result: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(result)
    if isinstance(value.get("config", {}).get("scheduler"), dict):
        value["config"]["scheduler"]["max_workers"] = 0
    for key in ("ledger", "scheduler", "depth_reuse", "manifest", "config_hash"):
        value.pop(key, None)
    for depth_map in value.get("depth_maps", {}).values():
        depth_map.pop("run_state", None)
    for voxel in value.get("voxels", []):
        voxel.pop("run_state", None)
    value.pop("validation", None)
    return value


def _full_result(project_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    return reconstruct(project_dir, config, previous_state={}, inherited_errors=[])


def _write_commands(root: Path, name: str, commands: list[dict[str, Any]]) -> Path:
    path = root / "commands" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"commands": commands}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _exception_samples(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="mgirecon-errors-") as temporary:
        temp_root = Path(temporary)
        save_config(temp_root, config)
        _, unknown_errors = apply_commands(
            temp_root,
            config,
            [{"op": "replace_view", "view": "v99", "render_seed": 99}],
        )
        results["E_VIEW_UNKNOWN"] = [error for error in unknown_errors if error["code"] == E_VIEW_UNKNOWN]
        _, invalid_errors = apply_commands(
            temp_root,
            config,
            [{"op": "set_param", "name": "min_disparity", "value": -4}],
        )
        results["E_PARAM_INVALID"] = [error for error in invalid_errors if error["code"] == E_PARAM_INVALID]
    misaligned = copy.deepcopy(config)
    misaligned["reconstruction"]["voxel_size"] = 64
    aligned_validation = validate_config(misaligned, root)
    results["E_VOXEL_ALIGN"] = [
        error for error in aligned_validation["errors"] if error["code"] == E_VOXEL_ALIGN
    ]
    budget_config = copy.deepcopy(config)
    budget_config["scheduler"]["global_point_budget"] = 0
    budget_config["scheduler"]["per_view_point_quota"] = 0
    budget_result = reconstruct(root, budget_config, previous_state={})
    results["E_RECON_BUDGET_EXCEEDED"] = [
        error for error in budget_result["errors"] if error["code"] == "E_RECON_BUDGET_EXCEEDED"
    ]
    return results


def run_demo(project_dir: str | Path, *, workers: int | None = None, budget: int | None = None) -> dict[str, Any]:
    root = Path(project_dir)
    _clean_demo(root)
    config = default_config()
    if workers is not None:
        config["scheduler"]["max_workers"] = workers
    if budget is not None:
        config["scheduler"]["global_point_budget"] = budget
    initialize_project(root, config=config)
    baseline = run_once(root, fresh=True, write_latest_report=False)
    baseline_report = build_report(baseline, project_dir=root)
    write_report(baseline_report, root / "reports" / "baseline.json")
    write_report(baseline_report, Path("reports") / "baseline.json")

    edit_steps = [
        ("01_replace_view.json", [{"op": "replace_view", "view": "v02", "render_seed": 9022}]),
        (
            "02_extrinsics.json",
            [{"op": "set_extrinsics", "view": "v03", "value": [[1, 0, 0, 10], [0, 1, 0, 5], [0, 0, 1, 198]]}],
        ),
        ("03_residual_threshold.json", [{"op": "set_param", "name": "residual_threshold", "value": 14}]),
        ("04_quantization.json", [{"op": "set_param", "name": "quantization_level", "value": 2}]),
        ("05_voxel_size.json", [{"op": "set_param", "name": "voxel_size", "value": 16}]),
        ("06_min_disparity.json", [{"op": "set_param", "name": "min_disparity", "value": 15}]),
        ("07_min_inliers.json", [{"op": "set_param", "name": "min_inliers", "value": 8}]),
        ("08_max_pairs.json", [{"op": "set_param", "name": "max_pairs", "value": 3}]),
    ]
    step_summaries: list[dict[str, Any]] = []
    relations: set[str] = set()
    all_reused = 0
    all_invalidated = 0
    incremental: dict[str, Any] = baseline
    for filename, commands in edit_steps:
        command_path = _write_commands(root, filename, commands)
        incremental = run_once(root, commands_path=command_path, write_latest_report=False)
        report = build_report(incremental, project_dir=root)
        step_relations = set(report["ledger"]["invalidated_by_relation"])
        relations.update(step_relations)
        all_reused += report["ledger"]["reused_total"]
        all_invalidated += report["ledger"]["invalidated_total"]
        step_summaries.append(
            {
                "step": filename,
                "commands": commands,
                "status": incremental["status"],
                "error_codes": report["error_codes"],
                "voxel_count": report["voxel"]["voxel_count"],
                "subdivided_count": report["voxel"]["subdivided_count"],
                "depth_distribution": report["voxel"]["depth_distribution"],
                "reused": report["ledger"]["reused_total"],
                "invalidated": report["ledger"]["invalidated_total"],
                "relations": sorted(step_relations),
                "boundary_pixels": sum(view["boundary_pixels"] for view in report["views"]),
            }
        )

    final_config = load_config(root)
    full = _full_result(root, final_config)
    canonical_incremental = canonical_result(incremental)
    canonical_full = canonical_result(full)
    incremental_digest = digest(canonical_incremental)
    full_digest = digest(canonical_full)
    equality = incremental_digest == full_digest
    if not equality:
        write_report(canonical_incremental, root / "reports" / "canonical_incremental.json")
        write_report(canonical_full, root / "reports" / "canonical_full.json")

    exception_samples = _exception_samples(root, final_config)
    final_report = build_report(incremental, project_dir=root)
    write_report(final_report, root / "reports" / "final.json")
    write_report(final_report, root / "reports" / "latest.json")
    write_report(final_report, Path("reports") / "demo_report.json")
    gui_status = gui_capability()
    if not gui_status.get("ok"):
        fallback = {
            "available": False,
            "error": gui_status.get("error"),
            "headless_report": final_report,
            "launch_command": f"python3 -m mgirecon gui --project {root}",
        }
        write_report(fallback, root / "reports" / "gui_fallback.json")
        write_report(fallback, Path("reports") / "gui_fallback.json")

    summary = {
        "project": str(root),
        "baseline": {
            "voxel_count": baseline_report["voxel"]["voxel_count"],
            "subdivided_count": baseline_report["voxel"]["subdivided_count"],
            "depth_distribution": baseline_report["voxel"]["depth_distribution"],
            "feature_counts": {view["view"]: view["features"] for view in baseline_report["views"]},
            "match_pairs": baseline_report["match_pairs"],
        },
        "steps": step_summaries,
        "final": {
            "voxel_count": final_report["voxel"]["voxel_count"],
            "subdivided_count": final_report["voxel"]["subdivided_count"],
            "depth_distribution": final_report["voxel"]["depth_distribution"],
            "boundary_pixels": sum(view["boundary_pixels"] for view in final_report["views"]),
            "reused_entries": final_report["ledger"]["reused_total"],
            "invalidated_entries": final_report["ledger"]["invalidated_total"],
            "relations_seen_across_steps": sorted(relations),
            "cumulative_reused": all_reused,
            "cumulative_invalidated": all_invalidated,
        },
        "incremental_full_equality": equality,
        "incremental_digest": incremental_digest,
        "full_digest": full_digest,
        "exception_samples": exception_samples,
        "gui": {
            "ok": gui_status.get("ok"),
            "error_code": gui_status.get("error", {}).get("code") if gui_status.get("error") else None,
            "launch_command": f"python3 -m mgirecon gui --project {root}",
        },
    }
    write_report(summary, root / "reports" / "summary.json")
    write_report(summary, Path("reports") / "demo_summary.json")
    text = (
        console_report(final_report)
        + "\n\nincremental == full: "
        + str(equality)
        + f" ({incremental_digest} / {full_digest})\n"
        + "relations seen across edits: "
        + ", ".join(sorted(relations))
        + "\n"
        + "exception samples: "
        + ", ".join(sorted(exception_samples))
        + "\n"
        + "gui: "
        + ("available" if gui_status.get("ok") else str(gui_status.get("error", {}).get("code")))
        + "\n"
    )
    (Path("reports") / "demo_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mgirecon", description="Fixed-point incremental multi-view reconstruction diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create a deterministic local sample project")
    init_parser.add_argument("--project", default="demo")
    init_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="run incremental reconstruction")
    run_parser.add_argument("--project", default="demo")
    run_parser.add_argument("--commands")
    run_parser.add_argument("--fresh", action="store_true")
    run_parser.add_argument("--json", action="store_true")

    edit_parser = subparsers.add_parser("edit", help="apply a command stream without running")
    edit_parser.add_argument("--project", default="demo")
    edit_parser.add_argument("--commands", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate project inputs")
    validate_parser.add_argument("--project", default="demo")

    report_parser = subparsers.add_parser("report", help="run incrementally and print diagnostics")
    report_parser.add_argument("--project", default="demo")

    gui_parser = subparsers.add_parser("gui", help="open the Tkinter diagnostics interface")
    gui_parser.add_argument("--project", default="demo")

    demo_parser = subparsers.add_parser("demo", help="run the complete reproducible acceptance demo")
    demo_parser.add_argument("--project", default="demo")
    demo_parser.add_argument("--workers", type=int)
    demo_parser.add_argument("--budget", type=int)

    args = parser.parse_args(argv)
    if args.command == "init":
        return init_project(args.project, force=args.force)
    if args.command == "run":
        result = run_once(args.project, commands_path=args.commands, fresh=args.fresh)
        report = build_report(result, project_dir=args.project)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(console_report(report))
        return 0 if result["status"] in {"ok", "completed_with_errors"} else 1
    if args.command == "edit":
        config = load_config(args.project)
        updated, errors = apply_commands(args.project, config, load_commands(args.commands))
        validation = validate_config(updated, args.project)
        for error in errors + validation["errors"]:
            print(f"[{error['code']}] {error['message']}", file=sys.stderr)
        return 0 if not errors and not validation["errors"] else 1
    if args.command == "validate":
        config = load_config(args.project)
        validation = validate_config(config, args.project)
        if validation["errors"]:
            for error in validation["errors"]:
                print(f"[{error['code']}] {error['message']}", file=sys.stderr)
            return 1
        print(f"validation ok: {len(validation['valid_views'])} views")
        return 0
    if args.command == "report":
        result = run_once(args.project)
        print(console_report(build_report(result, project_dir=args.project)))
        return 0
    if args.command == "gui":
        result = run_once(args.project)
        gui_result = launch_gui(args.project, result)
        if not gui_result.get("ok"):
            error = gui_result["error"]
            print(f"[{error['code']}] {error['message']}", file=sys.stderr)
            print(console_report(build_report(result, project_dir=args.project)))
            return 3
        return 0
    if args.command == "demo":
        summary = run_demo(args.project, workers=args.workers, budget=args.budget)
        return 0 if summary["incremental_full_equality"] else 1
    return 2
