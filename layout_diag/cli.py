"""Command-line interface for generation, validation, edits, and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .editing import apply_edit_stream, load_edit_commands
from .engine import run_from_paths
from .generator import generate_samples
from .utils import read_json, write_json
from .validation import validate_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="layout-diag", description="Deterministic vertical layout diagnostic tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic synthetic samples")
    generate.add_argument("--out", default="samples", help="sample directory")
    generate.add_argument("--config-out", default="config/default.json", help="default configuration path")
    generate.add_argument("--seed", type=int, default=20261126)

    validate = subparsers.add_parser("validate", help="validate samples, annotations, and configuration")
    validate.add_argument("--dataset", default="samples")
    validate.add_argument("--config", default="config/default.json")
    validate.add_argument("--json-out", default="")

    run = subparsers.add_parser("run", help="run diagnostics and commit an atomic snapshot")
    run.add_argument("--dataset", default="samples")
    run.add_argument("--config", default="config/default.json")
    run.add_argument("--out", default="diagnostics/current-run")
    run.add_argument("--rebuild", action="store_true", help="ignore the previous cache")
    run.add_argument("--serial", action="store_true", help="force one keypoint worker for equivalence checks")

    edit = subparsers.add_parser("edit", help="apply a replayable edit command stream")
    edit.add_argument("--dataset", default="samples")
    edit.add_argument("--config", default="config/default.json")
    edit.add_argument("--commands", required=True)
    edit.add_argument("--base-dir", default="")

    report = subparsers.add_parser("report", help="print a committed diagnostic report")
    report.add_argument("--out", default="diagnostics/current-run")

    compare = subparsers.add_parser("compare", help="compare canonical diagnostic payloads from two runs")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--include-reuse-status", action="store_true")

    subparsers.add_parser("gui", help="open the optional Tk layout inspector")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = generate_samples(args.out, seed=args.seed, config_path=args.config_out)
        print(json.dumps({"ok": True, "images": [item["name"] for item in manifest["images"]], "out": args.out}, indent=2))
        return 0
    if args.command == "validate":
        result = validate_documents(args.dataset, args.config)
        if args.json_out:
            write_json(args.json_out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.command == "edit":
        try:
            commands = load_edit_commands(args.commands)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 2
        base_dir = Path(args.base_dir) if args.base_dir else Path(args.commands).resolve().parent
        failures = apply_edit_stream(args.dataset, args.config, commands, base_dir=base_dir)
        applied_count = len(commands)
        history_path = Path(args.dataset) / "edit_history.json"
        if history_path.is_file():
            try:
                applied_count = int(read_json(history_path).get("applied_count", applied_count))
            except Exception:
                pass
        print(
            json.dumps(
                {
                    "ok": not failures,
                    "submitted_commands": len(commands),
                    "applied_commands": applied_count,
                    "failures": failures.as_dicts(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if not failures else 1
    if args.command == "run":
        result, failures = run_from_paths(
            args.dataset,
            args.config,
            args.out,
            force_rebuild=args.rebuild,
            serial=args.serial,
        )
        summary = {
            "ok": not failures,
            "snapshot": str(result.snapshot_dir),
            "run_mode": result.manifest["run_mode"],
            "tier": result.manifest["tier"],
            "global_version": result.manifest["global_version"],
            "completed_images": result.manifest["completed_images"],
            "pending_images": result.manifest["pending_images"],
            "completed_keypoints": result.manifest["completed_keypoints"],
            "pending_points": result.manifest["pending_points"],
            "conflict_counts": result.documents["report"]["conflict_counts"],
            "business_code_counts": result.documents["report"]["business_code_counts"],
            "cache": result.documents["reuse"]["summary"],
            "failures": failures.as_dicts(),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "report":
        path = Path(args.out) / "current" / "report.json"
        if not path.is_file():
            print(json.dumps({"ok": False, "error": f"report not found: {path}"}, indent=2))
            return 1
        print(json.dumps(read_json(path), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        left = _load_canonical(Path(args.left), include_reuse=args.include_reuse_status)
        right = _load_canonical(Path(args.right), include_reuse=args.include_reuse_status)
        equal = left == right
        print(json.dumps({"equal": equal, "left": args.left, "right": args.right}, indent=2))
        if not equal:
            _print_first_difference(left, right)
        return 0 if equal else 1
    if args.command == "gui":
        try:
            from .gui import launch_gui

            launch_gui()
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"GUI unavailable: {exc}"}, indent=2))
            return 2
        return 0
    return 2


def _load_canonical(path: Path, *, include_reuse: bool) -> dict[str, Any]:
    current = path / "current" if (path / "current").is_dir() else path
    documents = {
        stem: read_json(current / f"{stem}.json")
        for stem in ("schedule", "coarse", "components", "keypoints", "reading_order", "failures", "report")
    }
    if not include_reuse:
        report = documents["report"]
        if isinstance(report, dict):
            report.pop("cache", None)
            report.pop("run_decision", None)
            report.pop("snapshot_recovery", None)
    manifest = read_json(current / "manifest.json")
    canonical = {
        **documents,
        "manifest": {
            key: manifest.get(key)
            for key in (
                "schema_version",
                "global_version",
                "dataset_hash",
                "config_hash",
                "effective_hash",
                "tier",
                "keypoint_version",
                "completed_images",
                "pending_images",
                "failed_images",
                "completed_columns",
                "point_budget",
                "completed_keypoints",
                "pending_points",
            )
        },
    }
    if include_reuse:
        canonical["reuse"] = read_json(current / "reuse.json")
    return canonical


def _print_first_difference(left: Any, right: Any, path: str = "$") -> None:
    if type(left) is not type(right):
        print(f"first difference at {path}: type {type(left).__name__} != {type(right).__name__}")
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                print(f"first difference at {path}.{key}: missing on one side")
                return
            if left[key] != right[key]:
                _print_first_difference(left[key], right[key], f"{path}.{key}")
                return
    elif isinstance(left, list):
        if len(left) != len(right):
            print(f"first difference at {path}: list length {len(left)} != {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            if left_item != right_item:
                _print_first_difference(left_item, right_item, f"{path}[{index}]")
                return
    else:
        print(f"first difference at {path}: {left!r} != {right!r}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
