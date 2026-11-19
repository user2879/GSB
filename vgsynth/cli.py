from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .canonical import write_json
from .pipeline import compare_runs, run_demo
from .scene import generate_scene, validate_scene
from .verifier import verify_case_dir


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def command_demo(args: argparse.Namespace) -> int:
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if args.single:
        run_dir = output
        report = run_demo(run_dir, minimize=not args.no_minimize)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0 if report["success"] else 1
    first = output / "run_a"
    second = output / "run_b"
    _remove_tree(first)
    _remove_tree(second)
    first_report = run_demo(first, minimize=not args.no_minimize)
    second_report = run_demo(second, minimize=not args.no_minimize)
    comparison = compare_runs(first, second)
    comparison["run_id"] = first_report["run_id"]
    comparison["expected_run_id"] = second_report["run_id"]
    comparison["summary"] = first_report["summary"]
    write_json(output / "comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0 if first_report["success"] and second_report["success"] and comparison["equivalent"] else 1


def command_verify(args: argparse.Namespace) -> int:
    path = Path(args.case_dir or args.run_dir)
    if args.case_dir:
        result = verify_case_dir(path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "ok" else 2
    case_dirs = sorted(path.glob("cases/*"))
    results = []
    failed = []
    expected = []
    for case_dir in case_dirs:
        result = verify_case_dir(case_dir)
        record = {
            "case_id": case_dir.name,
            "status": result.get("status"),
            "code": result.get("code"),
            "first_mismatch": result.get("first_mismatch"),
        }
        results.append(record)
        if result.get("status") != "ok":
            layers_path = case_dir / "layers.json"
            is_expected = False
            if layers_path.exists():
                layers = json.loads(layers_path.read_text(encoding="utf-8"))
                is_expected = bool(layers.get("output_mutations"))
            if is_expected:
                expected.append(record)
            else:
                failed.append(record)
    summary = {
        "schema": "vgsynth.verify-summary/v1",
        "case_count": len(results),
        "failed": failed,
        "expected_mismatch": expected,
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failed else 2


def command_generate(args: argparse.Namespace) -> int:
    scene = generate_scene(args.kind, int(args.seed, 0), args.scene_id)
    issues = validate_scene(scene)
    write_json(Path(args.out), scene)
    print(
        json.dumps(
            {
                "scene_id": scene["scene_id"],
                "content_hash": scene["content_hash"],
                "issue_count": len(issues),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_compare(args: argparse.Namespace) -> int:
    result = compare_runs(Path(args.first), Path(args.second))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["equivalent"] else 2


def command_view(args: argparse.Namespace) -> int:
    from .viewer import show_run

    return show_run(Path(args.run_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vgsynth", description="Deterministic vector graphics synthesis demo")
    parser.add_argument("--version", action="version", version="vgsynth 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the fixed deterministic demonstration sequence")
    demo.add_argument("--out", default="artifacts/demo", help="output directory")
    demo.add_argument("--single", action="store_true", help="write one run directly to --out")
    demo.add_argument("--no-minimize", action="store_true", help="skip minimal counterexample generation")
    demo.set_defaults(func=command_demo)

    verify = subparsers.add_parser("verify", help="verify a run directory or one case directory")
    verify_group = verify.add_mutually_exclusive_group(required=True)
    verify_group.add_argument("--run-dir")
    verify_group.add_argument("--case-dir")
    verify.set_defaults(func=command_verify)

    generate = subparsers.add_parser("generate", help="generate one deterministic scene JSON file")
    generate.add_argument("--kind", required=True, choices=["simple_cross", "self_intersect_loop", "overlapping_large", "degenerate_segments"])
    generate.add_argument("--seed", required=True, help="integer seed, decimal or 0x-prefixed")
    generate.add_argument("--scene-id")
    generate.add_argument("--out", required=True)
    generate.set_defaults(func=command_generate)

    compare = subparsers.add_parser("compare", help="compare two complete run directories by file hash")
    compare.add_argument("--first", required=True)
    compare.add_argument("--second", required=True)
    compare.set_defaults(func=command_compare)

    view = subparsers.add_parser("view", help="open a PySide6 run viewer")
    view.add_argument("--run-dir", required=True)
    view.set_defaults(func=command_view)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
