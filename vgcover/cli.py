"""Command-line entry points for the local deterministic example."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .engine import IncrementalRenderer, render_scene, verify_ledger
from .errors import CoverageError, ErrorCode
from .export import export_render, verify_export_manifest
from .hashing import file_sha256, pretty_json_text
from .ledger import ledger_hash
from .scene import (
    generate_category_scenes,
    generate_scene,
    load_scene,
    save_scene,
    scene_hash,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json_text(payload), encoding="utf-8", newline="\n")


def _incremental_demo(initial_scene: dict[str, Any]) -> dict[str, Any]:
    engine = IncrementalRenderer(initial_scene)
    source_region = next(region for region in engine.scene["regions"] if region["region_id"] == 10)
    source_cubic = source_region["contours"][0]["segments"][1]
    changed_control_points = copy.deepcopy(source_cubic["control_points"])
    changed_control_points[1][0] = round(changed_control_points[1][0] + 0.75, 6)
    changed_control_points[1][1] = round(changed_control_points[1][1] - 0.5, 6)

    changes: list[dict[str, Any]] = [
        {"region_id": 10, "field": "tolerance", "value": 0.06},
        {"region_id": 20, "field": "fill_rule", "value": "nonzero"},
        {"region_id": 30, "field": "alpha", "value": 0.40},
        {
            "region_id": 10,
            "field": "control_points",
            "contour_index": 0,
            "segment_index": 1,
            "value": changed_control_points,
        },
    ]
    steps = []
    for change_index, change in enumerate(changes):
        plan = engine.plan_changes([change])
        receipt = engine.apply_changes([change], expected_closure=plan.closure)
        verification = verify_ledger(engine.scene, engine.ledger)
        if not (
            receipt.outside_layer_hashes_preserved
            and receipt.outside_ledger_entries_preserved
            and sorted(set(receipt.recomputed)) == sorted(set(receipt.closure))
            and verification["ok"]
        ):
            raise CoverageError(
                ErrorCode.CLOSURE_MISMATCH,
                "incremental demo invariants failed",
                details={"step": change_index, "receipt": receipt.to_dict()},
            )
        steps.append(
            {
                "step": change_index,
                "change": change,
                "plan_closure": list(plan.closure),
                "receipt": receipt.to_dict(),
                "ledger_verification": verification,
                "scene_hash": engine.scene_hash,
                "ledger_hash": ledger_hash(engine.ledger),
                "buffer_hash": engine.composite.buffer_hash,
            }
        )
    return {
        "schema": "vgcover.incremental_demo.v1",
        "steps": steps,
        "final_scene": engine.scene,
        "final_ledger": engine.ledger,
        "final_buffer_hash": engine.composite.buffer_hash,
        "final_scene_hash": engine.scene_hash,
        "final_ledger_hash": ledger_hash(engine.ledger),
    }


def _run_demo(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out).resolve()
    scene = generate_scene(seed=args.seed, width=args.width, height=args.height)
    render = render_scene(scene)
    manifest = export_render(render, out_dir)
    verification = verify_ledger(scene, render.ledger)
    manifest_verification = verify_export_manifest(out_dir, manifest)

    catalogue_manifest = {}
    catalogue = generate_category_scenes(seed=args.seed, width=args.width, height=args.height)
    catalogue_dir = out_dir / "catalogue"
    for category, category_scene in catalogue.items():
        path = catalogue_dir / f"{category}.json"
        digest = save_scene(path, category_scene, pretty=True)
        catalogue_manifest[category] = {
            "file": path.relative_to(out_dir).as_posix(),
            "sha256": digest,
            "file_sha256": file_sha256(path),
        }
    _write_json(catalogue_dir / "manifest.json", catalogue_manifest)

    incremental = _incremental_demo(scene)
    _write_json(out_dir / "incremental_demo.json", incremental)
    incremental_dir = out_dir / "incremental"
    incremental_manifest = export_render(render_scene(incremental["final_scene"]), incremental_dir)
    incremental_manifest_verification = verify_export_manifest(incremental_dir, incremental_manifest)

    result = {
        "ok": True,
        "output_dir": str(out_dir),
        "scene_hash": scene_hash(scene),
        "ledger_hash": ledger_hash(render.ledger),
        "buffer_hash": render.composite.buffer_hash,
        "ledger_complete": render.ledger["complete"],
        "image_total_error_bound": render.ledger["image"]["total_error_bound"],
        "ledger_verification": verification,
        "manifest_verification": manifest_verification,
        "incremental_manifest_verification": incremental_manifest_verification,
        "incremental_steps": len(incremental["steps"]),
        "files": {
            "scene": "scene.json",
            "ledger": "ledger.json",
            "buffer": "composite.bgra",
            "manifest": "manifest.json",
            "summary": "summary.json",
            "incremental_demo": "incremental_demo.json",
        },
    }
    return result


def _run_render(args: argparse.Namespace) -> dict[str, Any]:
    scene = load_scene(args.scene)
    render = render_scene(scene)
    verification = verify_ledger(scene, render.ledger)
    out_dir = Path(args.out).resolve()
    manifest = export_render(render, out_dir)
    manifest_verification = verify_export_manifest(out_dir, manifest)
    return {
        "ok": True,
        "output_dir": str(out_dir),
        "scene_hash": scene_hash(scene),
        "ledger_hash": ledger_hash(render.ledger),
        "buffer_hash": render.composite.buffer_hash,
        "ledger_verification": verification,
        "manifest_verification": manifest_verification,
    }


def _run_verify(args: argparse.Namespace) -> dict[str, Any]:
    scene = load_scene(args.scene)
    with Path(args.ledger).open("r", encoding="utf-8") as stream:
        ledger = json.load(stream)
    result = verify_ledger(scene, ledger)
    if args.out:
        _write_json(Path(args.out), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vgcover")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="generate, render, export and verify the local example")
    demo.add_argument("--out", default="output", help="output directory (default: output)")
    demo.add_argument("--seed", type=int, default=20261119)
    demo.add_argument("--width", type=int, default=192)
    demo.add_argument("--height", type=int, default=128)
    demo.set_defaults(handler=_run_demo)

    render = subparsers.add_parser("render", help="render an existing scene JSON file")
    render.add_argument("--scene", required=True)
    render.add_argument("--out", default="output", help="output directory (default: output)")
    render.set_defaults(handler=_run_render)

    verify = subparsers.add_parser("verify", help="independently recompute and verify a ledger")
    verify.add_argument("--scene", required=True)
    verify.add_argument("--ledger", required=True)
    verify.add_argument("--out", default=None)
    verify.set_defaults(handler=_run_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except CoverageError as error:
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

