"""Command line entry points for rendering and deterministic sample creation."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import render_scene
from .errors import EngineError
from .pointcloud import generate_sample_cloud


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pointcloud-engine",
        description="Local point-cloud offscreen renderer (stdlib + NumPy only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="render a scene JSON with resume support")
    render.add_argument("scene", help="path to the scene description JSON")
    render.add_argument("--no-signals", action="store_true",
                        help="do not install SIGINT/SIGTERM handlers (for tests)")

    gen = sub.add_parser("gen-sample", help="generate a deterministic sample cloud + scene")
    gen.add_argument("workdir", help="directory that will hold cloud, scene, state and outputs")
    gen.add_argument("--width", type=int, default=48)
    gen.add_argument("--height", type=int, default=32)
    gen.add_argument("--supersample", type=int, default=3)
    gen.add_argument("--tile-width", type=int, default=24)
    gen.add_argument("--tile-height", type=int, default=16)
    gen.add_argument("--concurrency", type=int, default=4)
    gen.add_argument("--injections", nargs="*", default=[],
                     help="fault injection types, e.g. zero_radius non_finite_normal")
    return parser


def default_scene_dict(workdir: str, args: argparse.Namespace) -> dict:
    cloud_rel = "cloud.pts3"
    return {
        "point_cloud": cloud_rel,
        "world_scale": 1.0,
        "origin_offset": [0.0, 0.0, 0.0],
        "camera": {
            "position": [0.0, 0.0, 4.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "vfov_deg": 60.0,
            "aspect": args.width / args.height,
        },
        "width": args.width,
        "height": args.height,
        "supersample": args.supersample,
        "point_radius": 0.09,
        "depth_decay": {"kind": "exp", "coefficient": 0.08},
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "concurrency": args.concurrency,
        "memory_budget_bytes": _memory_budget(args),
        "state_dir": "state",
        "output_dir": "output",
        "near": 1e-6,
        "tile_delay_seconds": 0.0,
        "fault_injections": [{"type": name} for name in args.injections],
    }


def _memory_budget(args: argparse.Namespace) -> int:
    per_tile = args.tile_width * args.tile_height * 8
    return per_tile * max(1, args.concurrency)


def cmd_gen_sample(args: argparse.Namespace) -> int:
    os.makedirs(args.workdir, exist_ok=True)
    cloud_path = os.path.join(args.workdir, "cloud.pts3")
    cloud = generate_sample_cloud(cloud_path)
    scene = default_scene_dict(args.workdir, args)
    scene_path = os.path.join(args.workdir, "scene.json")
    with open(scene_path, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, ensure_ascii=False, indent=2, sort_keys=True)
    payload = {
        "ok": True,
        "scene": os.path.abspath(scene_path),
        "cloud": os.path.abspath(cloud_path),
        "point_count": cloud.count,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    try:
        result = render_scene(args.scene, install_signals=not args.no_signals)
    except EngineError as exc:
        print(exc.to_json(), file=sys.stderr)
        return 2
    report = result["report"]
    report_path = os.path.join(os.path.dirname(os.path.abspath(args.scene)), "last_report.json")
    if result["status"] != "interrupted":
        from .io_utils import atomic_write_json

        atomic_write_json(report_path, report)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return int(result["exit_code"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gen-sample":
        return cmd_gen_sample(args)
    if args.command == "render":
        return cmd_render(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
