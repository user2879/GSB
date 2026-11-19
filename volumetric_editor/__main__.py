"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from .demo import run_demo, run_generate, run_render
from .errors import BusinessError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m volumetric_editor",
        description="Deterministic transparent-media and volumetric-light editor",
    )
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="run the complete repeatable demonstration")
    demo.add_argument("--root", default=".", help="workspace root (ASCII path recommended)")
    demo.add_argument("--seed", type=int, default=20261119)
    demo.add_argument("--width", type=int, default=96)
    demo.add_argument("--height", type=int, default=64)
    demo.add_argument("--backend", choices=("auto", "opengl", "software"), default="auto")

    generate = subparsers.add_parser("generate", help="generate the deterministic scene sample")
    generate.add_argument("--root", default=".")
    generate.add_argument("--seed", type=int, default=20261119)
    generate.add_argument("--width", type=int, default=96)
    generate.add_argument("--height", type=int, default=64)

    render = subparsers.add_parser("render", help="render one deterministic frame")
    render.add_argument("--root", default=".")
    render.add_argument("--seed", type=int, default=20261119)
    render.add_argument("--width", type=int, default=96)
    render.add_argument("--height", type=int, default=64)
    render.add_argument("--threads", type=int, default=4)
    render.add_argument("--tile-size", type=int, default=16)
    render.add_argument("--backend", choices=("auto", "opengl", "software"), default="auto")
    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command or "demo"
    if args.command is None:
        args = parser.parse_args([*sys.argv[1:], "demo"]) if argv is None else parser.parse_args(["demo"])
        command = "demo"
    try:
        if command == "demo":
            result = run_demo(
                root=args.root,
                seed=args.seed,
                width=args.width,
                height=args.height,
                backend=args.backend,
            )
            _print_json(result)
            return 0 if result["checks"]["all_pass"] else 1
        if command == "generate":
            _print_json(
                run_generate(
                    root=args.root,
                    seed=args.seed,
                    width=args.width,
                    height=args.height,
                )
            )
            return 0
        if command == "render":
            result = run_render(
                root=args.root,
                seed=args.seed,
                width=args.width,
                height=args.height,
                threads=args.threads,
                tile_size=args.tile_size,
                backend=args.backend,
            )
            _print_json(result)
            return 0
        parser.error(f"unknown command: {command}")
    except BusinessError as exc:
        _print_json({"status": "error", **exc.to_dict()})
        return 2
    except (ValueError, OSError) as exc:
        _print_json({"status": "error", "code": "E_INTERNAL", "message": str(exc)})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
