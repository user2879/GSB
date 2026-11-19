"""Command-line entry points for the NPR editor and determinism demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .demo import run_crash_worker, run_demo, run_verification
from .errors import (
    E_CLOSURE_MISMATCH,
    E_GL_CONTEXT,
    E_RECOVERY,
    E_SNAPSHOT_STALE,
    E_WORKSPACE,
    NPRError,
)
from .generator import DEFAULT_SEED, build_element_hash_manifest, generate_scene
from .gl_backend import probe_offscreen_gl
from .persistence import PersistenceStore
from .util import atomic_write_bytes, atomic_write_json, canonical_json_bytes

EXIT_BY_CODE = {
    E_GL_CONTEXT: 30,
    E_SNAPSHOT_STALE: 31,
    E_CLOSURE_MISMATCH: 32,
    E_RECOVERY: 33,
    E_WORKSPACE: 34,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npr-edit",
        description="Deterministic NPR line-art editor, format converter, incremental renderer and recovery demo.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the fixed end-to-end demonstration")
    demo.add_argument("--workspace", default="npr-workspace", help="writable application workspace")
    demo.add_argument("--seed", type=int, default=DEFAULT_SEED)
    demo.add_argument("--viewport", default="256x192", help="offscreen BGRA viewport, e.g. 256x192")
    demo.add_argument("--require-gl", action="store_true", help="fail with E_GL_CONTEXT instead of software fallback")
    demo.add_argument("--no-reset", action="store_true", help="keep existing application output directories")

    verify = subparsers.add_parser("verify", help="run the demo twice and compare every deterministic artifact")
    verify.add_argument("--workspace", default="npr-verify", help="workspace for run-a/run-b")
    verify.add_argument("--seed", type=int, default=DEFAULT_SEED)
    verify.add_argument("--viewport", default="256x192")

    generate = subparsers.add_parser("generate", help="write one deterministic scene and its element-hash manifest")
    generate.add_argument("--count", type=int, required=True)
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument("--out", default="generated")
    generate.add_argument("--canvas", default="640x480")

    recover = subparsers.add_parser("recover", help="restore only the last committed state and report orphan recovery")
    recover.add_argument("--workspace", required=True)

    gl_check = subparsers.add_parser("gl-check", help="probe the local hidden OpenGL FBO backend")
    gl_check.add_argument("--workspace", default="npr-workspace", help="unused; accepted for script symmetry")

    crash = subparsers.add_parser("_crash-worker")
    crash.add_argument("workspace")
    return parser


def _parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return width, height


def _print_json(value: Any) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode("utf-8") + "\n")


def _handle_error(exc: NPRError) -> int:
    sys.stderr.write(canonical_json_bytes(exc.to_dict()).decode("utf-8") + "\n")
    return EXIT_BY_CODE.get(exc.code, 3)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "_crash-worker":
            run_crash_worker(args.workspace)
            return 0
        if args.command == "demo":
            report = run_demo(
                args.workspace,
                seed=args.seed,
                viewport=_parse_size(args.viewport),
                require_gl=args.require_gl,
                reset=not args.no_reset,
            )
            _print_json(report)
            return 0 if report["verification"]["all_steps_passed"] else 4
        if args.command == "verify":
            result = run_verification(args.workspace, seed=args.seed, viewport=_parse_size(args.viewport))
            _print_json(result)
            return 0 if result["verified"] else 4
        if args.command == "generate":
            scene = generate_scene(args.count, seed=args.seed, canvas=_parse_size(args.canvas))
            out = Path(args.out).resolve()
            out.mkdir(parents=True, exist_ok=True)
            scene_path = out / f"scene-{args.count}.json"
            manifest_path = out / f"element-hashes-{args.count}.json"
            atomic_write_bytes(scene_path, scene.project_bytes + b"\n")
            manifest = build_element_hash_manifest(scene, args.count)
            atomic_write_json(manifest_path, manifest)
            _print_json(
                {
                    "element_count": len(scene.elements),
                    "layer_count": len(scene.layers),
                    "scene_hash": scene.content_hash,
                    "scene_file": scene_path.name,
                    "manifest_file": manifest_path.name,
                    "manifest_hash": __import__("hashlib").sha256(canonical_json_bytes(manifest) + b"\n").hexdigest(),
                }
            )
            return 0
        if args.command == "recover":
            report = PersistenceStore(args.workspace).recover()
            if report is None:
                _print_json({"committed_state": None, "clean_shutdown": False})
            else:
                payload = report.as_dict()
                payload["committed_state"] = True
                _print_json(payload)
            return 0
        if args.command == "gl-check":
            _print_json(probe_offscreen_gl())
            return 0
        parser.error("unknown command")
        return 2
    except NPRError as exc:
        return _handle_error(exc)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0
