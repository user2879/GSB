"""Command line interface for the fixed-point multimodal renderer."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from .errors import ErrorBundle
from .pipeline import Renderer
from .samplegen import default_config, generate_workspace
from .state import RenderState
from .workspace import load_workspace


def _print_errors(exc: ErrorBundle) -> int:
    for err in exc.errors:
        print(str(err), file=sys.stderr)
    return 2


def cmd_init(args) -> int:
    workdir = os.path.abspath(args.workdir)
    if os.path.exists(workdir) and os.listdir(workdir) and not args.force:
        print(f"workdir {workdir} not empty (use --force)", file=sys.stderr)
        return 2
    os.makedirs(workdir, exist_ok=True)
    config = default_config()
    generate_workspace(workdir, config, args.frames, seed=args.seed)
    print(f"initialized deterministic sample workspace at {workdir}")
    return 0


def _load_state(workdir: str) -> tuple[RenderState, dict]:
    loaded = load_workspace(workdir)
    state = RenderState(loaded["config"], len(loaded["frame_indices"]),
                        loaded["prompts"], loaded["images"], loaded["motions"])
    return state, loaded


def cmd_render(args) -> int:
    workdir = os.path.abspath(args.workdir)
    try:
        state, loaded = _load_state(workdir)
        for command in loaded["commands"]:
            state.apply(command)
        renderer = Renderer(workdir)
        outcome = renderer.run(state, resume=args.resume,
                               force_crash_after=args.crash_after)
    except ErrorBundle as exc:
        return _print_errors(exc)
    print(json.dumps({
        "version": outcome.version,
        "reused": outcome.reused,
        "recomputed": outcome.recomputed,
        "degraded_frames": outcome.report["degraded_frames"],
        "output_dir": outcome.output_dir,
    }, indent=2))
    return 0


def cmd_incremental(args) -> int:
    """Run an explicit edit command then an incremental render."""
    workdir = os.path.abspath(args.workdir)
    try:
        state, _ = _load_state(workdir)
        state_path = os.path.join(workdir, "state.json")
        if os.path.exists(state_path):
            state = RenderState.load(state_path)
        command = json.loads(args.command)
        state.apply(command)
        renderer = Renderer(workdir)
        outcome = renderer.run(state)
    except ErrorBundle as exc:
        return _print_errors(exc)
    print(json.dumps({"version": outcome.version,
                      "reused": outcome.reused,
                      "recomputed": outcome.recomputed,
                      "degraded_frames": outcome.report["degraded_frames"]},
                     indent=2))
    return 0


def cmd_validate(args) -> int:
    workdir = os.path.abspath(args.workdir)
    try:
        loaded = load_workspace(workdir)
    except ErrorBundle as exc:
        return _print_errors(exc)
    print(json.dumps({
        "status": "valid",
        "frames": loaded["frame_indices"],
        "config_hash": __import__(
            "fixedpoint_render.fingerprints", fromlist=["fingerprint"]
        ).fingerprint(loaded["config"].canonical_json(), "config-hash"),
    }, indent=2))
    return 0


def cmd_report(args) -> int:
    path = os.path.join(os.path.abspath(args.workdir),
                        "diagnostic_report.json")
    if not os.path.exists(path):
        print("no diagnostic report found", file=sys.stderr)
        return 2
    with open(path, encoding="utf-8") as fh:
        print(json.dumps(json.load(fh), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fixedpoint-render",
        description="Deterministic fixed-point multimodal rendering pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="generate deterministic samples")
    p_init.add_argument("workdir")
    p_init.add_argument("--frames", type=int, default=5)
    p_init.add_argument("--seed", type=int, default=20261127)
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_render = sub.add_parser("render", help="full or resumed render")
    p_render.add_argument("workdir")
    p_render.add_argument("--resume", action="store_true")
    p_render.add_argument("--crash-after", default=None,
                          help="simulate forced termination after a stage")
    p_render.set_defaults(func=cmd_render)

    p_inc = sub.add_parser("edit", help="apply one edit command incrementally")
    p_inc.add_argument("workdir")
    p_inc.add_argument("command", help="single JSON command")
    p_inc.set_defaults(func=cmd_incremental)

    p_val = sub.add_parser("validate", help="validate inputs only")
    p_val.add_argument("workdir")
    p_val.set_defaults(func=cmd_validate)

    p_rep = sub.add_parser("report", help="print diagnostic report")
    p_rep.add_argument("workdir")
    p_rep.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
