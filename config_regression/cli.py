"""Command line entry point for the configuration regression platform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .canonical import canonical_json
from .differ import diff_manifests
from .errorbound import analyze_error_bounds
from .errors import PlatformError
from .model import (
    SessionConfig,
    load_json_file,
    load_manifest,
    load_registry,
    assert_version_monotonic,
)
from .pipeline import recover_session, run_session


def _load_inputs(args: argparse.Namespace):
    registry_data = load_json_file(args.registry)
    registry = load_registry(registry_data)
    baseline = load_manifest(load_json_file(args.baseline), registry,
                             "baseline")
    candidate = load_manifest(load_json_file(args.candidate), registry,
                              "candidate")
    assert_version_monotonic(baseline, candidate)
    session_data = load_json_file(args.session) if args.session else {}
    session = SessionConfig.from_dict(session_data)
    return baseline, candidate, registry, session


def _print(payload: dict, output: str | None) -> None:
    text = canonical_json(payload, indent=2)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _error(exc: PlatformError) -> int:
    payload = {"error_code": exc.code, "message": exc.message,
               "violations": exc.details()}
    print(canonical_json(payload, indent=2), file=sys.stderr)
    return 2


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        baseline, candidate, registry, session = _load_inputs(args)
    except PlatformError as exc:
        return _error(exc)
    _print({
        "status": "ok",
        "registry_nodes": registry.node_ids,
        "baseline": {"version": baseline.version,
                     "item_count": len(baseline.items),
                     "topological_order": baseline.order},
        "candidate": {"version": candidate.version,
                      "item_count": len(candidate.items),
                      "topological_order": candidate.order},
    }, args.output)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    try:
        baseline, candidate, _registry, session = _load_inputs(args)
    except PlatformError as exc:
        return _error(exc)
    diff = diff_manifests(baseline, candidate)
    bounds = analyze_error_bounds(baseline, candidate, diff,
                                  session.scale_exponent)
    _print({"diff": diff.as_dict(), "error_bounds": bounds}, args.output)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        baseline, candidate, registry, session = _load_inputs(args)
        artifacts = run_session(baseline, candidate, registry, session,
                                args.database)
    except PlatformError as exc:
        return _error(exc)
    _print(artifacts.as_dict(), args.output)
    return 1 if artifacts.verdict != "ok" else 0


def cmd_recover(args: argparse.Namespace) -> int:
    try:
        baseline, candidate, registry, session = _load_inputs(args)
        artifacts = recover_session(baseline, candidate, registry, session,
                                    args.database)
    except PlatformError as exc:
        return _error(exc)
    _print(artifacts.as_dict(), args.output)
    return 1 if artifacts.verdict != "ok" else 0


def cmd_gui(_args: argparse.Namespace) -> int:
    try:
        from .gui import launch_viewer
    except ImportError as exc:
        print(f"Tkinter is unavailable in this environment: {exc}",
              file=sys.stderr)
        return 3
    launch_viewer()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfgreg",
        description="local configuration distribution regression platform")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--baseline", required=True)
        p.add_argument("--candidate", required=True)
        p.add_argument("--registry", required=True)
        p.add_argument("--session", default=None)
        p.add_argument("--output", default=None)
        p.add_argument("--database", default="artifacts/session.db")

    p_validate = sub.add_parser("validate", help="strong manifest validation")
    add_common(p_validate)
    p_validate.set_defaults(func=cmd_validate)

    p_diff = sub.add_parser("diff", help="diff + impact + error bounds")
    add_common(p_diff)
    p_diff.set_defaults(func=cmd_diff)

    p_run = sub.add_parser("run", help="full collection + regression session")
    add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_rec = sub.add_parser("recover", help="resume after forced termination")
    add_common(p_rec)
    p_rec.set_defaults(func=cmd_recover)

    p_gui = sub.add_parser("gui", help="open the Tkinter report viewer")
    p_gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
