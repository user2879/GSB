from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .engine import RenderEngine
from .errors import DiagnosticError
from .resources import load_command_stream
from .samples import generate_sample_workspace, validate_sample_workspace
from .util import atomic_write_json


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _cmd_sample(args: argparse.Namespace) -> int:
    _print_json(generate_sample_workspace(Path(args.out).resolve()))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    _print_json(
        validate_sample_workspace(
            Path(args.config).resolve(),
            None if args.commands is None else Path(args.commands).resolve(),
        )
    )
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).resolve())
    workspace = Path(args.workspace).resolve()
    commands = load_command_stream(Path(args.commands).resolve()) if args.commands else []
    engine = RenderEngine(
        config,
        workspace,
        prompt_name=args.prompt,
        image_path=args.image,
        motion_path=args.motion,
        store_root=args.store,
        output_dir=args.output,
        injections_path=args.injections,
        restore=not args.no_restore,
    )
    responses: list[dict[str, Any]] = []
    initial = engine.drain_current()
    responses.append(initial)
    if initial["code"] not in {"OK", "E_PARTIAL_FAILURE"}:
        raise RuntimeError(f"initial render failed: {initial}")
    for command in commands:
        response = engine.run_command(command)
        responses.append(response)
        if response["code"] not in {"OK", "E_PARTIAL_FAILURE"}:
            raise RuntimeError(f"command {command.index} failed: {response['failures']}")
    report = {
        "responses": responses,
        "results": engine.results(),
        "diagnostics": engine.diagnostic_report(),
        "snapshot_versions": engine.snapshot_store.versions(),
    }
    if args.report:
        atomic_write_json(Path(args.report).resolve(), report)
    _print_json(
        {
            "last_response": responses[-1],
            "frames": report["results"]["frames"],
            "artifact_count": engine.artifact_store.count(),
            "snapshot_versions": report["snapshot_versions"],
            "report": None if args.report is None else str(Path(args.report).resolve()),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="renderdiag", description="Deterministic multimodal render diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="generate a deterministic local sample workspace")
    sample.add_argument("--out", required=True)
    sample.set_defaults(func=_cmd_sample)

    validate = subparsers.add_parser("validate", help="validate config, prompts, PNG, motion and command stream")
    validate.add_argument("--config", required=True)
    validate.add_argument("--commands")
    validate.set_defaults(func=_cmd_validate)

    render = subparsers.add_parser("render", help="render the initial state and optional command stream")
    render.add_argument("--config", required=True)
    render.add_argument("--workspace", required=True)
    render.add_argument("--commands")
    render.add_argument("--prompt", default="p0")
    render.add_argument("--image")
    render.add_argument("--motion")
    render.add_argument("--store")
    render.add_argument("--output")
    render.add_argument("--injections")
    render.add_argument("--report")
    render.add_argument("--no-restore", action="store_true")
    render.set_defaults(func=_cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DiagnosticError as error:
        _print_json(error.to_dict())
        return 2
    except Exception as exc:
        _print_json({"code": "E_INTERNAL", "message": str(exc), "error": repr(exc)})
        return 1
