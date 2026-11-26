"""Command line entry point for generation, validation and verification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .commands import validate_command_stream
from .engine import LabEngine
from .generator import initialize_workspace
from .tk_bootstrap import tk_status
from .validation import load_json, load_jsonl, validate_workspace
from .workflows import run_verification, write_json


def _init(args: argparse.Namespace) -> int:
    workspace = initialize_workspace(args.root)
    print(json.dumps({"initialized": str(Path(args.root).resolve()), "samples": len(workspace["samples"]), "commands": len(workspace["commands"])}, ensure_ascii=False))
    return 0


def _validate(args: argparse.Namespace) -> int:
    config, config_errors = load_json(args.config)
    samples, sample_errors = load_jsonl(args.samples)
    errors = list(config_errors) + list(sample_errors)
    if config is not None:
        errors.extend(validate_workspace(config, samples))
        if args.commands:
            commands, command_parse_errors = load_jsonl(args.commands)
            errors.extend(command_parse_errors)
            errors.extend(validate_command_stream(commands, config, samples))
    report = {"ok": not errors, "error_count": len(errors), "errors": errors}
    if args.out:
        write_json(args.out, report)
    print(json.dumps({"ok": report["ok"], "error_count": report["error_count"]}, ensure_ascii=False))
    return 0 if not errors else 2


def _verify(args: argparse.Namespace) -> int:
    summary = run_verification(args.root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


def _tk_status(args: argparse.Namespace) -> int:
    status = tk_status(create_root=args.create_window)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status.get("tkinter_available") else 2


def _gui(args: argparse.Namespace) -> int:
    from .gui import launch_gui

    return launch_gui(args.root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rope-lab", description="Deterministic fixed-point RoPE diagnostic lab")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="generate config, samples, commands and injections")
    init_parser.add_argument("--root", default=".")
    init_parser.set_defaults(func=_init)

    validate_parser = sub.add_parser("validate", help="validate JSON config and JSONL samples")
    validate_parser.add_argument("--config", default="configs/lab_config.json")
    validate_parser.add_argument("--samples", default="data/samples.jsonl")
    validate_parser.add_argument("--commands", default="commands/edit_commands.jsonl")
    validate_parser.add_argument("--out")
    validate_parser.set_defaults(func=_validate)

    verify_parser = sub.add_parser("verify", help="run full local acceptance workflow")
    verify_parser.add_argument("--root", default=".")
    verify_parser.set_defaults(func=_verify)

    tk_parser = sub.add_parser("tk-status", help="check Tkinter and optionally create a window")
    tk_parser.add_argument("--create-window", action="store_true")
    tk_parser.set_defaults(func=_tk_status)

    gui_parser = sub.add_parser("gui", help="launch Tkinter diagnostic workbench")
    gui_parser.add_argument("--root", default=".")
    gui_parser.set_defaults(func=_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
