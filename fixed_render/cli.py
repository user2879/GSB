from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import RenderConfig
from .errors import BusinessError
from .inputs import load_command_stream, load_conditions
from .lease import LeaseServer
from .samples import generate_all_samples
from .storage import atomic_write_json
from .workbench import CommandSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fixed_render", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init-samples", help="generate deterministic local samples")
    init_parser.add_argument("--root", default=".")

    render_parser = sub.add_parser("render", help="render one config into a workspace")
    render_parser.add_argument("--root", default=".")
    render_parser.add_argument("--config", default="configs/render.yaml")
    render_parser.add_argument("--workspace", default="runs/manual")
    render_parser.add_argument("--ledger-name", default="render")
    render_parser.add_argument("--preview-budget-ms", type=int, default=None)
    render_parser.add_argument("--lease-ttl-ms", type=int, default=None)

    commands_parser = sub.add_parser("run-commands", help="apply and render a command stream")
    commands_parser.add_argument("--root", default=".")
    commands_parser.add_argument("--config", default="configs/render.yaml")
    commands_parser.add_argument("--commands", default="commands/edit_sequence.yaml")
    commands_parser.add_argument("--workspace", default="runs/incremental")
    commands_parser.add_argument("--report", default="artifacts/command_run.json")
    commands_parser.add_argument("--preview-budget-ms", type=int, default=None)
    commands_parser.add_argument("--lease-ttl-ms", type=int, default=None)

    diagnose_parser = sub.add_parser("diagnose", help="run built-in verification and fault injection")
    diagnose_parser.add_argument("--root", default=".")
    diagnose_parser.add_argument("--report", default="artifacts/diagnostic_report.json")

    verify_parser = sub.add_parser("verify", help="run full local acceptance verification")
    verify_parser.add_argument("--root", default=".")
    verify_parser.add_argument("--report", default="artifacts/verification_report.json")

    gui_parser = sub.add_parser("gui", help="open Tkinter ledger workbench")
    gui_parser.add_argument("--report", default="artifacts/diagnostic_report.json")
    gui_parser.add_argument("--smoke-ms", type=int, default=0)

    lease_parser = sub.add_parser("serve-lease", help="run loopback lease executor")
    lease_parser.add_argument("--host", default="127.0.0.1")
    lease_parser.add_argument("--port", type=int, default=48765)
    lease_parser.add_argument("--slots", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init-samples":
            paths = generate_all_samples(args.root)
            print(json.dumps({"ok": True, "samples": paths}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "render":
            result = _render_once(args)
            print(json.dumps(_public_render(result), ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-commands":
            report = _run_commands(args)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            return 0 if report["summary"]["ok"] else 2
        if args.command == "diagnose":
            from .diagnostics import run_diagnostics

            report = run_diagnostics(Path(args.root), Path(args.report))
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            return 0 if report["summary"]["failed_cases"] == 0 else 2
        if args.command == "verify":
            from .verification import run_verification

            report = run_verification(Path(args.root), Path(args.report))
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            return 0 if report["summary"]["ok"] else 2
        if args.command == "gui":
            from .gui import run_gui

            return run_gui(Path(args.report), smoke_ms=args.smoke_ms)
        if args.command == "serve-lease":
            with LeaseServer(args.host, args.port, args.slots) as server:
                print(json.dumps({"ok": True, "host": server.host, "port": server.port}))
                try:
                    server.thread.join()
                except KeyboardInterrupt:
                    pass
            return 0
        return 1
    except BusinessError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 3


def _load_config(args: argparse.Namespace) -> RenderConfig:
    root = Path(args.root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = RenderConfig.load(config_path)
    changes: dict[str, Any] = {}
    if getattr(args, "preview_budget_ms", None) is not None:
        changes["preview_budget_ms"] = args.preview_budget_ms
    if getattr(args, "lease_ttl_ms", None) is not None:
        changes["lease_ttl_ms"] = args.lease_ttl_ms
    if changes:
        config = config.with_changes(changes)
    return config


def _render_once(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    config = _load_config(args)
    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = root / workspace
    with CommandSession(root, workspace, config) as session:
        decision = session.engine.startup_decision(session.config, load_conditions(session.config))
        result = session.render_current(ledger_name=args.ledger_name)
        result["startup_decision"] = decision
        return result


def _run_commands(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    config = _load_config(args)
    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = root / workspace
    command_path = Path(args.commands)
    if not command_path.is_absolute():
        command_path = root / command_path
    commands = load_command_stream(command_path)

    with CommandSession(root, workspace, config) as session:
        session.validate_command_stream(commands)
        baseline = session.render_current(ledger_name="00_baseline")
        command_reports: list[dict[str, Any]] = []
        for index, command in enumerate(commands, 1):
            name = f"{index:02d}_{command['op']}"
            result = session.apply_and_render(command, ledger_name=name)
            manifest = result["manifest"]
            command_reports.append(
                {
                    "index": index,
                    "op": command["op"],
                    "preview": result["preview"],
                    "recompute_order": manifest["recompute_order"],
                    "reuse_order": manifest["reuse_order"],
                    "failed_order": manifest["failed_order"],
                    "affected_frame_range": manifest["affected_frame_range"],
                    "version": manifest["version"],
                }
            )
        summary = {
            "ok": baseline["manifest"]["ok"] and all(not item["failed_order"] for item in command_reports),
            "workspace": str(workspace),
            "command_count": len(commands),
            "final_version": command_reports[-1]["version"] if command_reports else 1,
            "final_recompute_order": command_reports[-1]["recompute_order"] if command_reports else [],
            "final_reuse_order": command_reports[-1]["reuse_order"] if command_reports else [],
            "reports": command_reports,
        }
        report = {"summary": summary}
        if args.report:
            report_path = Path(args.report)
            if not report_path.is_absolute():
                report_path = root / report_path
            atomic_write_json(report_path, report)
        return report


def _public_render(result: dict[str, Any]) -> dict[str, Any]:
    manifest = result["manifest"]
    return {
        "ok": manifest["ok"],
        "version": manifest["version"],
        "stage_order": manifest["stage_order"],
        "recompute_order": manifest["recompute_order"],
        "reuse_order": manifest["reuse_order"],
        "frames": manifest["frames"],
        "affected_frame_range": manifest["affected_frame_range"],
        "ledger_path": manifest["ledger_path"],
        "startup_decision": result.get("startup_decision"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
