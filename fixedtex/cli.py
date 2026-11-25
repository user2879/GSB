"""Command-line entry points for validation, rendering and verification."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .config import load_commands, load_injections, load_material, load_render_config
from .diagnostics import build_diagnostics
from .errors import (
    EXIT_INTERNAL,
    EXIT_OK,
    EXIT_RENDER,
    EXIT_STATE,
    EXIT_VALIDATION,
    BizError,
    exit_code_for_codes,
)
from .io_utils import atomic_write_json
from .session import Session
from .texture import load_texture, scan_png_directory
from .verification import verify_project

INFORMATIONAL_CODES = {
    "E_CONFIG_CHANGED_REBUILD",
    "E_TEXTURE_EXTERNAL_CHANGE",
    "E_PREVIEW_STALE",
}


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _validation_exit(codes: set[str]) -> int:
    filtered = set(codes) - INFORMATIONAL_CODES
    return exit_code_for_codes(filtered)


def _issues_codes(issues: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("code")) for item in issues if item.get("code")}


def command_validate(args: argparse.Namespace) -> int:
    issues: list[dict[str, Any]] = []
    config, config_issues = load_render_config(args.config)
    issues.extend(config_issues)
    material = None
    material_issues: list[dict[str, Any]] = []
    texture = None
    texture_issues: list[dict[str, Any]] = []
    if config is not None:
        material, material_issues = load_material(args.material, len(config.layers))
        issues.extend(material_issues)
        texture_path = Path(args.texture) if args.texture else Path(config.texture_path)
        texture, texture_issues = load_texture(texture_path, expected_resolution=config.resolution)
        issues.extend(texture_issues)
    scan_report = None
    if args.scan_dir:
        scan_report = scan_png_directory(args.scan_dir, expected_resolution=config.resolution if config else None)
        issues.extend(scan_report.get("errors", []))
    report = {
        "ok": not issues,
        "exit_code": _validation_exit(_issues_codes(issues)),
        "errors": issues,
        "valid_items": {
            "config": None
            if config is None
            else {
                "name": config.name,
                "resolution": config.resolution,
                "resolution_tiers": list(config.resolution_tiers),
                "layer_count": len(config.layers),
                "layers": [layer.parameter_dict() for layer in config.layers],
                "fixed": config.fixed.to_dict(),
                "budgets": config.budgets.to_dict(),
                "blend_mode": config.blend_mode,
            },
            "material": None
            if material is None
            else {
                "name": material.name,
                "tie_rule": material.tie_rule,
                "equal_weight_rule": material.equal_weight_rule,
                "channels": {name: channel.parameter_dict() for name, channel in material.channels.items()},
            },
            "texture": texture.summary() if texture is not None else None,
            "png_scan": scan_report,
        },
    }
    if args.json_out:
        atomic_write_json(args.json_out, report)
    _json_print(report)
    return int(report["exit_code"])


def _run_session(
    *,
    config_path: str,
    material_path: str,
    state_dir: str,
    commands_path: str | None,
    injections_path: str | None,
    resume: bool,
    drain: bool,
    commit: bool,
    texture_override: str | None,
    report_path: str | None,
) -> tuple[int, dict[str, Any]]:
    root_errors: list[dict[str, Any]] = []
    command_issues: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    if commands_path:
        commands, command_issues = load_commands(commands_path)
    try:
        probe_config, probe_issues = load_render_config(config_path)
        root_errors.extend(probe_issues)
        layer_count = len(probe_config.layers) if probe_config is not None else 0
        injections, injection_issues = load_injections(injections_path, layer_count)
        root_errors.extend(injection_issues)
        session = Session.open(
            config_path,
            material_path,
            state_dir,
            resume=resume,
            injections=injections,
            texture_override=texture_override,
        )
        root_errors.extend(session.issues)
        command_results: list[dict[str, Any]] = []
        for command in commands:
            result = session.apply_command(command)
            command_results.append(result)
        if drain:
            summary = session.complete()
        else:
            session._join_job()
            summary = session.state.last_preview
        committed = None
        if commit and session.texture is not None:
            manifest = session.commit()
            committed = {
                "version": manifest["version"],
                "ledger_count": manifest["ledger_count"],
                "complete": session.state.composite is not None,
            }
        elif commit:
            committed = {"error": "E_TEXTURE_UNAVAILABLE"}
        final_codes = set(_issues_codes(root_errors)) | set(_issues_codes(command_issues))
        for result in command_results:
            final_codes |= _issues_codes(result.get("errors", []))
        final_codes |= _issues_codes(session.state.failures)
        if session.state.pending:
            final_codes.add("E_POINT_BUDGET_PENDING")
        if session.state.composite is None and not session.state.failures:
            final_codes.add("E_RENDER")
        exit_code = _validation_exit(final_codes)
        if exit_code == EXIT_OK and session.state.failures:
            exit_code = EXIT_RENDER
        report = {
            "ok": exit_code == EXIT_OK and session.state.composite is not None,
            "exit_code": exit_code,
            "status": session.status(),
            "command_issues": command_issues,
            "command_results": command_results,
            "final_summary": summary,
            "committed": committed,
            "pending": session.state.pending,
            "failures": session.state.failures,
            "injection_hits": session.state.injection_hits,
            "ledger_tail": session.state.ledger[-32:],
        }
        if report_path:
            atomic_write_json(report_path, report)
        return exit_code, report
    except BizError as exc:
        exit_code = exit_code_for_codes({exc.code})
        report = {"ok": False, "exit_code": exit_code, "errors": [exc.to_dict()], "command_issues": command_issues}
        if report_path:
            atomic_write_json(report_path, report)
        return exit_code, report
    except BaseException as exc:
        report = {
            "ok": False,
            "exit_code": EXIT_INTERNAL,
            "errors": [
                {
                    "code": "E_INTERNAL",
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            ],
        }
        if report_path:
            atomic_write_json(report_path, report)
        return EXIT_INTERNAL, report


def command_run(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    exit_code, report = _run_session(
        config_path=args.config,
        material_path=args.material,
        state_dir=str(state_dir),
        commands_path=args.commands,
        injections_path=args.injections,
        resume=not args.fresh,
        drain=not args.no_drain,
        commit=not args.no_commit,
        texture_override=args.texture,
        report_path=str(state_dir / "run-report.json"),
    )
    _json_print(report)
    return exit_code


def command_recover(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    exit_code, report = _run_session(
        config_path=args.config,
        material_path=args.material,
        state_dir=str(state_dir),
        commands_path=None,
        injections_path=args.injections,
        resume=True,
        drain=True,
        commit=True,
        texture_override=args.texture,
        report_path=str(state_dir / "recover-report.json"),
    )
    _json_print(report)
    return exit_code


def command_diagnose(args: argparse.Namespace) -> int:
    session = Session.open(
        args.config,
        args.material,
        args.state_dir,
        resume=not args.fresh,
    )
    if session.texture is None:
        report = {"ok": False, "errors": [{"code": "E_TEXTURE_UNAVAILABLE", "message": "valid texture is required"}]}
        _json_print(report)
        return EXIT_VALIDATION
    quant_steps = None
    if args.quant_steps:
        parts = [int(item) for item in args.quant_steps.split(",")]
        if len(parts) != 2:
            raise BizError("E_ARG", "--quant-steps requires two comma-separated steps")
        quant_steps = (parts[0], parts[1])
    report = build_diagnostics(
        session.config,
        session.material,
        session.texture,
        args.output_dir,
        quant_steps=quant_steps,
        extra={
            "session_version": session.version,
            "injection_hits": session.state.injection_hits,
            "failures": session.state.failures,
        },
    )
    _json_print(report)
    return EXIT_OK if report.get("ok") else EXIT_RENDER


def command_verify(args: argparse.Namespace) -> int:
    report = verify_project(
        config_path=args.config,
        material_path=args.material,
        commands_path=args.commands,
        output_dir=args.output_dir,
        injections_path=args.injections,
        low_budget_config_path=args.low_budget_config,
    )
    _json_print(report)
    return EXIT_OK if report.get("ok") else EXIT_RENDER


def command_env(args: argparse.Namespace) -> int:
    modules: dict[str, Any] = {}
    for name in ("numpy", "PIL", "tkinter"):
        try:
            module = importlib.import_module(name)
            modules[name] = {"available": True, "version": getattr(module, "__version__", "available")}
        except BaseException as exc:
            modules[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "writable": Path.cwd().is_dir() and (Path.cwd() / ".write-test").parent.exists(),
        "modules": modules,
        "note": "Tkinter is optional for this headless renderer; python3-tk is a system package.",
    }
    _json_print(report)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fixedtex", description="Deterministic layered fixed-point texture renderer")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    validate = subparsers.add_parser("validate", help="validate config, material and PNG inputs")
    validate.add_argument("--config", required=True)
    validate.add_argument("--material", required=True)
    validate.add_argument("--texture")
    validate.add_argument("--scan-dir")
    validate.add_argument("--json-out")
    validate.set_defaults(func=command_validate)

    run = subparsers.add_parser("run", help="apply a command stream and render")
    run.add_argument("--config", required=True)
    run.add_argument("--material", required=True)
    run.add_argument("--commands")
    run.add_argument("--injections")
    run.add_argument("--state-dir", default="artifacts")
    run.add_argument("--texture", help="override base texture for this run")
    run.add_argument("--fresh", action="store_true", help="do not resume the latest snapshot")
    run.add_argument("--no-drain", action="store_true", help="run one batch only")
    run.add_argument("--no-commit", action="store_true", help="do not commit after the run")
    run.set_defaults(func=command_run)

    recover = subparsers.add_parser("recover", help="resume a snapshot and finish pending batches")
    recover.add_argument("--config", required=True)
    recover.add_argument("--material", required=True)
    recover.add_argument("--injections")
    recover.add_argument("--state-dir", default="artifacts")
    recover.add_argument("--texture")
    recover.set_defaults(func=command_recover)

    diagnose = subparsers.add_parser("diagnose", help="build stage-wise error and quantization diagnostics")
    diagnose.add_argument("--config", required=True)
    diagnose.add_argument("--material", required=True)
    diagnose.add_argument("--state-dir", default="artifacts")
    diagnose.add_argument("--output-dir", default="artifacts/diagnostics")
    diagnose.add_argument("--quant-steps")
    diagnose.add_argument("--fresh", action="store_true")
    diagnose.set_defaults(func=command_diagnose)

    verify = subparsers.add_parser("verify", help="run acceptance checks for incremental, recovery and faults")
    verify.add_argument("--config", default="samples/config.json")
    verify.add_argument("--material", default="samples/material.json")
    verify.add_argument("--commands", default="samples/commands.jsonl")
    verify.add_argument("--injections", default="samples/injections.json")
    verify.add_argument("--low-budget-config", default="samples/config_low_budget.json")
    verify.add_argument("--output-dir", default="artifacts/verification")
    verify.set_defaults(func=command_verify)

    env = subparsers.add_parser("env", help="report local Python dependency readiness")
    env.set_defaults(func=command_env)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BizError as exc:
        _json_print({"ok": False, "exit_code": exit_code_for_codes({exc.code}), "errors": [exc.to_dict()]})
        return exit_code_for_codes({exc.code})
    except BaseException as exc:
        _json_print(
            {
                "ok": False,
                "exit_code": EXIT_INTERNAL,
                "errors": [
                    {
                        "code": "E_INTERNAL",
                        "message": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                ],
            }
        )
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
