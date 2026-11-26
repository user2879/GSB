from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .acceptance import run_acceptance
from .demo import create_demo_workspace
from .engine import FingerprintEngine
from .manifest import load_json_file, validate_commands, validate_manifest
from .workspace import apply_command_to_inputs


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def command_generate(args: argparse.Namespace) -> int:
    result = create_demo_workspace(args.workspace)
    _print({"status": "generated", **result})
    return 0


def command_validate(args: argparse.Namespace) -> int:
    root = Path(args.workspace)
    manifest, load_issues = load_json_file(root / "manifest.json")
    config, config_issues = load_json_file(root / "config.json")
    from .config import parse_config

    parsed_config, parse_issues = parse_config(config if isinstance(config, dict) else {})
    clips, manifest_issues = validate_manifest(manifest if isinstance(manifest, dict) else {}, root / "data")
    command_payload, _ = load_json_file(root / "edit_commands.json")
    commands, command_issues = validate_commands(command_payload)
    result = {
        "valid_clip_count": len(clips),
        "valid_command_count": len(commands),
        "issues": load_issues + config_issues + parse_issues + manifest_issues + command_issues,
        "issue_codes": [
            item["code"]
            for item in load_issues + config_issues + parse_issues + manifest_issues + command_issues
        ],
        "internal_sample_rate": parsed_config["internal_sample_rate"],
    }
    _print(result)
    return 0


def command_rebuild(args: argparse.Namespace) -> int:
    engine = FingerprintEngine.from_workspace(args.workspace, args.state)
    engine.serial = args.serial
    engine.install_signal_handlers()
    engine.start_run(run_id=args.run_id or "cli-rebuild")
    result = engine.run_until_complete(max_batches=args.max_batches)
    _print(
        {
            "status": result.get("status"),
            "completed": result.get("completed"),
            "global_version": engine.snapshot["global_version"],
            "canonical_fingerprint": engine.canonical_fingerprint(),
            "last_receipt": {
                key: value
                for key, value in (engine.snapshot.get("last_receipt") or {}).items()
                if key != "ledger"
            },
        }
    )
    return 0 if result.get("completed") else 2


def command_apply(args: argparse.Namespace) -> int:
    root = Path(args.workspace)
    engine = FingerprintEngine.from_workspace(root, args.state)
    engine.serial = args.serial
    engine.install_signal_handlers()
    payload, load_issues = load_json_file(args.commands)
    commands, issues = validate_commands(payload)
    if load_issues:
        issues.extend(load_issues)
    summaries: list[dict[str, Any]] = []
    for command in commands:
        apply_issues = apply_command_to_inputs(engine.manifest, engine.config, command, root / "data")
        if apply_issues:
            issues.extend(apply_issues)
            continue
        engine._context = None
        engine.start_run(command)
        result = engine.run_until_complete()
        receipt = engine.snapshot.get("last_receipt") or {}
        summaries.append(
            {
                "op": command.get("op"),
                "command_id": command.get("command_id"),
                "completed": result.get("completed"),
                "reuse_hit_count": receipt.get("reuse_hit_count"),
                "rebuilt_count": receipt.get("rebuilt_count"),
            }
        )
    _print(
        {
            "applied": summaries,
            "validation_issues": issues,
            "canonical_fingerprint": engine.canonical_fingerprint(),
            "global_version": engine.snapshot["global_version"],
        }
    )
    return 0


def command_verify(_args: argparse.Namespace) -> int:
    report = run_acceptance(write_report=True)
    _print({"acceptance_passed": report["acceptance_passed"], "report": "artifacts/verification_report.json"})
    return 0 if report["acceptance_passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afp", description="Deterministic audio fingerprint diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate deterministic sample workspace")
    generate.add_argument("--workspace", default="sample_workspace")
    generate.set_defaults(func=command_generate)

    validate = sub.add_parser("validate", help="validate manifest, config and command stream")
    validate.add_argument("--workspace", default="sample_workspace")
    validate.set_defaults(func=command_validate)

    rebuild = sub.add_parser("rebuild", help="build frames, peaks, pairs, buckets and query results")
    rebuild.add_argument("--workspace", default="sample_workspace")
    rebuild.add_argument("--state", default="state/cli")
    rebuild.add_argument("--run-id", default=None)
    rebuild.add_argument("--serial", action="store_true")
    rebuild.add_argument("--max-batches", type=int, default=10000)
    rebuild.set_defaults(func=command_rebuild)

    apply_cmd = sub.add_parser("apply", help="validate and apply a command stream")
    apply_cmd.add_argument("--workspace", default="sample_workspace")
    apply_cmd.add_argument("--commands", required=True)
    apply_cmd.add_argument("--state", default="state/cli")
    apply_cmd.add_argument("--serial", action="store_true")
    apply_cmd.set_defaults(func=command_apply)

    verify = sub.add_parser("verify", help="run all local acceptance checks")
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
