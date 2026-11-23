"""CLI entry point."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import load_config
from .executor import CommandExecutor, execution_summary
from .hashing import sha256_json, stable_json
from .validation import merge_issues, parse_commands_jsonl, parse_samples_jsonl


DEFAULT_CONFIG = "samples/position_encoding.json"
DEFAULT_SAMPLES = "samples/attention_samples.jsonl"
DEFAULT_COMMANDS = "samples/commands.jsonl"


def _clean(*paths: str | Path) -> None:
    for path in paths:
        target = Path(path)
        if target.exists():
            shutil.rmtree(target)


def _load_all(args: argparse.Namespace):
    config, config_issues = load_config(args.config)
    if config is None:
        return None, [], [], [], config_issues
    samples = parse_samples_jsonl(args.samples, config)
    commands = parse_commands_jsonl(args.commands, config)
    issues = merge_issues(samples.issues, commands.issues)
    sample_content_hash = sha256_json(
        [sample.to_canonical_dict() for sample in samples.samples]
    )
    return config, samples.samples, commands.commands, issues, sample_content_hash


def _cmd_validate(args: argparse.Namespace) -> int:
    config, config_issues = load_config(args.config)
    if config is None:
        print(stable_json({"ok": False, "issues": [issue.to_dict() for issue in config_issues]}, pretty=True))
        return 1
    samples = parse_samples_jsonl(args.samples, config)
    commands = parse_commands_jsonl(args.commands, config)
    issue_list = merge_issues(samples.issues, commands.issues)
    payload = {
        "ok": not issue_list,
        "config": config.to_dict(),
        "valid_sample_count": len(samples.samples),
        "invalid_sample_issue_count": len(samples.issues),
        "valid_command_count": len(commands.commands),
        "command_issue_count": len(commands.issues),
        "issues": [issue.to_dict() for issue in issue_list],
    }
    print(stable_json(payload, pretty=True))
    return 0 if not issue_list else 2


def _cmd_run(args: argparse.Namespace) -> int:
    if args.clean:
        _clean(args.state_dir, args.reports_dir)
    config, config_issues = load_config(args.config)
    if config is None:
        print(
            stable_json(
                {
                    "exit_code": 1,
                    "issues": [issue.to_dict() for issue in config_issues],
                },
                pretty=True,
            )
        )
        return 1
    samples = parse_samples_jsonl(args.samples, config)
    commands = parse_commands_jsonl(args.commands, config)
    validation_issues = list(samples.issues)
    command_issues = list(commands.issues)
    sample_content_hash = sha256_json(
        [sample.to_canonical_dict() for sample in samples.samples]
    )
    executor = CommandExecutor(
        config,
        samples.samples,
        commands.commands,
        state_dir=args.state_dir,
        reports_dir=args.reports_dir,
        sample_content_hash=sample_content_hash,
        validation_issues=validation_issues,
        command_issues=command_issues,
    )
    result = executor.execute()
    print(stable_json(execution_summary(result), pretty=True))
    return result.exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pe-diagnostics",
        description="Deterministic long-context positional-encoding diagnostics workbench",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_parser = subparsers.add_parser("run", help="execute a command stream")
    run_parser.add_argument("--config", default=DEFAULT_CONFIG)
    run_parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    run_parser.add_argument("--commands", default=DEFAULT_COMMANDS)
    run_parser.add_argument("--state-dir", default="state")
    run_parser.add_argument("--reports-dir", default="reports")
    run_parser.add_argument("--clean", action="store_true", help="remove state/reports before running")
    run_parser.set_defaults(func=_cmd_run)

    validate_parser = subparsers.add_parser("validate", help="validate inputs without executing")
    validate_parser.add_argument("--config", default=DEFAULT_CONFIG)
    validate_parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    validate_parser.add_argument("--commands", default=DEFAULT_COMMANDS)
    validate_parser.set_defaults(func=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
