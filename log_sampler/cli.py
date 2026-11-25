"""Command-line interface for the diagnostic sampling component."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, read_json
from .evidence import convert_evidence, load_evidence_file
from .pipeline import Pipeline, RuntimeConfig, compare_logical_results
from .state import CommandError, SessionStore
from .validation import load_batches, load_config, load_json_file


def _load_commands(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload, issues = load_json_file(path, "commands")
    if issues:
        raise CommandError(issues[0].message)
    commands = payload.get("commands") if isinstance(payload, dict) else payload
    if not isinstance(commands, list):
        raise CommandError("commands must be a list or an object with a commands list")
    return commands


def _load_injections(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload, issues = load_json_file(path, "injections")
    if issues:
        raise CommandError(issues[0].message)
    injections = payload.get("injections") if isinstance(payload, dict) else payload
    if not isinstance(injections, list):
        raise CommandError(
            "injections must be a list or an object with an injections list"
        )
    return injections


def _prepare_session(
    state_root: Path,
    config_path: str | None,
    batches_path: str | None,
    injections_path: str | None,
) -> SessionStore:
    store = SessionStore(state_root)
    if store.path.exists():
        return store
    if not config_path or not batches_path:
        raise CommandError("new state requires --config and --batches")
    config, config_issues = load_config(config_path)
    batches, batch_issues = load_batches(batches_path)
    if config_issues or batch_issues:
        for issue in [*config_issues, *batch_issues]:
            print(json.dumps({"error": issue.to_dict()}, ensure_ascii=False))
    injections = _load_injections(injections_path)
    store.initialize(config, batches, injections)
    return store


def _run(
    state_root: Path,
    config_path: str | None,
    batches_path: str | None,
    injections_path: str | None,
    commands_path: str | None,
    crash_after: str | None,
) -> dict[str, Any]:
    store = _prepare_session(
        state_root, config_path, batches_path, injections_path
    )
    commands = _load_commands(commands_path)
    if commands:
        store.apply_commands(commands)
    config, batches, injections = store.effective_input()
    result = Pipeline(
        config,
        batches,
        injections,
        RuntimeConfig(state_root=state_root, crash_after=crash_after),
    ).run()
    return {
        "status": "ok",
        "state_root": str(state_root),
        "global_version": result.manifest["global_version"],
        "version_dir": str(result.version_dir),
        "latest_result": str(state_root / "latest_result.json"),
        "final_sample_count": len(result.logical_result["final_samples"]),
        "failure_count": len(result.logical_result["validation"]["failures"]),
        "bucket_count": len(result.logical_result["buckets"]),
        "stage_versions": result.manifest["stage_versions"],
        "exchange_version": result.logical_result["exchange_version"],
    }


def _latest_logical(state_root: Path) -> dict[str, Any] | None:
    path = state_root / "latest_result.json"
    if not path.exists():
        return None
    return read_json(path)


def cmd_run(args: argparse.Namespace) -> int:
    try:
        output = _run(
            Path(args.state),
            args.config,
            args.batches,
            args.injections,
            args.commands,
            args.crash_after,
        )
    except CommandError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    source_state = Path(args.source_state)
    destination = Path(args.state)
    try:
        if args.fresh and destination.exists():
            shutil.rmtree(destination)
        session = SessionStore(source_state).load()
        config = session["effective_config"]
        batches = session["batches"]
        injections = session.get("injections", [])
        store = SessionStore(destination)
        store.initialize(config, batches, injections)
        result = Pipeline(
            config,
            batches,
            injections,
            RuntimeConfig(state_root=destination),
        ).run()
    except (CommandError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    left = _latest_logical(source_state)
    diffs = compare_logical_results(left, result.logical_result) if left else []
    comparison = {
        "status": "match" if left is not None and not diffs else "mismatch" if left else "no_source_result",
        "differences": diffs,
        "incremental_state": str(source_state),
        "replay_state": str(destination),
        "replay_version": result.manifest["global_version"],
    }
    atomic_write_json(destination / "replay_comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, sort_keys=True))
    return 0 if comparison["status"] == "match" else 1


def cmd_convert(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    sidecar = None
    if args.metadata:
        metadata, issues = load_json_file(args.metadata, "metadata")
        if issues:
            print(json.dumps({"status": "error", "issues": [item.to_dict() for item in issues]}, ensure_ascii=False))
            return 2
        sidecar = metadata
    converted = convert_evidence(
        input_path.read_bytes(), args.from_version, args.to_version, sidecar
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converted.payload)
    metadata_out = Path(args.metadata_out) if args.metadata_out else output_path.with_suffix(
        output_path.suffix + ".metadata.json"
    )
    atomic_write_json(metadata_out, converted.sidecar)
    output = {
        "status": "ok" if not converted.issues else "issues",
        "from": args.from_version,
        "to": args.to_version,
        "output": str(output_path),
        "metadata_out": str(metadata_out),
        "slots": len(converted.slots),
        "byte_length": len(converted.payload),
        "issues": [item.to_dict() for item in converted.issues],
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if not converted.issues else 1


def cmd_compare(args: argparse.Namespace) -> int:
    left = read_json(args.left)
    right = read_json(args.right)
    diffs = compare_logical_results(left, right)
    print(
        json.dumps(
            {
                "status": "match" if not diffs else "mismatch",
                "difference_count": len(diffs),
                "differences": diffs[:50],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not diffs else 1


def cmd_validate_evidence(args: argparse.Namespace) -> int:
    result = load_evidence_file(args.input, args.version, args.metadata)
    print(
        json.dumps(
            {
                "status": "ok" if not result.issues else "issues",
                "version": args.version,
                "slots": len(result.slots),
                "issues": [item.to_dict() for item in result.issues],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not result.issues else 1


def cmd_verify(args: argparse.Namespace) -> int:
    from .demo import run_verification

    report = run_verification(
        config_path=Path(args.config),
        batches_path=Path(args.batches),
        injections_path=Path(args.injections) if args.injections else None,
        commands_path=Path(args.commands) if args.commands else None,
        state_root=Path(args.state),
        fresh=args.fresh,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="log-sampler",
        description="Deterministic staged device-log reservoir sampler",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="initialize/apply commands and run pipeline")
    run.add_argument("--state", required=True)
    run.add_argument("--config")
    run.add_argument("--batches")
    run.add_argument("--injections")
    run.add_argument("--commands")
    run.add_argument("--crash-after", choices=None, help="validate, bucket:N, truncate, serialize")
    run.set_defaults(func=cmd_run)

    replay = subparsers.add_parser("replay", help="blank-cache full replay from effective session")
    replay.add_argument("--source-state", required=True)
    replay.add_argument("--state", required=True)
    replay.add_argument("--fresh", action="store_true")
    replay.set_defaults(func=cmd_replay)

    convert = subparsers.add_parser("convert", help="convert p1/p2/p3 evidence with sidecar metadata")
    convert.add_argument("--from", dest="from_version", required=True, choices=["p1", "p2", "p3"])
    convert.add_argument("--to", dest="to_version", required=True, choices=["p1", "p2", "p3"])
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--metadata")
    convert.add_argument("--metadata-out")
    convert.set_defaults(func=cmd_convert)

    compare = subparsers.add_parser("compare", help="field-exact comparison of logical results")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.set_defaults(func=cmd_compare)

    validate_evidence = subparsers.add_parser("validate-evidence")
    validate_evidence.add_argument("--version", required=True, choices=["p1", "p2", "p3"])
    validate_evidence.add_argument("--input", required=True)
    validate_evidence.add_argument("--metadata")
    validate_evidence.set_defaults(func=cmd_validate_evidence)

    verify = subparsers.add_parser("verify", help="run the full reproducible acceptance demo")
    verify.add_argument("--config", required=True)
    verify.add_argument("--batches", required=True)
    verify.add_argument("--injections")
    verify.add_argument("--commands")
    verify.add_argument("--state", required=True)
    verify.add_argument("--fresh", action="store_true")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

