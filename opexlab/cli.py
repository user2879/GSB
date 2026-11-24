"""Command-line interface for runs, conversion, replay and verification."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .commands import load_command_stream
from .config import LabConfig, default_state, load_config
from .conversion import conversion_artifact
from .engine import ExecutionEngine
from .errors import LabError
from .injections import run_injections
from .loaders import load_all_sources, read_source_bundle
from .util import read_json, write_json
from .verification import run_verification


def _summary(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "global_version": report.get("global_version"),
        "exchange_version": report.get("exchange_version"),
        "completed_layers": report.get("completed_layers"),
        "failures": len(report.get("failures", [])),
        "failure_codes": sorted({item["code"] for item in report.get("failures", [])}),
        "cache": {
            stage: {
                "reused": len(values.get("reused_rows", [])),
                "recomputed": len(values.get("recomputed_rows", [])),
            }
            for stage, values in report.get("cache_plan", {}).items()
        },
        "consumed_budget": report.get("schedule", {}).get("consumed_budget"),
        "pending_rows": len(report.get("schedule", {}).get("pending_queue", [])),
        "snapshot": report.get("snapshot"),
    }


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _load_engine(args: argparse.Namespace) -> tuple[LabConfig, ExecutionEngine]:
    project_root = Path(args.project_root).resolve() if args.project_root else None
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root or Path.cwd().resolve()) / config_path
    config = load_config(config_path, project_root)
    runtime = Path(args.runtime).resolve() if getattr(args, "runtime", None) else config.runtime_dir
    return config, ExecutionEngine(config, runtime_dir=runtime)


def _command_from_args(value: str) -> Dict[str, Any]:
    if value.startswith("@"):
        payload = read_json(Path(value[1:]))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise LabError("E_CONFIG", "command must be a JSON object", stage="command")
    return payload


def cmd_run(args: argparse.Namespace) -> int:
    _config, engine = _load_engine(args)
    state = engine.load_state()
    engine.save_state(state)
    report = engine.run(state, mode=args.mode)
    if args.json:
        _print_json(report)
    else:
        _print_json(_summary(report))
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    _config, engine = _load_engine(args)
    command = _command_from_args(args.command)
    report = engine.apply_and_run(command, mode=args.mode)
    if args.json:
        _print_json(report)
    else:
        _print_json(_summary(report))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    config, engine = _load_engine(args)
    if args.reset:
        if engine.runtime_dir.exists():
            shutil.rmtree(engine.runtime_dir)
        engine = ExecutionEngine(config, runtime_dir=engine.runtime_dir)
    state = default_state(config) if args.reset else engine.load_state()
    engine.save_state(state)
    base_report = engine.run(state, mode="incremental")
    results = [
        {
            "command": "<initial>",
            "summary": _summary(base_report),
        }
    ]
    state = engine.load_state()
    for command in load_command_stream(args.commands or config.paths["commands"]):
        state = engine.apply_and_run(command, mode="incremental")
        results.append(
            {
                "command": command.get("id", command.get("type")),
                "summary": _summary(state),
            }
        )
    if args.json:
        _print_json(results)
    else:
        print(
            json.dumps(
                {
                    "commands": len(results) - 1,
                    "final": results[-1]["summary"],
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    config, _engine = _load_engine(args)
    loaded = load_all_sources(config, read_source_bundle(config))
    target = args.to
    source_rows = loaded.weights_v1 if target == "v2" else loaded.weights_v2
    records = []
    failures = list(loaded.failures)
    for key in sorted(source_rows):
        try:
            artifact = conversion_artifact(config, source_rows[key], target)
        except LabError as exc:
            failures.append(exc.failure())
            continue
        records.append(artifact)
    output_path = Path(args.output).resolve()
    sidecar_path = Path(args.sidecar).resolve() if args.sidecar else output_path.with_suffix(
        output_path.suffix + ".sidecar.jsonl"
    )
    if target == "v2":
        lines = []
        for artifact in records:
            target_row = artifact["target"]
            lines.append(
                json.dumps(
                    {
                        "layer": target_row["layer"],
                        "row": target_row["row"],
                        "scale": target_row["scale"],
                        "bias": target_row["bias"],
                        "scheme": target_row["scheme"],
                        "precision": target_row["precision"],
                        "precision_tag": target_row["raw"].get("precision_tag"),
                        "sidecar": target_row["sidecar"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    else:
        lines = []
        for artifact in records:
            encoded = artifact["target"]["raw"].get("encoded_line")
            if not isinstance(encoded, str):
                raise LabError(
                    "E_FIELD_TRUNCATED",
                    "conversion did not produce a fixed-width line",
                    stage="version_conversion",
                )
            lines.append(encoded)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    sidecar_lines = [
        {
            "layer": artifact["target"]["layer"],
            "row": artifact["target"]["row"],
            "direction": artifact["direction"],
            "sidecar": artifact["sidecar"],
            "artifact_fingerprint": artifact["artifact_fingerprint"],
        }
        for artifact in records
    ]
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            for item in sidecar_lines
        ),
        encoding="utf-8",
    )
    result = {
        "source_version": "v1" if target == "v2" else "v2",
        "target_version": target,
        "converted_rows": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "output": str(output_path),
        "sidecar": str(sidecar_path),
    }
    _print_json(result)
    return 0 if not failures else 2


def cmd_inject(args: argparse.Namespace) -> int:
    config, _engine = _load_engine(args)
    manifest = Path(args.injections).resolve() if args.injections else config.paths.get("injections")
    report = run_injections(config, manifest_path=manifest)
    if args.json:
        _print_json(report)
    else:
        _print_json(
            {
                "all_matched": report["all_matched"],
                "items": [
                    {
                        "id": item["id"],
                        "kind": item["kind"],
                        "expected_code": item["expected_code"],
                        "matched": item["matched"],
                        "hit_count": item["hit_count"],
                    }
                    for item in report["items"]
                ],
            }
        )
    return 0 if report["all_matched"] else 1


def cmd_verify(args: argparse.Namespace) -> int:
    config, _engine = _load_engine(args)
    report = run_verification(config)
    if args.json:
        _print_json(report)
    else:
        _print_json(
            {
                "all_passed": report["all_passed"],
                "summary": report["summary"],
                "checks": [
                    {"name": check["name"], "passed": check["passed"]}
                    for check in report["checks"]
                ],
            }
        )
    return 0 if report["all_passed"] else 1


def cmd_manifest(args: argparse.Namespace) -> int:
    _config, engine = _load_engine(args)
    manifest = engine.store.load_manifest()
    if manifest is not None:
        manifest.pop("_resolved_snapshot_path", None)
    _print_json(manifest)
    return 0 if manifest is not None else 2


def cmd_reset(args: argparse.Namespace) -> int:
    _config, engine = _load_engine(args)
    if engine.runtime_dir.exists():
        shutil.rmtree(engine.runtime_dir)
    print(f"removed runtime: {engine.runtime_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opexlab",
        description="Softmax/LayerNorm format interoperability and fixed-point lab",
    )
    parser.add_argument(
        "--config",
        default="samples/config.json",
        help="path to config JSON (default: samples/config.json)",
    )
    parser.add_argument("--project-root", default=None, help="project root for relative paths")
    parser.add_argument("--runtime", default=None, help="override runtime directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the pipeline")
    run.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    run.add_argument("--json", action="store_true", help="print the full report JSON")
    run.set_defaults(func=cmd_run)

    apply_parser = subparsers.add_parser("apply", help="apply one editing command, then run")
    apply_parser.add_argument("--command", required=True, help="JSON text or @path-to-json")
    apply_parser.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    apply_parser.add_argument("--json", action="store_true")
    apply_parser.set_defaults(func=cmd_apply)

    replay = subparsers.add_parser("replay", help="replay a command stream")
    replay.add_argument("--commands", default=None, help="command JSONL path")
    replay.add_argument("--reset", action="store_true", help="remove runtime before replay")
    replay.add_argument("--json", action="store_true")
    replay.set_defaults(func=cmd_replay)

    convert = subparsers.add_parser("convert", help="convert weight files between v1 and v2")
    convert.add_argument("--to", choices=("v1", "v2"), required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--sidecar", default=None)
    convert.set_defaults(func=cmd_convert)

    inject = subparsers.add_parser("inject", help="run isolated fault injections")
    inject.add_argument("--injections", default=None)
    inject.add_argument("--json", action="store_true")
    inject.set_defaults(func=cmd_inject)

    verify = subparsers.add_parser("verify", help="run the full local acceptance suite")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=cmd_verify)

    manifest = subparsers.add_parser("manifest", help="print the current committed snapshot manifest")
    manifest.set_defaults(func=cmd_manifest)

    reset = subparsers.add_parser("reset", help="remove the configured runtime directory")
    reset.set_defaults(func=cmd_reset)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except LabError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"E_IO: {exc}", file=sys.stderr)
        return 2
