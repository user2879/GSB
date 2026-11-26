from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .commands import apply_command, load_commands
from .config import load_config
from .engine import AuditEngine
from .errors import AuditError, ConfigInvalid
from .injections import InjectionHub, load_injections
from .samples import generate_samples, load_samples, write_jsonl
from .snapshot import SnapshotStore


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_fixture_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dimension": 8,
        "seed": 393,
        "train_length": 12,
        "target_length": 48,
        "fixed_point": {"frac_bits": 20, "quant_level": 1, "level_shifts": [0, 3, 6, 9]},
        "tolerances": {
            "position": 2.0e-3,
            "rotation": 8.0e-3,
            "dot": 2.0e-2,
            "normalization": 5.0e-3,
        },
        "budget": {"batch_point_budget": 96, "sample_quota_points": 64},
        "concurrency": {"workers": 2},
        "snapshot_dir": "artifacts/snapshots",
        "active_scheme": "rope_base",
        "schemes": [
            {
                "name": "rope_base",
                "base_freq": 10000.0,
                "angle_scale": 1.0,
                "block_size": 2,
                "correction_level": 0,
            },
            {
                "name": "rope_flash",
                "base_freq": 50000.0,
                "angle_scale": 0.75,
                "block_size": 2,
                "correction_level": 2,
            },
            {
                "name": "rope_block4",
                "base_freq": 10000.0,
                "angle_scale": 1.15,
                "block_size": 4,
                "correction_level": 1,
            },
        ],
    }


def build_fixture_commands(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replacement = json.loads(json.dumps(samples[1]))
    replacement["train_length"] = 16
    replacement["query_vector"][0] = round(float(replacement["query_vector"][0]) + 0.125, 6)
    replacement["key_vectors"][2][3] = round(float(replacement["key_vectors"][2][3]) - 0.0625, 6)
    return [
        {"op": "switch_scheme", "line": 1, "name": "rope_flash"},
        {"op": "switch_scheme", "line": 2, "name": "rope_base"},
        {"op": "update_scheme", "line": 3, "name": "rope_base", "params": {"base_freq": 12000.0}},
        {"op": "update_scheme", "line": 4, "name": "rope_flash", "params": {"angle_scale": 0.825}},
        {
            "op": "add_scheme",
            "line": 5,
            "scheme": {
                "name": "rope_added",
                "base_freq": 24000.0,
                "angle_scale": 1.05,
                "block_size": 4,
                "correction_level": 2,
            },
        },
        {"op": "set_lengths", "line": 6, "train_length": 16, "target_length": 64},
        {"op": "replace_sample", "line": 7, "sample": replacement},
        {"op": "delete_sample", "line": 8, "id": "s006"},
        {"op": "set_fixed_point", "line": 9, "frac_bits": 12, "quant_level": 3},
        {"op": "set_fixed_point", "line": 10, "frac_bits": 20, "quant_level": 1},
    ]


def cmd_generate(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = build_fixture_config()
    config_path = out / "config.json"
    write_json(config_path, config)
    parsed_config, _ = load_config(str(config_path))
    samples = generate_samples(parsed_config, count=6)
    sample_path = out / "samples.jsonl"
    write_jsonl(sample_path, samples)
    commands = build_fixture_commands(samples)
    command_path = out / "commands.jsonl"
    write_jsonl(command_path, commands)
    injections = {
        "injections": [
            {
                "id": "INJ-ROT-NAN-001",
                "sample_id": "s004",
                "scheme_id": "rope_base",
                "stage": "rotation",
                "fault_type": "nonfinite_feature",
                "code": "E_INJECTED_NONFINITE",
                "message": "injected NaN into rotation query feature",
            }
        ]
    }
    injection_path = out / "injections.json"
    write_json(injection_path, injections)
    print(json.dumps({"generated": True, "config": str(config_path), "samples": str(sample_path), "commands": str(command_path), "injections": str(injection_path)}))
    return 0


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config, _ = load_config(args.config)
    samples = load_samples(args.samples, config["dimension"])
    injections = load_injections(getattr(args, "injections", None))
    return config, samples, injections


def run_until_drained(
    engine: AuditEngine,
    drain: bool,
    snapshot_store: SnapshotStore | None = None,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    maximum_batches = 200
    for _ in range(maximum_batches):
        report = engine.run_batch(workers=workers)
        reports.append(report)
        if snapshot_store is not None:
            snapshot_store.commit(engine, report)
        if not drain or not report["budget"]["pending_sample_ids"]:
            break
    return reports


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        config, _ = load_config(args.config)
        samples = load_samples(args.samples, config["dimension"])
        commands = load_commands(args.commands) if args.commands else []
        for command in commands:
            apply_command(config, samples, command)
        injections = load_injections(args.injections) if args.injections else []
    except AuditError as error:
        print(json.dumps({"valid": False, "error": error.to_dict()}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"valid": True, "schemes": len(config["schemes"]), "samples": len(samples), "commands": len(commands), "injections": len(injections)}, sort_keys=True))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config, samples, injections = load_inputs(args)
    hub = InjectionHub(injections)
    store = SnapshotStore(args.snapshot_dir or config["snapshot_dir"]) if not args.no_snapshot else None
    engine = AuditEngine(config, samples, hub)
    resumed = False
    if args.resume and store is not None:
        state = store.load_state(engine.config_hash)
        if state is not None:
            engine.import_state(state)
            engine.hub = hub
            resumed = True
    if args.batch_budget is not None:
        engine.config["budget"]["batch_point_budget"] = int(args.batch_budget)
    if args.quota is not None:
        engine.config["budget"]["sample_quota_points"] = int(args.quota)
    reports = run_until_drained(engine, args.drain, store, args.workers)
    final_report = reports[-1]
    output = {
        "resumed_snapshot": resumed,
        "batch_count": len(reports),
        "final_report": final_report,
        "snapshot_recovery": store.recover(engine.config_hash) if store is not None else None,
    }
    if args.output:
        write_json(args.output, output)
    summary = {
        "status": "ok",
        "output": args.output,
        "batch_count": len(reports),
        "pending": final_report["budget"]["pending_sample_ids"],
        "failures": len(final_report["failures"]),
        "cache_entries": len(final_report["cache_index"]),
        "injection_hits": len(final_report["injection_hits"]),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _ledger_counts(ledger: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for entry in ledger:
        stage = entry["stage"]
        counts.setdefault(stage, {"hit": 0, "miss": 0})
        counts[stage][entry["status"]] = counts[stage].get(entry["status"], 0) + 1
    return counts


def cmd_replay(args: argparse.Namespace) -> int:
    config, samples, injection_rows = load_inputs(args)
    commands = load_commands(args.commands)
    hub = InjectionHub(injection_rows)
    store = SnapshotStore(args.snapshot_dir or config["snapshot_dir"]) if not args.no_snapshot else None

    incremental = AuditEngine(config, samples, hub)
    step_reports: list[dict[str, Any]] = []
    current_config = config
    current_samples = samples
    for command in commands:
        current_config, current_samples, edit = apply_command(current_config, current_samples, command)
        incremental.replace_state(current_config, current_samples)
        reports = run_until_drained(incremental, True, None, args.workers)
        report = reports[-1]
        if store is not None:
            store.commit(incremental, report)
        step_ledger = [entry for batch_report in reports for entry in batch_report["run_ledger"]]
        step_reports.append(
            {
                "step": len(step_reports) + 1,
                "command": {key: value for key, value in command.items() if not key.startswith("_")},
                "edit": edit,
                "batch_count": len(reports),
                "ledger_counts": _ledger_counts(step_ledger),
                "batch_ledger_counts": [_ledger_counts(batch_report["run_ledger"]) for batch_report in reports],
                "budget": report["budget"],
                "failures": report["failures"],
                "first_breaches": {
                    name: stats["first_breaches"] for name, stats in report["statistics"].items()
                },
                "run_ledger": step_ledger,
            }
        )

    final_config = incremental.config
    final_samples = incremental.samples
    serial_engine = AuditEngine(deepcopy_json(final_config), deepcopy_json(final_samples))
    serial_reports = run_until_drained(serial_engine, True, None, 1)
    serial_payload = serial_engine.comparison_payload(serial_reports[-1])

    threaded_engine = AuditEngine(deepcopy_json(final_config), deepcopy_json(final_samples))
    threaded_reports = run_until_drained(threaded_engine, True, None, max(2, int(args.workers or final_config["concurrency"]["workers"])))
    threaded_payload = threaded_engine.comparison_payload(threaded_reports[-1])

    incremental_payload = incremental.comparison_payload(incremental.last_report)
    incremental_vs_full = canonical(incremental_payload) == canonical(serial_payload)
    serial_vs_threaded = canonical(serial_payload) == canonical(threaded_payload)

    output = {
        "equivalence": {
            "incremental_vs_blank_full_serial": incremental_vs_full,
            "serial_vs_threaded": serial_vs_threaded,
        },
        "step_count": len(step_reports),
        "steps": step_reports,
        "incremental_final": {
            "report": incremental.last_report,
        },
        "snapshot_recovery": store.recover(incremental.config_hash) if store is not None else None,
    }
    if args.output:
        write_json(args.output, output)
    print(
        json.dumps(
            {
                "status": "ok" if incremental_vs_full and serial_vs_threaded else "mismatch",
                "output": args.output,
                "steps": len(step_reports),
                "incremental_vs_full": incremental_vs_full,
                "serial_vs_threaded": serial_vs_threaded,
            },
            sort_keys=True,
        )
    )
    return 0 if incremental_vs_full and serial_vs_threaded else 1


def deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rope_audit", description="Fixed-point position-encoding extrapolation audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local fixtures")
    generate.add_argument("--out-dir", default="examples")
    generate.set_defaults(func=cmd_generate)

    validate = subparsers.add_parser("validate", help="validate config, samples and commands")
    validate.add_argument("--config", required=True)
    validate.add_argument("--samples", required=True)
    validate.add_argument("--commands")
    validate.add_argument("--injections")
    validate.set_defaults(func=cmd_validate)

    run = subparsers.add_parser("run", help="run one or more budget batches")
    run.add_argument("--config", required=True)
    run.add_argument("--samples", required=True)
    run.add_argument("--injections")
    run.add_argument("--output")
    run.add_argument("--snapshot-dir")
    run.add_argument("--workers", type=int)
    run.add_argument("--batch-budget", type=int)
    run.add_argument("--quota", type=int)
    run.add_argument("--drain", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--no-snapshot", action="store_true")
    run.set_defaults(func=cmd_run)

    replay = subparsers.add_parser("replay", help="apply command stream incrementally and verify rebuild equivalence")
    replay.add_argument("--config", required=True)
    replay.add_argument("--samples", required=True)
    replay.add_argument("--commands", required=True)
    replay.add_argument("--injections")
    replay.add_argument("--output")
    replay.add_argument("--snapshot-dir")
    replay.add_argument("--workers", type=int, default=2)
    replay.add_argument("--no-snapshot", action="store_true")
    replay.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AuditError as error:
        print(json.dumps({"status": "error", "error": error.to_dict()}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
