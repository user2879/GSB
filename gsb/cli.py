from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_config, validate_config
from .edits import apply_commands, load_command_stream, write_batch_files
from .engine import run_pipeline
from .errors import (
    E_BUDGET_PENDING,
    E_DOMAIN_QUOTA_EXCEEDED,
    E_INPUT_INVALID,
    PipelineError,
)
from .fixtures import write_fixture_tree
from .samples import load_sample_set
from .store import SnapshotStore


EXIT_CODES = {
    "complete": 0,
    E_INPUT_INVALID: 2,
    E_DOMAIN_QUOTA_EXCEEDED: 3,
    E_BUDGET_PENDING: 4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsb", description="Deterministic ledger pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic fixtures")
    generate.add_argument("--root", default=".")

    run = subparsers.add_parser("run", help="run pipeline")
    run.add_argument("--config", required=True)
    run.add_argument("--samples", required=True)
    run.add_argument("--state", required=True)
    run.add_argument("--budget", type=int)
    run.add_argument("--workers", type=int)
    run.add_argument("--crash-before-commit", action="store_true")
    run.add_argument("--output", help="write concise JSON result to this path")

    edits = subparsers.add_parser("apply-edits", help="apply command stream to a fixture copy")
    edits.add_argument("--source-root", required=True)
    edits.add_argument("--output-dir", required=True)
    edits.add_argument("--commands", required=True, help="command_stream.jsonl path")
    edits.add_argument("--through", type=int, help="apply only first N commands")

    closeout = subparsers.add_parser("write-closeout", help="write a graceful closeout report")
    closeout.add_argument("--state", required=True)
    closeout.add_argument("--reason", default="graceful_shutdown")

    ui = subparsers.add_parser("ui", help="launch optional Tkinter diagnostic browser")
    ui.add_argument("--state", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            paths = write_fixture_tree(Path(args.root))
            print(json.dumps({"ok": True, "generated": {k: str(v) for k, v in paths.items()}}, indent=2))
            return 0
        if args.command == "run":
            return _run(args)
        if args.command == "apply-edits":
            return _apply_edits(args)
        if args.command == "write-closeout":
            store = SnapshotStore(args.state)
            path = store.write_closeout_report({"ok": True, "reason": args.reason, "handled_exit": True})
            print(json.dumps({"ok": True, "closeout_report": str(path)}, indent=2))
            return 0
        if args.command == "ui":
            from .diagnostic_ui import launch

            return launch(args.state)
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return EXIT_CODES.get(exc.code, 1)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports unexpected failures
        payload = {"ok": False, "error_code": "E_INTERNAL", "message": str(exc)}
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.command}")
    return 2


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    sample_set = load_sample_set(args.samples, config)
    result = run_pipeline(
        config,
        sample_set,
        args.state,
        budget=args.budget,
        workers=args.workers,
        crash_before_commit=args.crash_before_commit,
    )
    concise = _concise_result(result)
    text = json.dumps(concise, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return EXIT_CODES.get(result["manifest"]["status"], 0 if result["manifest"]["status"] == "complete" else 1)


def _apply_edits(args: argparse.Namespace) -> int:
    source = Path(args.source_root)
    output = Path(args.output_dir)
    raw_config = json.loads((source / "configs" / "pipeline.json").read_text(encoding="utf-8"))
    sample_dir = source / "data" / "batches"
    batch_files: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(sample_dir.glob("*.jsonl")):
        batch_files[path.name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    commands = load_command_stream(args.commands)
    if args.through is not None:
        commands = commands[: args.through]
    new_config, new_batches = apply_commands(raw_config, batch_files, commands)
    validate_config(new_config)
    (output / "configs").mkdir(parents=True, exist_ok=True)
    (output / "data" / "batches").mkdir(parents=True, exist_ok=True)
    (output / "configs" / "pipeline.json").write_text(json.dumps(new_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_batch_files(output / "data" / "batches", new_batches)
    print(json.dumps({"ok": True, "output_dir": str(output), "applied": len(commands)}, indent=2))
    return 0


def _concise_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result["manifest"]["status"] == "complete",
        "status": result["manifest"]["status"],
        "business_code": result["manifest"].get("business_code"),
        "global_version": result["manifest"]["global_version"],
        "snapshot_dir": result.get("snapshot_dir"),
        "cleaning_count": result["cleaning"]["record_count"],
        "dedup_shard_count": len(result["dedup"]["shards"]),
        "sampling_count": len(result["sampling"]["selections"]) if result["sampling"] else 0,
        "ledger_entry_count": len(result["ledger"]["entries"]),
        "queue_next": result["queue"].get("next_priority_subject"),
        "budget": result["budget"],
        "error": result.get("error"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
