"""Command line entry point for local deterministic NER review runs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from .codes import Code
from .config import RuntimeConfig
from .coordinator import CommandCoordinator
from .diagnostics import run_injection_suite
from .engine import IncrementalEngine
from .io_utils import write_json
from .snapshot import SnapshotStore
from .synth import write_demo_bundle
from .validation import load_package, parse_command_stream, parse_corpus, parse_lexicon


def _load_inputs(data_dir: str | Path) -> tuple[dict[str, Any], Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data_dir = Path(data_dir)
    package_result = load_package(data_dir / "model_params.json")
    if not package_result["ok"]:
        raise SystemExit(json.dumps({"ok": False, "failures": package_result["failures"]}, ensure_ascii=False, indent=2))
    corpus = parse_corpus(data_dir / "corpus.jsonl")
    lexicon = parse_lexicon(data_dir / "lexicon.json")
    commands_path = data_dir / "command_stream.json"
    commands = parse_command_stream(commands_path) if commands_path.exists() else []
    return package_result["package"], RuntimeConfig(), corpus["samples"], lexicon["entries"], commands


def _validation_report(path: Path) -> int:
    result = parse_corpus(path)
    print(json.dumps({"sample_count": len(result["samples"]), "failures": result["failures"], "business_counts": result["business_counts"]}, ensure_ascii=False, indent=2))
    return 0 if not result["failures"] else 1


def _package_report(path: Path) -> int:
    result = load_package(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


def _run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    data_dir = Path(args.data)
    package, config, samples, lexicon, file_commands = _load_inputs(data_dir)
    commands = parse_command_stream(args.commands) if args.commands else file_commands
    workspace = Path(args.workspace)
    store = SnapshotStore(workspace)

    recovery: dict[str, Any]
    if args.full:
        engine = IncrementalEngine(package=package, config=config, samples=samples, lexicon=lexicon)
        recovery = {"recovery_action": "full_blank", "code": Code.OK}
        store.save(engine, phase="blank")
    else:
        engine, recovery = store.recover_or_create(package, config, samples, lexicon)

    applied = []
    run_ledger = {"reused": [], "recomputed": []}
    while engine.command_cursor < len(commands):
        command = commands[engine.command_cursor]
        command_result = engine.apply_command(command)
        applied.append(command_result)
        run_ledger = engine.run_incremental()
        snapshot = store.save(engine, phase="command-stable")
    if not applied:
        run_ledger = engine.run_incremental()
        snapshot = store.save(engine, phase="stable")
    result = engine.authoritative_result()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    ledger_path = Path(args.ledger) if args.ledger else workspace / "authoritative_ledger.json"
    write_json(ledger_path, result)
    failed_samples = [failure.get("sample_id") for failure in [*engine.corpus_failures, *engine.command_failures] if failure.get("sample_id")]
    report = {
        "ok": True,
        "workspace": str(workspace),
        "data_dir": str(data_dir),
        "recovery": recovery,
        "snapshot": {"snapshot_id": snapshot["snapshot_id"], "manifest": snapshot["manifest"], "checkpoint_per_command": True},
        "elapsed_ms": round(elapsed_ms, 3),
        "command_count": len(commands),
        "applied_command_results": applied,
        "business_code_counts": engine.business_counts(),
        "failed_sample_ids": failed_samples,
        "corpus_failure_count": len(engine.corpus_failures),
        "command_failure_count": len(engine.command_failures),
        "invalidation_log_count": len(engine.invalidation_log),
        "cumulative_counts": engine.counters,
        "last_run_reused_count": len(run_ledger["reused"]),
        "last_run_recomputed_count": len(run_ledger["recomputed"]),
        "last_run_reused": run_ledger["reused"],
        "last_run_recomputed": run_ledger["recomputed"],
        "summary": engine.summary(),
        "ledger_path": str(ledger_path),
    }
    report_path = Path(args.report) if args.report else workspace / "run_report.json"
    write_json(report_path, report)
    print(json.dumps({k: report[k] for k in ("ok", "recovery", "elapsed_ms", "business_code_counts", "cumulative_counts", "summary", "ledger_path")}, ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    return 0


def _inject(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    report = run_injection_suite(data_dir / "corpus.jsonl", data_dir / "model_params.json", args.manifest or data_dir / "injection_manifest.json", args.out)
    print(json.dumps({k: report[k] for k in ("ok", "injection_count", "passed_count", "business_code_counts")}, ensure_ascii=False, indent=2))
    print(f"report={Path(args.out) / 'injection_report.json'}")
    return 0 if report["ok"] else 1


def _budget_demo(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    package, config, samples, lexicon, commands = _load_inputs(data_dir)
    workspace = Path(args.workspace)
    store = SnapshotStore(workspace)
    engine, recovery = store.recover_or_create(package, config, samples, lexicon)
    engine.run_incremental()
    store.save(engine, phase="initial")
    coordinator = CommandCoordinator(engine, store, budget_ms=args.budget_ms, work_hook=lambda eng, cmd: time.sleep(args.work_ms / 1000.0))
    command = commands[args.command_index] if commands else {"op": "set_transition_weight", "from": "O", "to": "O", "value": 0.7}
    immediate = coordinator.submit(command, budget_ms=args.budget_ms)
    final = coordinator.wait_for_latest(timeout=max(2.0, args.work_ms / 1000.0 + 2.0))
    coordinator.shutdown()
    output = {"recovery": recovery, "immediate_response": immediate, "final_response": final}
    out_path = Path(args.report)
    write_json(out_path, output)
    print(json.dumps({"immediate_status": immediate.get("status"), "immediate_code": immediate.get("code"), "stale_generation": immediate.get("stale_generation"), "final_status": final.get("status")}, ensure_ascii=False, indent=2))
    print(f"report={out_path}")
    return 0 if immediate.get("code") == Code.STALE_SUMMARY and final.get("status") == "fresh" else 1


def _kill_target(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    package, config, samples, lexicon, commands = _load_inputs(data_dir)
    store = SnapshotStore(args.workspace)
    engine, recovery = store.recover_or_create(package, config, samples, lexicon)
    if engine.command_cursor == 0:
        engine.run_incremental()
        store.save(engine, phase="before-kill-demo")
    index = min(args.command_index, len(commands) - 1)
    command = commands[index]
    result = engine.apply_command(command)
    engine.run_incremental()
    if args.mode == "after-save":
        store.save(engine, phase="just-before-sigkill")
    marker = Path(args.marker) if args.marker else Path(args.workspace) / "kill_marker.json"
    write_json(marker, {"pid": os.getpid(), "mode": args.mode, "seq": engine.command_cursor, "affected": result.get("affected_samples", [])})
    os.kill(os.getpid(), signal.SIGKILL)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ner_review", description="Deterministic Chinese NER review ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed-demo", help="generate deterministic local sample bundle")
    seed.add_argument("--out", default="examples/generated")
    seed.add_argument("--samples", type=int, default=24)
    seed.set_defaults(func=lambda args: (write_demo_bundle(args.out, RuntimeConfig(corpus_size=args.samples)), print(f"generated={args.out}"))[1])

    validate_corpus = sub.add_parser("validate-corpus")
    validate_corpus.add_argument("--file", required=True)
    validate_corpus.set_defaults(func=lambda args: _validation_report(Path(args.file)))

    validate_package = sub.add_parser("validate-package")
    validate_package.add_argument("--file", required=True)
    validate_package.set_defaults(func=lambda args: _package_report(Path(args.file)))

    run = sub.add_parser("run")
    run.add_argument("--data", default="examples/generated")
    run.add_argument("--workspace", default="state/demo")
    run.add_argument("--commands")
    run.add_argument("--report")
    run.add_argument("--ledger")
    run.add_argument("--full", action="store_true")
    run.set_defaults(func=_run)

    inject = sub.add_parser("inject")
    inject.add_argument("--data", default="examples/generated")
    inject.add_argument("--manifest")
    inject.add_argument("--out", default="reports/injections")
    inject.set_defaults(func=_inject)

    budget = sub.add_parser("budget-demo")
    budget.add_argument("--data", default="examples/generated")
    budget.add_argument("--workspace", default="state/budget-demo")
    budget.add_argument("--report", default="reports/budget_demo.json")
    budget.add_argument("--budget-ms", type=int, default=5)
    budget.add_argument("--work-ms", type=int, default=120)
    budget.add_argument("--command-index", type=int, default=3)
    budget.set_defaults(func=_budget_demo)

    kill = sub.add_parser("kill-target")
    kill.add_argument("--data", default="examples/generated")
    kill.add_argument("--workspace", default="state/kill-demo")
    kill.add_argument("--marker")
    kill.add_argument("--mode", choices=["before-save", "after-save"], default="before-save")
    kill.add_argument("--command-index", type=int, default=0)
    kill.set_defaults(func=_kill_target)

    gui = sub.add_parser("gui", help="launch optional Tkinter review console")
    gui.add_argument("--data", default="examples/generated")
    gui.add_argument("--workspace", default="state/gui")

    def _gui(args: argparse.Namespace) -> int:
        from .gui import launch

        return launch(args.data, args.workspace)

    gui.set_defaults(func=_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
