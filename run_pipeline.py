#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from pipeline.config import load_config
from pipeline.generator import generate_sample
from pipeline.runner import PipelineRunner


def _print_summary(report: dict) -> None:
    summary = {
        "status": report.get("status"),
        "exit_code": report.get("run", {}).get("exit_code"),
        "report": report.get("run", {}).get("report_path"),
        "queries": len(report.get("queries", [])),
        "failures": len(report.get("failures", [])),
        "snapshot_version": report.get("snapshot", {}).get("version"),
        "determinism_sha256": report.get("determinism", {}).get("sha256"),
        "metrics": report.get("evaluation", {}).get("metrics"),
        "budget": {
            "global": report.get("budget", {}).get("global_budget"),
            "consumed": report.get("budget", {}).get("consumed_total"),
            "query_budget_exhausted": report.get("budget", {}).get("query_budget_exhausted"),
            "budget_exhausted": report.get("budget", {}).get("budget_exhausted"),
        },
        "fatal_errors": report.get("fatal_errors"),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


def command_generate(args: argparse.Namespace) -> int:
    result = generate_sample(
        args.output,
        seed=args.seed,
        documents=args.documents,
        queries=args.queries,
        zh_ratio=args.zh_ratio,
        mixed_ratio=args.mixed_ratio,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    sample_dir = Path(args.sample_dir)
    state_dir = Path(args.state_dir)
    report_path = Path(args.report)
    stop_event = threading.Event()
    signal_state = {"reason": "cooperative_signal"}
    runner_holder: dict[str, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        signal_state["reason"] = f"signal:{signal.Signals(signum).name}"
        stop_event.set()
        if runner_holder.get("runner") is not None:
            runner_holder["runner"].stop_reason = signal_state["reason"]

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        except (ValueError, OSError):
            pass
    try:
        runner = PipelineRunner.from_sample(
            sample_dir=sample_dir,
            state_dir=state_dir,
            report_path=report_path,
            config_path=args.config,
            injections_path=args.injections,
            fresh=args.fresh,
            stop_after_batches=args.stop_after_batches,
            stop_event=stop_event,
            stop_reason=signal_state["reason"],
            max_workers=args.threads,
            global_budget=args.global_budget,
            per_query_quota=args.per_query_quota,
        )
        runner_holder["runner"] = runner
        if stop_event.is_set():
            runner.stop_reason = signal_state["reason"]
        report = runner.run()
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    _print_summary(report)
    return int(report.get("run", {}).get("exit_code", 2))


def _comparable_sections(report: dict) -> dict:
    return {
        "determinism": report.get("determinism"),
        "queries": report.get("queries"),
        "budget": report.get("budget"),
        "evaluation": report.get("evaluation"),
        "failures": report.get("failures"),
        "injections": report.get("injections"),
        "snapshot_version": report.get("snapshot", {}).get("version"),
        "completed_batches": report.get("snapshot", {}).get("completed_batches"),
    }


def command_verify(args: argparse.Namespace) -> int:
    sample_dir = Path(args.sample_dir)
    config, _ = load_config(Path(args.config) if args.config else sample_dir / "config.json")
    configured_workers = int(config["threads"]["max_workers"])
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    Path(args.temp_parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pipeline-verify-", dir=args.temp_parent) as temporary:
        temp = Path(temporary)

        serial = PipelineRunner.from_sample(
            sample_dir=sample_dir,
            state_dir=temp / "state-serial",
            report_path=temp / "serial.json",
            config_path=args.config,
            fresh=True,
            max_workers=1,
        ).run()
        parallel = PipelineRunner.from_sample(
            sample_dir=sample_dir,
            state_dir=temp / "state-parallel",
            report_path=temp / "parallel.json",
            config_path=args.config,
            fresh=True,
            max_workers=configured_workers,
        ).run()
        checks["serial_parallel_equal"] = _comparable_sections(serial) == _comparable_sections(parallel)
        checks["serial_worker_bound"] = serial["run"]["observed_max_active_workers"] <= 1
        checks["parallel_worker_bound"] = parallel["run"]["observed_max_active_workers"] <= configured_workers
        details["serial_parallel"] = {
            "serial_hash": serial.get("determinism", {}).get("sha256"),
            "parallel_hash": parallel.get("determinism", {}).get("sha256"),
            "configured_workers": configured_workers,
            "serial_observed_workers": serial["run"]["observed_max_active_workers"],
            "parallel_observed_workers": parallel["run"]["observed_max_active_workers"],
            "query_count": len(parallel.get("queries", [])),
        }

        full = PipelineRunner.from_sample(
            sample_dir=sample_dir,
            state_dir=temp / "state-full",
            report_path=temp / "full.json",
            config_path=args.config,
            fresh=True,
            max_workers=configured_workers,
        ).run()
        interrupted = PipelineRunner.from_sample(
            sample_dir=sample_dir,
            state_dir=temp / "state-incremental",
            report_path=temp / "incremental-partial.json",
            config_path=args.config,
            fresh=True,
            max_workers=configured_workers,
            stop_after_batches=5,
        ).run()
        resumed = PipelineRunner.from_sample(
            sample_dir=sample_dir,
            state_dir=temp / "state-incremental",
            report_path=temp / "incremental-final.json",
            config_path=args.config,
            fresh=False,
            max_workers=configured_workers,
        ).run()
        checks["cooperative_stop_exit_zero"] = interrupted.get("status") == "interrupted" and interrupted.get("run", {}).get("exit_code") == 0
        checks["incremental_resume_complete"] = resumed.get("status") == "complete"
        checks["full_incremental_equal"] = _comparable_sections(full) == _comparable_sections(resumed)
        details["incremental"] = {
            "full_hash": full.get("determinism", {}).get("sha256"),
            "resumed_hash": resumed.get("determinism", {}).get("sha256"),
            "partial_snapshot_version": interrupted.get("snapshot", {}).get("version"),
            "final_snapshot_version": resumed.get("snapshot", {}).get("version"),
            "commits_during_resume": resumed.get("run", {}).get("commits_this_run"),
        }

        budget = PipelineRunner.from_sample(
            sample_dir=sample_dir,
            state_dir=temp / "state-budget",
            report_path=temp / "budget.json",
            config_path=args.config,
            fresh=True,
            max_workers=configured_workers,
            global_budget=15,
            per_query_quota=5,
        ).run()
        budget_sequence = budget.get("budget", {}).get("sequence", [])
        checks["budget_cap_respected"] = budget.get("budget", {}).get("consumed_total", 0) <= 15
        checks["budget_exhaustion_observed"] = any(
            item.get("code") in {"E_QUERY_BUDGET_EXHAUSTED", "E_BUDGET_EXHAUSTED"} for item in budget_sequence
        )
        checks["budget_sequence_stable"] = [item["query_id"] for item in budget_sequence] == sorted(
            item["query_id"] for item in budget_sequence
        )
        details["budget"] = {
            "global": budget.get("budget", {}).get("global_budget"),
            "consumed": budget.get("budget", {}).get("consumed_total"),
            "allocated": budget.get("budget", {}).get("allocated_total"),
            "query_exhausted": budget.get("budget", {}).get("query_budget_exhausted"),
            "global_exhausted": budget.get("budget", {}).get("budget_exhausted"),
        }

        injection_report = PipelineRunner.from_sample(
            sample_dir=sample_dir,
            state_dir=temp / "state-injection",
            report_path=temp / "injection.json",
            config_path=args.config,
            injections_path=args.injections,
            fresh=True,
            max_workers=configured_workers,
        ).run()
        injection_items = injection_report.get("injections", [])
        checks["injection_all_hit"] = bool(injection_items) and all(item.get("hit") for item in injection_items)
        checks["injection_isolated"] = len(injection_report.get("queries", [])) == len(full.get("queries", [])) and any(
            item.get("status") == "ok" for item in injection_report.get("queries", [])
        )
        checks["cross_language_bridge_observed"] = any(
            query.get("translation_expansions")
            for query in parallel.get("queries", [])
        )
        details["injections"] = [
            {
                "injection_id": item.get("injection_id"),
                "kind": item.get("kind"),
                "hit": item.get("hit"),
                "hit_stage": item.get("hit_stage"),
                "code": (item.get("failure") or {}).get("code"),
            }
            for item in injection_items
        ]
        details["bridge_expansion_count"] = sum(
            len(query.get("translation_expansions", [])) for query in parallel.get("queries", [])
        )
        details["report_statuses"] = {
            "serial": serial.get("status"),
            "parallel": parallel.get("status"),
            "full": full.get("status"),
            "interrupted": interrupted.get("status"),
            "resumed": resumed.get("status"),
            "budget": budget.get("status"),
            "injection": injection_report.get("status"),
        }

    verification = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "details": details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if verification["status"] == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic cross-lingual retrieval pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a deterministic local sample")
    generate.add_argument("--output", default="samples/default")
    generate.add_argument("--seed", type=int, default=20261120)
    generate.add_argument("--documents", type=int, default=48)
    generate.add_argument("--queries", type=int, default=20)
    generate.add_argument("--zh-ratio", type=float, default=0.5)
    generate.add_argument("--mixed-ratio", type=float, default=0.25)
    generate.set_defaults(func=command_generate)

    run = subparsers.add_parser("run", help="run or resume the pipeline")
    run.add_argument("--sample-dir", default="samples/default")
    run.add_argument("--config")
    run.add_argument("--injections")
    run.add_argument("--state-dir", default="artifacts/state")
    run.add_argument("--report", default="artifacts/report.json")
    run.add_argument("--fresh", action="store_true")
    run.add_argument("--stop-after-batches", type=int, default=None, help="test hook for cooperative stop semantics")
    run.add_argument("--threads", type=int, default=None)
    run.add_argument("--global-budget", type=int, default=None)
    run.add_argument("--per-query-quota", type=int, default=None)
    run.set_defaults(func=command_run)

    verify = subparsers.add_parser("verify", help="run acceptance comparisons")
    verify.add_argument("--sample-dir", default="samples/default")
    verify.add_argument("--config")
    verify.add_argument("--injections", default="samples/default/injections.json")
    verify.add_argument("--temp-parent", default="artifacts")
    verify.add_argument("--output", default="artifacts/verification.json")
    verify.set_defaults(func=command_verify)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
