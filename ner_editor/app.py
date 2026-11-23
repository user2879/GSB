"""Command-line application orchestration."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

from .constants import E_LATENCY_BUDGET_EXCEEDED
from .engine import LocalNerEngine
from .loader import (
    config_fingerprint,
    corpus_fingerprint,
    load_commands,
    load_config,
    load_corpus,
    load_model,
    vocabulary_fingerprint,
)
from .models import Failure
from .snapshot import SnapshotManager
from .util import atomic_write_text, digest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _deep_diff(left: Any, right: Any, path: str = "$", differences: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if differences is None:
        differences = []
    if len(differences) >= 25:
        return differences
    if type(left) is not type(right):
        differences.append({"path": path, "incremental": left, "full": right})
        return differences
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                differences.append({"path": f"{path}.{key}", "incremental": "<missing>", "full": right[key]})
            elif key not in right:
                differences.append({"path": f"{path}.{key}", "incremental": left[key], "full": "<missing>"})
            else:
                _deep_diff(left[key], right[key], f"{path}.{key}", differences)
            if len(differences) >= 25:
                break
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            differences.append({"path": f"{path}.length", "incremental": len(left), "full": len(right)})
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _deep_diff(left_item, right_item, f"{path}[{index}]", differences)
            if len(differences) >= 25:
                break
        return differences
    if left != right:
        differences.append({"path": path, "incremental": left, "full": right})
    return differences


def _verify_full(engine: LocalNerEngine) -> dict[str, Any]:
    full_engine = LocalNerEngine(
        dict(engine.samples),
        engine.model.clone(),
        engine.config.clone(),
    )
    try:
        full_engine.settle()
        incremental_projection = engine.business_projection()
        full_projection = full_engine.business_projection()
        differences = _deep_diff(incremental_projection, full_projection)
        return {
            "passed": not differences,
            "incremental_hash": digest(incremental_projection),
            "full_hash": digest(full_projection),
            "differences": differences,
            "comparison_fields": [
                "sample_order",
                "samples.entities",
                "samples.path_total",
                "samples.candidate_scores",
                "samples.boundary_records",
                "samples.repairs",
                "samples.emission_quantization",
                "samples.stage_versions",
                "samples.stage_cache_keys",
                "stage_failure_codes",
            ],
            "note": "reuse/recompute action status is diagnostic and excluded; content keys and versions are compared",
        }
    finally:
        full_engine.close()


def _fatal_report(failures: list[Failure], exit_code: int) -> dict[str, Any]:
    code_counts: dict[str, int] = {}
    for failure in failures:
        code_counts[failure.code] = code_counts.get(failure.code, 0) + 1
    return {
        "schema": "ner-editor-report-v1",
        "fatal": True,
        "exit_code": exit_code,
        "failures": [failure.as_dict() for failure in failures],
        "code_counts": dict(sorted(code_counts.items())),
        "samples": [],
        "command_responses": [],
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _resolve(path_value: str, *, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic local Chinese NER data/model editor",
    )
    parser.add_argument("--workspace", default=".", help="writable workspace root")
    parser.add_argument("--corpus", default=None, help="char-level corpus JSONL")
    parser.add_argument("--model", default=None, help="model parameter package JSON")
    parser.add_argument("--config", default=None, help="interaction config JSON")
    parser.add_argument("--commands", default=None, help="command stream JSON")
    parser.add_argument("--report", default=None, help="output report JSON path")
    parser.add_argument("--run-id", default="default", help="snapshot run id")
    parser.add_argument("--resume", action="store_true", help="resume from CURRENT snapshot")
    parser.add_argument("--no-snapshot", action="store_true", help="disable snapshot commits")
    parser.add_argument("--max-commands", type=int, default=None, help="stop after N commands this invocation")
    parser.add_argument("--budget-ms", type=float, default=None, help="override interaction latency budget")
    parser.add_argument("--verify-full", action="store_true", help="compare final state with a blank-cache full run")
    parser.add_argument("--quiet", action="store_true", help="suppress per-command JSONL on stdout")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    sample_root = PROJECT_ROOT / "samples"
    corpus_path = _resolve(args.corpus, base=workspace) if args.corpus else sample_root / "corpus.jsonl"
    model_path = _resolve(args.model, base=workspace) if args.model else sample_root / "model.json"
    config_path = _resolve(args.config, base=workspace) if args.config else sample_root / "config.json"
    commands_path = _resolve(args.commands, base=workspace) if args.commands else sample_root / "commands.json"
    report_path = _resolve(args.report, base=workspace) if args.report else workspace / "artifacts" / "report.json"

    samples, corpus_failures = load_corpus(corpus_path)
    model, model_failures = load_model(model_path)
    config, config_failures = load_config(config_path, base_dir=workspace)
    commands, command_load_failures = load_commands(commands_path)

    fatal_failures = list(model_failures) + list(config_failures)
    if model is None or config is None:
        report = _fatal_report(fatal_failures, 2)
        _write_report(report_path, report)
        if not args.quiet:
            print(_json_line({"event": "fatal", "report": str(report_path), "exit_code": 2}))
        return 2

    if args.budget_ms is not None:
        config.latency_budget_ms = max(0.0, float(args.budget_ms))

    snapshot_manager = None if args.no_snapshot else SnapshotManager(workspace, args.run_id)
    restored = None
    resume_failure = None
    if args.resume and snapshot_manager is not None:
        restored, resume_failure = snapshot_manager.load()

    if restored is not None:
        engine = LocalNerEngine(
            restored.samples,
            restored.model,
            restored.config,
            load_failures=restored.load_failures,
            command_failures=restored.command_failures,
            workspace=str(workspace),
            snapshot_manager=snapshot_manager,
        )
        engine.version = restored.engine_version
        engine.last_command_position = restored.last_command_position
        engine.processed_commands = restored.processed_commands
        engine.command_responses = list(restored.command_responses)
        engine._last_ready_summary = restored.last_ready_summary
        engine.restore_artifacts(restored.artifacts, restored.runtime_failures)
        known_failure_ids = {failure.identity() for failure in engine.command_failures}
        for failure in command_load_failures:
            if (
                (failure.command_position or 0) > restored.last_command_position
                and failure.identity() not in known_failure_ids
            ):
                engine.command_failures.append(failure)
                known_failure_ids.add(failure.identity())
        if args.budget_ms is not None:
            engine.config.latency_budget_ms = max(0.0, float(args.budget_ms))
    else:
        load_failures = list(corpus_failures) + list(command_load_failures)
        if resume_failure is not None:
            load_failures.append(resume_failure)
        engine = LocalNerEngine(
            samples,
            model,
            config,
            load_failures=load_failures,
            command_failures=command_load_failures,
            workspace=str(workspace),
            snapshot_manager=snapshot_manager,
        )
        # The loader already reports command parsing failures in engine.command_failures;
        # avoid duplication in the general failure list.
        engine.load_failures = list(corpus_failures)
        if resume_failure is not None:
            engine.load_failures.append(resume_failure)

    stop_signal: dict[str, int | None] = {"value": None}

    def signal_handler(signum: int, _frame: Any) -> None:
        stop_signal["value"] = signum

    old_handlers: dict[int, Any] = {}
    for signum in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if signum is None:
            continue
        try:
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, signal_handler)
        except (ValueError, OSError):
            pass

    max_commands = args.max_commands
    if max_commands is not None and max_commands < 0:
        max_commands = 0
    pending_commands = [
        command
        for command in commands
        if command.position > engine.last_command_position
    ]
    processed_this_run = 0
    interval = max(1, engine.config.snapshot_commit_interval_commands)
    try:
        for command in pending_commands:
            if max_commands is not None and processed_this_run >= max_commands:
                break
            if stop_signal["value"] is not None:
                break
            response = engine.apply_command(command)
            processed_this_run += 1
            if not args.quiet:
                print(_json_line(response), flush=True)
            needs_interval_snapshot = (
                response.get("accepted")
                and engine.processed_commands % interval == 0
                and snapshot_manager is not None
            )
            if needs_interval_snapshot:
                engine.settle()
                snapshot_manager.commit(
                    engine,
                    reason=f"interval after command {command.position}",
                )
            if command.command_type == "quit":
                break
            if stop_signal["value"] is not None:
                break

        engine.settle()
        equivalence = _verify_full(engine) if args.verify_full else None
        report = engine.full_report(equivalence=equivalence)
        snapshot_info = None
        if snapshot_manager is not None:
            snapshot_info = snapshot_manager.commit(
                engine,
                reason=(
                    f"graceful signal {stop_signal['value']}"
                    if stop_signal["value"] is not None
                    else "final"
                ),
            )
        report["snapshot"] = snapshot_info
        report["resumed_from_snapshot"] = restored is not None
        report["resume_failure"] = (
            resume_failure.as_dict() if resume_failure is not None else None
        )
        exit_code = 0
        if stop_signal["value"] is not None:
            exit_code = 128 + int(stop_signal["value"])
        report["exit_code"] = exit_code
        _write_report(report_path, report)
        if not args.quiet:
            print(
                _json_line(
                    {
                        "event": "run_complete",
                        "report": str(report_path),
                        "exit_code": exit_code,
                        "snapshot_generation": (
                            snapshot_info.get("generation")
                            if isinstance(snapshot_info, dict)
                            else None
                        ),
                        "equivalence_passed": (
                            equivalence.get("passed")
                            if isinstance(equivalence, dict)
                            else None
                        ),
                    }
                ),
                flush=True,
            )
        return exit_code
    finally:
        engine.close()
        for signum, old_handler in old_handlers.items():
            try:
                signal.signal(signum, old_handler)
            except (ValueError, OSError):
                pass


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
