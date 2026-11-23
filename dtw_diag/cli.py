"""Command line interface for the DTW diagnostics component."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .config import Config, load_config
from .engine import (
    DiagnosticEngine,
    canonical_equivalence,
    full_rebuild,
    run_plan_reloading_state,
)
from .fingerprints import canonical_json
from .storage import load_edit_plan, load_library, write_library_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_config(path: Path, config: Config) -> None:
    payload = {
        "reference_sequence_id": config.reference_sequence_id,
        "step_mode": config.step_mode,
        "bandwidth_permille": config.bandwidth_permille,
        "distance_coefficient": config.distance_coefficient,
        "fixed_point": {
            "fraction_bits": config.fraction_bits,
            "quantization_step": config.quantization_step,
        },
        "missing_marker": config.missing_marker,
        "missing_penalty": config.missing_penalty,
        "max_path_length": config.max_path_length,
        "path_length_policy": config.path_length_policy,
        "thread_pool_workers": config.thread_pool_workers,
        "cell_recompute_budget": config.cell_recompute_budget,
        "snapshot_interval_batches": config.snapshot_interval_batches,
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _load_inputs(
    config_path: Path, library_path: Path, edits_path: Path
) -> tuple[Config, Any, list[Any], list[Any]]:
    config_result = load_config(config_path)
    if config_result.config is None:
        raise ValueError(
            "configuration validation failed: "
            + canonical_json([item.to_dict() for item in config_result.failures])
        )
    library_result = load_library(library_path, config_result.config)
    plan_result = load_edit_plan(edits_path)
    if plan_result.failures:
        raise ValueError(
            "edit-plan validation failed: "
            + canonical_json([item.to_dict() for item in plan_result.failures])
        )
    return (
        config_result.config,
        library_result.library,
        plan_result.batches,
        library_result.failures,
    )


def _run_once(
    config: Config,
    library: Any,
    validation_failures: list[Any],
    batches: list[Any],
    workdir: Path,
) -> tuple[DiagnosticEngine, list[dict[str, Any]]]:
    engine = DiagnosticEngine(
        config,
        library.copy(),
        list(validation_failures),
        workdir=workdir,
        reset=True,
    )
    return run_plan_reloading_state(
        engine, batches, workdir=workdir, auto_drain=True
    )


def _report_payload(
    engine: DiagnosticEngine,
    reports: list[dict[str, Any]],
    incremental: dict[str, Any],
    full: dict[str, Any],
    equivalence: dict[str, Any],
    serial_result: dict[str, Any] | None,
    serial_engine: DiagnosticEngine | None,
) -> dict[str, Any]:
    budget_equal = None
    invalidation_equal = None
    reuse_equal = None
    result_equal = None
    if serial_engine is not None and serial_result is not None:
        budget_equal = engine.budget_ledger == serial_engine.budget_ledger
        invalidation_equal = engine.invalidation_ledger == serial_engine.invalidation_ledger
        reuse_equal = engine.reuse_ledger == serial_engine.reuse_ledger
        result_equal = canonical_equivalence(incremental, serial_result)["equal"]
    total_recomputed = sum(item.get("recomputed_cells", 0) for item in engine.reuse_ledger)
    total_reused = sum(
        item.get("reused_cells", 0)
        for item in engine.reuse_ledger
        if not item.get("result_reused")
    )
    result_reuse_batches = sum(
        1 for item in engine.reuse_ledger if item.get("result_reused")
    )
    pending = [
        seq_id
        for seq_id, slot in sorted(engine.slots.items())
        if slot.working is not None and not slot.working.complete
    ]
    pending_cells = sum(
        len(engine.slots[seq_id].working.pending_cells)
        for seq_id in pending
        if engine.slots[seq_id].working is not None
    )
    equal_all = bool(equivalence["equal"]) and all(
        item is not False
        for item in (budget_equal, invalidation_equal, reuse_equal, result_equal)
    )
    return {
        "status": "PASS" if equal_all else "FAIL",
        "incremental_vs_full_rebuild": equivalence,
        "serial_vs_parallel": {
            "results_equal": result_equal,
            "budget_ledger_equal": budget_equal,
            "invalidation_ledger_equal": invalidation_equal,
            "reuse_ledger_equal": reuse_equal,
            "parallel_workers": engine.config.thread_pool_workers,
            "serial_workers": serial_engine.config.thread_pool_workers
            if serial_engine
            else None,
        },
        "batch_count": len(reports),
        "state_reloaded_each_batch": True,
        "final_config": engine.config.to_dict(),
        "frozen_budget": engine.config.cell_recompute_budget,
        "final_pending_queue": pending,
        "final_pending_cell_count": pending_cells,
        "totals": {
            "recomputed_cells": total_recomputed,
            "reused_cells": total_reused,
            "result_reuse_batches": result_reuse_batches,
            "edit_failure_count": len(engine.edit_failure_ledger),
            "active_failure_count": len(incremental["failures"]),
        },
        "sequence_stats": engine.sequence_stats(),
        "failure_list": incremental["failures"],
        "edit_failure_ledger": engine.edit_failure_ledger,
        "budget_ledger": engine.budget_ledger,
        "invalidation_ledger": engine.invalidation_ledger,
        "reuse_ledger": engine.reuse_ledger,
        "batch_ledger": engine.batch_ledger,
    }


def _write_demo_outputs(
    *,
    outdir: Path,
    engine: DiagnosticEngine,
    reports: list[dict[str, Any]],
    incremental: dict[str, Any],
    full: dict[str, Any],
    equivalence: dict[str, Any],
    serial_result: dict[str, Any] | None,
    serial_engine: DiagnosticEngine | None,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    write_library_jsonl(outdir / "final_library.jsonl", engine.library.ordered_records())
    _write_config(outdir / "final_config.yaml", engine.config)
    _write_json(outdir / "incremental_result.json", incremental)
    _write_json(outdir / "full_rebuild_result.json", full)
    _write_json(outdir / "equivalence.json", equivalence)
    _write_json(outdir / "budget_ledger.json", engine.budget_ledger)
    _write_json(outdir / "invalidation_ledger.json", engine.invalidation_ledger)
    _write_json(outdir / "reuse_ledger.json", engine.reuse_ledger)
    _write_json(outdir / "failure_list.json", incremental["failures"])
    _write_json(outdir / "edit_failure_ledger.json", engine.edit_failure_ledger)
    report = _report_payload(
        engine,
        reports,
        incremental,
        full,
        equivalence,
        serial_result,
        serial_engine,
    )
    _write_json(outdir / "report.json", report)
    return report


def run_demo(args: argparse.Namespace) -> int:
    samples = Path(args.samples)
    if args.generate or not (samples / "config.yaml").exists():
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_samples.py")],
            check=True,
            cwd=PROJECT_ROOT,
        )
    config, library, batches, validation_failures = _load_inputs(
        samples / "config.yaml",
        samples / "sequences.jsonl",
        samples / "edits.json",
    )
    outdir = Path(args.out)
    workdir = outdir / "incremental_state"
    engine, reports = _run_once(
        config,
        library,
        validation_failures,
        batches,
        workdir,
    )
    incremental = engine.export_result()
    full = full_rebuild(engine.config, engine.library, engine.validation_failures)
    equivalence = canonical_equivalence(incremental, full)

    serial_config = replace(config, thread_pool_workers=1)
    with tempfile.TemporaryDirectory(prefix="dtw_diag_serial_") as temporary:
        serial_engine, _serial_reports = _run_once(
            serial_config,
            library,
            validation_failures,
            batches,
            Path(temporary) / "state",
        )
        serial_result = serial_engine.export_result()

        report = _write_demo_outputs(
            outdir=outdir,
            engine=engine,
            reports=reports,
            incremental=incremental,
            full=full,
            equivalence=equivalence,
            serial_result=serial_result,
            serial_engine=serial_engine,
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "batches": report["batch_count"],
                "sequences": len(incremental["sequences"]),
                "failures": len(incremental["failures"]),
                "report": str(outdir / "report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


def run_incremental(args: argparse.Namespace) -> int:
    config, library, batches, validation_failures = _load_inputs(
        Path(args.config), Path(args.library), Path(args.edits)
    )
    if getattr(args, "workers", None) is not None:
        config = replace(config, thread_pool_workers=args.workers)
    workdir = Path(args.workdir)
    engine, reports = _run_once(
        config, library, validation_failures, batches, workdir
    )
    incremental = engine.export_result()
    full = full_rebuild(engine.config, engine.library, engine.validation_failures)
    equivalence = canonical_equivalence(incremental, full)
    report = _write_demo_outputs(
        outdir=workdir,
        engine=engine,
        reports=reports,
        incremental=incremental,
        full=full,
        equivalence=equivalence,
        serial_result=None,
        serial_engine=None,
    )
    print(json.dumps({"status": report["status"], "report": str(workdir / "report.json")}, indent=2))
    return 0 if equivalence["equal"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the deterministic sample end-to-end")
    demo.add_argument("--samples", default=str(PROJECT_ROOT / "samples"))
    demo.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "demo"))
    demo.add_argument("--generate", action="store_true")
    demo.set_defaults(func=run_demo)

    run = subparsers.add_parser("run", help="apply an edit plan and compare with rebuild")
    run.add_argument("--config", required=True)
    run.add_argument("--library", required=True)
    run.add_argument("--edits", required=True)
    run.add_argument("--workdir", required=True)
    run.add_argument("--workers", type=int)
    run.set_defaults(func=run_incremental)

    generate = subparsers.add_parser("generate", help="regenerate deterministic samples")
    generate.set_defaults(func=lambda _args: subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_samples.py")],
        check=True,
        cwd=PROJECT_ROOT,
    ).returncode)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
