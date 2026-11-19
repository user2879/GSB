"""One-command deterministic demonstration."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .canonical import sha256_file, write_canonical
from .config import (
    BudgetConfig,
    ClientBufferConfig,
    DecisionConfig,
    EncoderConfig,
    FaultConfig,
    MediaConfig,
    RetryConfig,
    WorkbenchConfig,
)
from .errors import WorkbenchError, require
from .pipeline import PipelineResult, run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hls-workbench",
        description="Deterministic HLS parallel transcode and adaptive-buffer workbench.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the complete deterministic demonstration")
    demo.add_argument("--output", default="artifacts/demo", help="output root (default: artifacts/demo)")
    demo.add_argument("--seed", type=int, default=20240517)
    demo.add_argument("--duration", type=float, default=30.0)
    demo.add_argument("--slice-duration", type=float, default=2.0)
    demo.add_argument("--fps", type=int, default=30)
    demo.add_argument("--width", type=int, default=1280)
    demo.add_argument("--height", type=int, default=720)
    demo.add_argument("--workers", type=int, default=4)
    demo.add_argument("--memory-budget-mb", type=int, default=300)
    demo.add_argument("--cpu-budget-ms", type=int, default=200)
    demo.add_argument("--fault-task", default="720p:00003")
    demo.add_argument("--fault-failures", type=int, default=3)
    demo.add_argument("--retry-max", type=int, default=2)
    demo.add_argument("--backoff-base-ms", type=int, default=15)
    demo.add_argument("--backoff-multiplier", type=float, default=2.0)
    demo.add_argument("--buffer-capacity-ms", type=int, default=8000)
    demo.add_argument("--buffer-threshold-ms", type=int, default=2600)
    demo.add_argument("--buffer-step-ms", type=int, default=100)
    demo.add_argument("--startup-buffer-ms", type=int, default=0)
    demo.add_argument("--bandwidth-kbps", type=int, default=2600)
    demo.add_argument("--bandwidth-safety-ratio", type=float, default=0.85)
    demo.add_argument("--variant-bandwidth-kbps", type=int, default=900)
    demo.add_argument("--variant-threshold-ms", type=int, default=2600)
    demo.add_argument("--clean", action="store_true", help="remove the output root before running")
    demo.add_argument("--skip-repeat", action="store_true", help="run only one full pass")
    return parser


def _config(args: argparse.Namespace) -> WorkbenchConfig:
    return WorkbenchConfig(
        media=MediaConfig(
            seed=args.seed,
            duration_seconds=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            slice_seconds=args.slice_duration,
        ),
        encoder=EncoderConfig(),
        budget=BudgetConfig(
            max_workers=args.workers,
            memory_budget_mb=args.memory_budget_mb,
            cpu_budget_ms=args.cpu_budget_ms,
        ),
        retry=RetryConfig(
            max_retries=args.retry_max,
            base_delay_ms=args.backoff_base_ms,
            multiplier=args.backoff_multiplier,
        ),
        fault=FaultConfig(task_key=args.fault_task, failures=args.fault_failures),
        client_buffer=ClientBufferConfig(
            capacity_ms=args.buffer_capacity_ms,
            threshold_ms=args.buffer_threshold_ms,
            step_ms=args.buffer_step_ms,
            startup_buffer_ms=args.startup_buffer_ms,
        ),
        decision=DecisionConfig(
            bandwidth_kbps=args.bandwidth_kbps,
            bandwidth_safety_ratio=args.bandwidth_safety_ratio,
        ),
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _compare_runs(run_a: PipelineResult, run_b: PipelineResult) -> dict[str, Any]:
    hashes_a = _tree_hashes(run_a.root)
    hashes_b = _tree_hashes(run_b.root)
    files = sorted(set(hashes_a) | set(hashes_b))
    mismatches = [
        name for name in files if hashes_a.get(name) != hashes_b.get(name)
    ]
    report_a = run_a.report_path.read_bytes()
    report_b = run_b.report_path.read_bytes()
    return {
        "schema_version": 1,
        "report_bytes_equal": report_a == report_b,
        "run_a_report_sha256": run_a.report_sha256,
        "run_b_report_sha256": run_b.report_sha256,
        "file_count": len(files),
        "all_file_hashes_equal": not mismatches,
        "mismatched_files": mismatches,
        "run_a_tree_sha256": _hash_tree(hashes_a),
        "run_b_tree_sha256": _hash_tree(hashes_b),
    }


def _hash_tree(values: dict[str, str]) -> str:
    from .canonical import hash_object

    return hash_object(values)


def _variant(
    run_a: PipelineResult,
    *,
    bandwidth_kbps: int,
    threshold_ms: int,
    output_path: Path,
) -> dict[str, Any]:
    before = {
        artifact.relative_path: artifact.sha256 for artifact in run_a.artifacts
    }
    variant_decision = run_a.engine.decide(
        bandwidth_kbps=bandwidth_kbps,
        threshold_ms=threshold_ms,
    )
    after = {
        artifact.relative_path: sha256_file(run_a.root / artifact.relative_path)
        for artifact in run_a.artifacts
    }
    primary_candidates = {item["profile"] for item in run_a.decision.candidates}
    variant_candidates = {item["profile"] for item in variant_decision.candidates}
    result = {
        "schema_version": 1,
        "primary_decision": run_a.decision.decision_key,
        "variant_decision": variant_decision.to_dict(),
        "affected_profiles": sorted(primary_candidates | variant_candidates),
        "removed_candidate_profiles": sorted(primary_candidates - variant_candidates),
        "artifact_hashes_before": before,
        "artifact_hashes_after": after,
        "artifacts_unchanged": before == after,
        "reused_artifact_count": len(before),
        "reuse": {
            "model_cache_hits": list(variant_decision.model_cache_hits),
            "model_cache_misses": list(variant_decision.model_cache_misses),
            "recomputed_decisions": list(variant_decision.recomputed_decisions),
        },
    }
    write_canonical(output_path, result)
    return result


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output)
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _config(args)
    config.validate()

    run_a = run_pipeline(config, output_root / "run_a")
    if args.skip_repeat:
        return {
            "schema_version": 1,
            "report_sha256": run_a.report_sha256,
            "report": str(run_a.report_path),
            "selected_profile": run_a.decision.selected_profile,
            "repeatability": None,
        }

    run_b = run_pipeline(config, output_root / "run_b")
    comparison = _compare_runs(run_a, run_b)
    comparison_path = output_root / "repeatability.json"
    write_canonical(comparison_path, comparison)
    require(
        comparison["report_bytes_equal"] and comparison["all_file_hashes_equal"],
        "E_DETERMINISM_MISMATCH",
        "identical runs did not produce identical bytes",
        mismatches=comparison["mismatched_files"][:20],
    )

    variant = _variant(
        run_a,
        bandwidth_kbps=args.variant_bandwidth_kbps,
        threshold_ms=args.variant_threshold_ms,
        output_path=output_root / "variant_decision.json",
    )
    require(
        variant["artifacts_unchanged"],
        "E_DETERMINISM_MISMATCH",
        "variant decision changed transcode artifacts",
    )
    return {
        "schema_version": 1,
        "report_sha256": run_a.report_sha256,
        "report": str(run_a.report_path),
        "repeatability": str(comparison_path),
        "all_file_hashes_equal": comparison["all_file_hashes_equal"],
        "selected_profile": run_a.decision.selected_profile,
        "parallelism": run_a.decision.parallelism,
        "degradation_count": len(run_a.scheduler.degradations),
        "budget_denial_count": len(run_a.scheduler.budget_denials),
        "isolation_count": len(run_a.scheduler.isolation_events),
        "variant": str(output_root / "variant_decision.json"),
        "variant_selected_profile": variant["variant_decision"]["selected_profile"],
        "variant_artifacts_unchanged": variant["artifacts_unchanged"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            summary = run_demo(args)
        else:
            parser.error(f"unknown command: {args.command}")
            return 2
    except WorkbenchError as exc:
        error_path = Path(getattr(args, "output", "artifacts/demo")) / "error.json"
        write_canonical(error_path, exc.to_dict())
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
