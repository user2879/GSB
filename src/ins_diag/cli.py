from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import ConfigValidationError, load_config
from .pipeline import config_error_report, run_pipeline
from .util import atomic_write_json


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _write_or_emit(report: dict[str, Any], output: str | None, *, compact: bool = True) -> int:
    if output:
        atomic_write_json(Path(output), report)
    if compact:
        summary = {
            "status": report.get("status"),
            "error_codes": report.get("error_codes", []),
            "window_count": report.get("data", {}).get("window_count"),
            "boundary_windows": report.get("boundary_windows", []),
            "snapshot": report.get("snapshot", {}).get("commit"),
            "recovery": report.get("recovery"),
            "cache": {
                key: report.get("cache", {}).get(key)
                for key in ("reused_count", "recomputed_count")
            },
            "output": output,
        }
        _emit(summary)
    else:
        _emit(report)
    if report.get("status") == "error":
        return 2
    if report.get("status") == "failed":
        return 3
    return 0


def command_validate(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigValidationError as error:
        report = config_error_report(error)
        _write_or_emit(report, args.output, compact=False)
        return 2
    result = {
        "status": "ok",
        "valid": True,
        "config": str(config.config_path),
        "config_hash": config.config_hash,
        "waveform": str(config.waveform_path),
        "edits": str(config.edits_path) if config.edits_path else None,
        "lease": str(config.lease_path),
        "window": {"length": config.window_length, "shift": config.window_shift},
        "families": config.families,
        "family_limit": config.family_limit,
        "fixed_point": {
            "fractional_bits": config.fixed_point.fractional_bits,
            "quantization_units": config.fixed_point.quantization_units,
        },
        "kernel_bandwidth": config.bandwidth_label,
        "lease_ttl_ms": config.lease.ttl_ms,
    }
    if args.output:
        atomic_write_json(Path(args.output), result)
    _emit(result)
    return 0


def command_run(args: argparse.Namespace) -> int:
    report = run_pipeline(
        args.config,
        state_dir=args.state_dir,
        output_path=args.output,
        crash_after=args.crash_after,
        edits_limit=args.edits_limit,
    )
    return _write_or_emit(report, args.output, compact=not args.full)


def _canonical_multiset(events: list[dict[str, Any]]) -> list[tuple[str, Any, Any]]:
    return sorted(
        (str(event.get("kind")), event.get("window_index"), event.get("path"))
        for event in events
    )


def canonical_analysis(report: dict[str, Any]) -> dict[str, Any]:
    windows = []
    for window in report.get("windows", []):
        windows.append(
            {
                "window_index": window.get("window_index"),
                "window_id": window.get("window_id"),
                "device_id": window.get("device_id"),
                "channel_id": window.get("channel_id"),
                "start": window.get("start"),
                "end": window.get("end"),
                "status": window.get("status"),
                "raw_features": window.get("raw_features"),
                "fill_source": window.get("fill_source"),
                "standardized_vector": window.get("standardized_vector"),
                "standardized_fixed_units": window.get("standardized_fixed_units"),
                "cluster_id": window.get("cluster_id"),
                "cluster_distance_units": window.get("cluster_distance_units"),
                "density": window.get("density"),
            }
        )
    clusters = report.get("clusters")
    if clusters is not None:
        clusters = {
            key: clusters.get(key)
            for key in (
                "fixed_point",
                "initial_centroid_window_indices",
                "centroids_units",
                "assignments",
                "cluster_members",
                "cluster_sse_units",
                "sse_units",
                "iterations",
                "converged",
                "rounds",
                "result_fingerprint",
            )
        }
    density = report.get("density")
    if density is not None:
        density = {
            key: density.get(key)
            for key in (
                "bandwidth",
                "threshold_permille",
                "maximum_log_density",
                "window_scores",
                "boundary_windows",
                "below_threshold_windows",
                "baseline_fingerprint",
            )
        }
    return {
        "config_hash": report.get("config", {}).get("hash"),
        "data_fingerprint": report.get("data", {}).get("fingerprint"),
        "window_set_fingerprint": report.get("data", {}).get("window_set_fingerprint"),
        "windows": windows,
        "standardization": report.get("standardization"),
        "clusters": clusters,
        "density": density,
        "boundary_windows": report.get("boundary_windows"),
        "failures": report.get("failures"),
        "failed_windows": report.get("failed_windows"),
        "artifact_order": _canonical_multiset(
            report.get("cache", {}).get("events", [])
        ),
    }


def command_verify(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    command_count = len(config.edits) if args.edits_limit is None else args.edits_limit
    with tempfile.TemporaryDirectory(prefix="ins-diag-verify-") as temporary:
        root = Path(temporary)
        incremental_state = root / "incremental-state"
        probes = list(range(command_count + 1))
        last_report = None
        for limit in probes:
            last_report = run_pipeline(
                args.config,
                state_dir=incremental_state,
                edits_limit=limit,
            )
        final_incremental = run_pipeline(
            args.config,
            state_dir=incremental_state,
            edits_limit=command_count,
        )
        full_state = root / "full-state"
        full_report = run_pipeline(
            args.config,
            state_dir=full_state,
            edits_limit=command_count,
        )
        canonical_incremental = canonical_analysis(final_incremental)
        canonical_full = canonical_analysis(full_report)
        mismatch = canonical_incremental != canonical_full
        result = {
            "status": "ok" if not mismatch else "mismatch",
            "incremental_status": final_incremental.get("status"),
            "full_status": full_report.get("status"),
            "edits_tested": command_count,
            "probe_limits": probes,
            "incremental_cache": {
                key: final_incremental.get("cache", {}).get(key)
                for key in ("reused_count", "recomputed_count")
            },
            "full_cache": {
                key: full_report.get("cache", {}).get(key)
                for key in ("reused_count", "recomputed_count")
            },
            "analysis_fields_equal": not mismatch,
            "incremental_result_fingerprint": (
                final_incremental.get("clusters") or {}
            ).get("result_fingerprint"),
            "full_result_fingerprint": (full_report.get("clusters") or {}).get(
                "result_fingerprint"
            ),
        }
        if args.output:
            atomic_write_json(Path(args.output), result)
        _emit(result)
        return 0 if not mismatch else 4


def command_faults(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    results: list[dict[str, Any]] = []
    expected_files = sorted(root.glob("*/expected.json"))
    with tempfile.TemporaryDirectory(prefix="ins-diag-faults-") as temporary:
        temp_root = Path(temporary)
        copied_root = temp_root / "faults"
        shutil.copytree(root, copied_root)
        for expected_path in expected_files:
            fault_dir = expected_path.parent
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            run_dir = copied_root / fault_dir.name
            seed_state = run_dir / "state_seed"
            if seed_state.exists():
                target_state = run_dir / "state"
                if target_state.exists():
                    shutil.rmtree(target_state)
                shutil.copytree(seed_state, target_state)
            report = run_pipeline(
                run_dir / "config.yaml",
                state_dir=run_dir / "state",
            )
            codes = {item["code"] for item in report.get("business_errors", [])}
            expected_codes = set(expected.get("expected_codes", []))
            hit = expected_codes.issubset(codes)
            results.append(
                {
                    "fault": fault_dir.name,
                    "expected_codes": sorted(expected_codes),
                    "observed_codes": sorted(codes),
                    "hit": hit,
                    "status": report.get("status"),
                    "window_count": report.get("data", {}).get("window_count"),
                    "failed_windows": len(report.get("failed_windows", [])),
                    "business_errors": report.get("business_errors", []),
                }
            )
    all_hit = all(item["hit"] for item in results)
    output = {
        "status": "ok" if all_hit else "mismatch",
        "all_faults_hit": all_hit,
        "faults": results,
    }
    if args.output:
        atomic_write_json(Path(args.output), output)
    _emit(output)
    return 0 if all_hit else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ins-diag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate YAML configuration")
    validate.add_argument("--config", required=True)
    validate.add_argument("--output")
    validate.set_defaults(func=command_validate)

    run = subparsers.add_parser("run", help="run deterministic diagnostics")
    run.add_argument("--config", required=True)
    run.add_argument("--state-dir")
    run.add_argument("--output")
    run.add_argument("--full", action="store_true", help="print the complete report")
    run.add_argument("--edits-limit", type=int)
    run.add_argument("--crash-after", choices=["after_lease", "after_cluster", "before_snapshot"])
    run.set_defaults(func=command_run)

    verify = subparsers.add_parser(
        "verify", help="compare incremental runs with a clean full rebuild"
    )
    verify.add_argument("--config", required=True)
    verify.add_argument("--edits-limit", type=int)
    verify.add_argument("--output")
    verify.set_defaults(func=command_verify)

    faults = subparsers.add_parser("faults", help="run independent failure injection cases")
    faults.add_argument("--root", default="examples/faults")
    faults.add_argument("--output")
    faults.set_defaults(func=command_faults)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
