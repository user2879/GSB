from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import InferenceConfig, sha256_canonical
from .errors import failure_sort_key
from .injector import InjectionSet
from .loader import CorpusLoadResult
from .params import ParameterPackage


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _stage_breaches(stage_timings: dict[str, dict[str, float]], latency_budget_ms: int) -> list[dict[str, Any]]:
    breaches: list[dict[str, Any]] = []
    for sample_id in sorted(stage_timings):
        timings = stage_timings[sample_id]
        for stage in ("emission", "decode", "merge", "total"):
            value = float(timings.get(stage, 0.0))
            if value > latency_budget_ms:
                breaches.append(
                    {
                        "sample_id": sample_id,
                        "stage": stage,
                        "elapsed_ms": round(value, 6),
                        "budget_ms": latency_budget_ms,
                    }
                )
    return breaches


def build_report(
    *,
    cfg: InferenceConfig,
    corpus: CorpusLoadResult,
    params: ParameterPackage | None,
    injections: InjectionSet,
    manifest: dict[str, Any],
    artifacts: dict[str, Any],
    results: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    ledger: Sequence[dict[str, Any]],
    injection_hits: Sequence[dict[str, Any]],
    cache: dict[str, Any],
    collisions: dict[str, Any],
    stage_timings_ms: dict[str, dict[str, float]],
    total_wall_ms: float,
    param_summary_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sorted_results = sorted(results, key=lambda item: str(item.get("sample_id", "")))
    sorted_failures = sorted(failures, key=failure_sort_key)
    sorted_ledger = sorted(ledger, key=lambda item: int(item.get("sequence", 0)))
    sorted_hits = sorted(
        injection_hits,
        key=lambda item: (str(item.get("sample_id")), str(item.get("stage")), str(item.get("injection_id"))),
    )
    consumed = sum(int(entry.get("allocated", 0)) for entry in sorted_ledger)
    entity_count = sum(len(item.get("entities", [])) for item in sorted_results)
    repair_count = sum(len(item.get("repairs", [])) for item in sorted_results)
    unique_failed = {str(item.get("sample_id")) for item in sorted_failures}
    summary = {
        "valid_sample_count": len(corpus.samples),
        "load_failure_count": len(corpus.failures),
        "processed_sample_count": len(sorted_results) + len({str(item.get("sample_id")) for item in sorted_failures if str(item.get("stage")) != "load"}),
        "successful_sample_count": len(sorted_results),
        "failed_sample_count": len(unique_failed),
        "entity_count": entity_count,
        "repair_count": repair_count,
        "injection_configured_count": len(injections.all),
        "injection_hit_count": len(sorted_hits),
        "global_char_budget": cfg.global_char_budget,
        "consumed_char_budget": consumed,
        "remaining_char_budget": max(0, cfg.global_char_budget - consumed),
        "cache_eviction_count": int(cache.get("eviction_count", 0)),
        "cache_recomputation_count": int(cache.get("recomputations", cache.get("recomputation_count", 0))),
        "collision_events": int(collisions.get("collision_events", 0)),
    }
    model_summary = param_summary_override or (params.summary() if params is not None else {})
    deterministic = {
        "samples": sorted_results,
        "failures": sorted_failures,
        "budget": {
            "global_char_budget": cfg.global_char_budget,
            "per_sample_quota": cfg.per_sample_quota,
            "consumed_char_budget": consumed,
            "remaining_char_budget": max(0, cfg.global_char_budget - consumed),
            "deduction_sequence": sorted_ledger,
        },
        "cache": cache,
        "injections": {
            "configured": [item.as_dict() for item in injections.all],
            "hits": sorted_hits,
        },
        "feature_hash_collisions": collisions,
        "snapshot": {
            "completed_sample_ids": list(manifest.get("completed_sample_ids", [])),
            "consumed_char_budget": int(manifest.get("consumed_char_budget", 0)),
            "recovery_point_sample_id": manifest.get("recovery_point_sample_id"),
            "status": manifest.get("status"),
        },
        "summary": summary,
    }
    deterministic_digest = sha256_canonical(deterministic)
    diagnostics = {
        "total_wall_ms": round(float(total_wall_ms), 6),
        "stage_timings_ms": stage_timings_ms,
        "latency_budget_ms": cfg.latency_budget_ms,
        "latency_breaches": _stage_breaches(stage_timings_ms, cfg.latency_budget_ms),
        "artifact_counts": {
            name: len(artifacts.get(name, {})) if isinstance(artifacts.get(name), dict) else 0
            for name in ("emission", "decode", "merge")
        },
    }
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": str(manifest.get("status", "unknown")),
        "deterministic_digest": deterministic_digest,
        "configuration": {
            "normalized": cfg.as_dict(),
            "config_hash": cfg.config_hash,
            "corpus_hash": corpus.corpus_hash,
            "injection_hash": injections.source_hash,
        },
        "model": model_summary,
        "snapshot_manifest": manifest,
        "run_metadata": {
            "max_workers": cfg.max_workers,
            "processing_unit": cfg.processing_unit,
            "total_wall_ms": round(float(total_wall_ms), 6),
        },
        "resume": {
            "recovery_point_sample_id": manifest.get("recovery_point_sample_id"),
            "completed_sample_count": len(manifest.get("completed_sample_ids", [])),
            "consumed_char_budget": int(manifest.get("consumed_char_budget", 0)),
        },
        "deterministic": deterministic,
        "diagnostics": diagnostics,
    }
    return report


def write_report_bundle(
    report: dict[str, Any],
    output_dir: Path,
    cfg: InferenceConfig,
    ctx: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    from .dashboard import render_dashboard

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    deterministic_path = output_dir / "deterministic.json"
    dashboard_path = output_dir / "dashboard.html"
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    deterministic_bytes = (
        json.dumps(report["deterministic"], ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(report_path, report_bytes)
    _atomic_write(deterministic_path, deterministic_bytes)
    _atomic_write(dashboard_path, render_dashboard(report).encode("utf-8"))
    return report_path, dashboard_path
