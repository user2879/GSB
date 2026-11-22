"""Collaborative diagnostic report assembly and cross-run consistency checks."""

from __future__ import annotations

from typing import Any, Optional

from .fixedpoint import to_float
from .models import RegionResult, LedgerEntry
from .engine import EngineResult
from .injection import Injection


def region_result_row(result: RegionResult, frac_bits: int) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "region": result.region,
        "category": result.category,
        "confidence_fixed": result.confidence_fixed,
        "confidence_float": round(to_float(result.confidence_fixed, frac_bits), 6),
        "boundary": result.boundary,
        "boundary_reasons": list(result.boundary_reasons),
        "visual_score_fixed": result.visual_score_fixed,
        "text_score_fixed": result.text_score_fixed,
        "visual_weight_fixed": result.visual_weight_fixed,
        "text_weight_fixed": result.text_weight_fixed,
        "visual_quantized": result.visual_quantized,
        "text_quantized": result.text_quantized,
        "contributions": [c.to_dict() for c in result.contributions],
        "referenced_assertions": list(result.referenced_assertions),
        "referenced_regions": list(result.referenced_regions),
        "cache_key": result.cache_key,
        "reused": result.reused,
    }


def ledger_comparison_payload(ledger: list[LedgerEntry]) -> list[dict[str, Any]]:
    return [entry.to_dict() for entry in sorted(ledger, key=lambda e: e.sort_key)]


def compare_runs(label_a: str, run_a: dict, label_b: str, run_b: dict) -> dict[str, Any]:
    """Field-by-field diff of two serialized run payloads."""
    keys = sorted(set(run_a) | set(run_b))
    mismatches: list[dict[str, Any]] = []
    for key in keys:
        if run_a.get(key) != run_b.get(key):
            mismatches.append(
                {
                    "field": key,
                    label_a: run_a.get(key),
                    label_b: run_b.get(key),
                }
            )
    return {
        "equal": not mismatches,
        "compared_fields": keys,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def injection_summary(
    injections: list[Injection], failures: list  # list[FailureRecord]
) -> list[dict[str, Any]]:
    by_code: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for failure in failures:
        by_code.setdefault((failure.sample or "", failure.code), []).append(
            failure.to_dict()
        )
    summary: list[dict[str, Any]] = []
    for injection in injections:
        matched: list[dict[str, Any]] = []
        if injection.injection_type in (
            "nonfinite_fusion_param",
            "threshold_out_of_range",
        ):
            matched = by_code.get(("", injection.expected_code), [])
        else:
            matched = by_code.get(
                (injection.sample_id or "", injection.expected_code), []
            )
        summary.append(
            {
                "injection_id": injection.injection_id,
                "injection_type": injection.injection_type,
                "sample_id": injection.sample_id,
                "stage": injection.stage,
                "expected_code": injection.expected_code,
                "hit": bool(matched),
                "matched_failures": matched,
            }
        )
    return summary


def build_report(
    engine: EngineResult,
    config,
    *,
    injections: Optional[list[Injection]] = None,
    consistency: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    injections = injections or []
    boundary_rows = [
        region_result_row(r, config.frac_bits)
        for r in engine.results
        if r.boundary
    ]
    report = {
        "meta": {
            "config_hash": config.fingerprint(),
            "quant_level": config.quant_level,
            "frac_bits": config.frac_bits,
            "confidence_threshold": config.confidence_threshold,
            "fusion_temperature": config.fusion_temperature,
            "max_workers": config.max_workers,
            "snapshot_interval": config.snapshot_interval,
            "relation_hash": engine.relation_hash,
            "recovery_version": engine.recovery_version,
            "final_version": engine.final_version,
        },
        "region_results": [
            region_result_row(r, config.frac_bits) for r in engine.results
        ],
        "boundary_samples": boundary_rows,
        "ledger": ledger_comparison_payload(engine.ledger),
        "audit": {
            "reused": list(engine.audit.reused),
            "recomputed": list(engine.audit.recomputed),
            "cache_hits": dict(engine.audit.cache_hits),
        },
        "failures": [f.to_dict() for f in engine.failures],
        "injections": injection_summary(injections, engine.failures),
        "consistency": consistency or {},
    }
    if engine.stale_error is not None:
        report["stale_snapshot"] = engine.stale_error.to_record()
    if extra:
        report["extra"] = extra
    return report
