"""可重复诊断报告与修正文本输出（临时文件 + 原子替换）。"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .models import ERROR_TYPES, SampleResult
from .precision import format_sig6
from .snapshot import Manifest

HIST_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
HIST_LABELS = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]


def _histogram(scores) -> Dict[str, int]:
    counts = {label: 0 for label in HIST_LABELS}
    for item in scores:
        value = item.total_confidence
        for idx in range(len(HIST_BINS) - 1):
            if HIST_BINS[idx] <= value < HIST_BINS[idx + 1]:
                counts[HIST_LABELS[idx]] += 1
                break
    return counts


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                              indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_report(results: Sequence[SampleResult], failures: List[Dict[str, Any]],
                 manifest: Manifest, config_fp: str, rule_fp: str,
                 cache_counters: Dict[str, Any],
                 recovery: Dict[str, Any]) -> Dict[str, Any]:
    etype_dist: Counter = Counter()
    boundary_samples: List[str] = []
    max_dev: Dict[str, Any] = {"max_abs_diff": 0.0}
    all_audits = []
    decision_flips: List[Dict[str, Any]] = []
    contradictions: List[Dict[str, Any]] = []

    fp32_scores = []
    fp64_scores = []
    for result in sorted(results, key=lambda item: item.sample.sample_id):
        for candidate in result.candidates:
            etype_dist[candidate.etype] += 1
        fp32_scores.extend(result.scores_fp32)
        fp64_scores.extend(result.scores_fp64)
        boundary_any = result.boundary or any(item.boundary for item in result.scores_fp64)
        if boundary_any:
            boundary_samples.append(result.sample.sample_id)
        for audit in result.audits:
            all_audits.append(audit)
            if audit.max_abs_diff >= max_dev["max_abs_diff"]:
                max_dev = {
                    "sample_id": audit.sample_id,
                    "rule_id": audit.rule_id,
                    "start": audit.start,
                    "max_abs_diff": format_sig6(audit.max_abs_diff),
                    "first_breach_stage": audit.first_breach_stage,
                    "attribution": audit.attribution,
                }
            if audit.decision_flip:
                decision_flips.append({
                    "sample_id": audit.sample_id,
                    "rule_id": audit.rule_id,
                    "start": audit.start,
                    "attribution": audit.attribution,
                })
            if audit.ranking_contradiction:
                contradictions.append({
                    "sample_id": audit.sample_id,
                    "rule_id": audit.rule_id,
                    "start": audit.start,
                    "attribution": audit.attribution,
                })

    report = {
        "schema_version": 1,
        "fingerprints": {
            "config": config_fp,
            "rules": rule_fp,
            "scoring": manifest.scoring_fingerprint,
        },
        "snapshot": {
            "global_version": manifest.global_version,
            "stage_versions": dict(manifest.stage_versions),
            "completed_cursor": list(manifest.completed_cursor),
            "recovery": recovery,
        },
        "error_type_distribution": {etype: etype_dist.get(etype, 0)
                                    for etype in ERROR_TYPES},
        "histograms": {
            "fixed_point_float32": _histogram(fp32_scores),
            "high_precision_float64": _histogram(fp64_scores),
        },
        "max_deviation_candidate": max_dev,
        "boundary_sensitive_samples": sorted(boundary_samples),
        "decision_flips": decision_flips,
        "ranking_contradictions": contradictions,
        "precision_audits": [
            {
                "sample_id": audit.sample_id,
                "rule_id": audit.rule_id,
                "start": audit.start,
                "stage_diffs": [
                    {key: (format_sig6(value) if key in
                           ("abs_diff", "rel_diff", "fp32_value", "fp64_value")
                           and value is not None else value)
                     for key, value in diff.items()}
                    for diff in audit.stage_diffs
                ],
                "first_breach_stage": audit.first_breach_stage,
                "decision_flip": audit.decision_flip,
                "ranking_contradiction": audit.ranking_contradiction,
                "attribution": audit.attribution,
                "max_abs_diff": format_sig6(audit.max_abs_diff),
            }
            for audit in sorted(all_audits,
                                key=lambda a: (a.sample_id, a.start, a.rule_id))
        ],
        "cache": {
            "hits": cache_counters.get("hits", 0),
            "misses": cache_counters.get("misses", 0),
            "reused_count": cache_counters.get("reused_count", 0),
            "recomputed_count": cache_counters.get("recomputed_count", 0),
            "reused": cache_counters.get("reused", []),
            "recomputed": cache_counters.get("recomputed", []),
        },
        "failures": sorted(failures, key=lambda item:
                           (item.get("sample_id") or "", item.get("code", ""),
                            item.get("source", ""))),
        "sample_count": len({r.sample.sample_id for r in results}),
    }
    return report


def write_report(path: str | Path, report: Dict[str, Any]) -> None:
    _atomic_write_json(Path(path), report)


def write_corrections(path: str | Path, results: Sequence[SampleResult]) -> None:
    lines: List[str] = []
    for result in sorted(results, key=lambda item: item.sample.sample_id):
        record = {
            "sample_id": result.sample.sample_id,
            "original_text": result.sample.text,
            "corrected_text": result.corrected_text,
            "edits": [edit.to_fingerprint_obj() for edit in result.edits],
            "boundary": result.boundary,
        }
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
    _atomic_write_text(Path(path), "\n".join(lines) + ("\n" if lines else ""))
