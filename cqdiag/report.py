"""诊断报告输出 (业务要求 8)."""

from __future__ import annotations

import json

from . import FORMAT_VERSION
from .pipeline import RunResult


def build_report(result: RunResult, extra: dict | None = None) -> dict:
    columns_summary = {}
    for col_key in sorted(result.columns):
        entry = result.columns[col_key]
        calib = entry.get("calib", {})
        scale = entry.get("scale", {})
        quant = entry.get("quant", {})
        dequant = entry.get("dequant", {})
        dominant = None
        if dequant:
            from collections import Counter
            counter = Counter(a["source"] for a in dequant["attributions"] if a)
            if counter:
                dominant = counter.most_common(1)[0][0]
        columns_summary[col_key] = {
            "scale": scale.get("scale"),
            "zero_point": scale.get("zero_point"),
            "valid_count": calib.get("valid_count"),
            "clipped_count": quant.get("clipped_count"),
            "oor_count": dequant.get("oor_count"),
            "max_abs_err": dequant.get("max_abs_err"),
            "max_rel_err": dequant.get("max_rel_err"),
            "dominant_source": dominant,
        }
    report = {
        "format_version": FORMAT_VERSION,
        "global_version": result.global_version,
        "config_hash": result.config_hash,
        "version_change": result.version_change,
        "recovery_actions": result.recovery_actions,
        "schedule": {
            "budget_spent": result.budget_spent,
            "budget_log": result.budget_log,
            "stage_stats": result.stats,
        },
        "columns": columns_summary,
        "groups": {k: result.stratify[k] for k in sorted(result.stratify)},
        "failures": result.failures.to_list(),
    }
    if extra:
        report.update(extra)
    return report


def write_report(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")
