"""完整运行的 JSON 诊断报告生成。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .pipeline import PipelineResult


def _safe_float(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def build_report(result: PipelineResult, calibration_summary: list[dict] | None = None) -> dict:
    desc = result.desc
    position_summaries = []
    for position in sorted(result.positions):
        item = result.positions[position]
        if item.get("failed"):
            position_summaries.append(
                {
                    "position": position,
                    "status": "isolated_failure",
                    "failure_code": item.get("failure_code"),
                    "injection_id": item.get("injection_id"),
                }
            )
            continue
        kernel = item["kernel"]
        trunc = item["truncation"]
        deq = item["dequant"]
        position_summaries.append(
            {
                "position": position,
                "batch": item["batch"],
                "status": "reused" if item.get("reused") else "computed",
                "out_channel": kernel["out_channel"],
                "out_h": kernel["out_h"],
                "out_w": kernel["out_w"],
                "raw_sum": kernel["raw_sum"],
                "buffered": kernel["buffered"],
                "overflow": kernel["overflow"],
                "truncated": trunc["truncated"],
                "boundary_sample": trunc["boundary_sample"],
                "dequant_value": _safe_float(deq["value"]),
                "clipped": deq["clipped"],
                "first_fail_stage": item["attribution"].first_fail_stage,
                "cause": item["attribution"].cause,
            }
        )

    attribution_dict = result.attribution.to_dict()
    failures = result.ledger.to_list() if result.ledger is not None else []
    report = {
        "descriptor": desc.to_summary(),
        "run": {
            "threaded": result.plan is not None,
            "resumed": result.resumed,
            "concurrency_configured": desc.concurrency,
            "concurrency_peak": result.concurrency_peak,
            "total_positions": desc.total_positions,
            "positions_ok": len([p for p in position_summaries if p["status"] != "isolated_failure"]),
            "positions_isolated": len([p for p in position_summaries if p["status"] == "isolated_failure"]),
        },
        "snapshot": {
            "versions": result.snapshot_versions,
            "completed_batches": result.completed_batches,
            "consumed_budget": result.consumed_budget,
        },
        "budget": {
            "batch_count": len(result.plan.batches) if result.plan else 0,
            "mac_budget_per_batch": desc.mac_budget,
            "macs_per_position": desc.macs_per_position,
            "deduction_sequence": result.plan.budget_ledger if result.plan else [],
        },
        "cache": result.cache.snapshot(),
        "attribution": {
            "exceed_counts": attribution_dict["exceed_counts"],
            "max_diff_per_stage": attribution_dict["max_diff_per_stage"],
            "error_histogram": attribution_dict["error_histogram"],
        },
        "injections": {
            "hits": result.injection_hits,
            "global_artifacts": result.global_injection_artifacts,
        },
        "calibration": calibration_summary or [],
        "position_results": position_summaries,
        "failures": failures,
        "failure_code_counts": _count_codes(failures),
    }
    return report


def _count_codes(failures: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in failures:
        counts[item["code"]] = counts.get(item["code"], 0) + 1
    return dict(sorted(counts.items()))


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return path


def ensure_writable_dirs(*paths: str | Path) -> list[Path]:
    result = []
    for raw in paths:
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        os.remove(probe)
        result.append(path)
    return result
