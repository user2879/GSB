"""Fixed-point statistics with explicit quantization error bounds.

All ratios are expressed in basis points (per 10000) by FLOOR truncation
(rounding direction: toward zero); the quantization error is therefore in
[0, 1) bp and the reported upper bound is 1 bp. Average durations are
integer milliseconds, also floor-truncated with a 1 ms bound. Exact
(numerator, denominator) pairs are always carried alongside so results can
be independently recomputed at any scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

RATIO_UNIT = 10000
RATIO_ERROR_BOUND_BP = 1   # floor truncation: 0 <= error < 1 bp
DURATION_ERROR_BOUND_MS = 1
ROUNDING = "floor"


@dataclass(frozen=True)
class FixedRatio:
    numerator: int
    denominator: int
    value_bp: int          # floor(numerator * 10000 / denominator)
    error_bound_bp: int = RATIO_ERROR_BOUND_BP
    rounding: str = ROUNDING


@dataclass(frozen=True)
class FixedAverage:
    total: int
    count: int
    value: int             # floor(total / count)
    error_bound: int = DURATION_ERROR_BOUND_MS
    rounding: str = ROUNDING


def ratio_bp(part: int, whole: int) -> FixedRatio:
    value = (part * RATIO_UNIT) // whole if whole else 0
    return FixedRatio(part, whole, value)


def average_ms(total_ms: int, count: int) -> FixedAverage:
    value = total_ms // count if count else 0
    return FixedAverage(total_ms, count, value)


def canary_distribution(ledger_rows: List[dict]) -> Dict[str, Dict[str, FixedRatio]]:
    """Per-rule stable/canary hit ratios in basis points."""
    per_rule: Dict[str, Dict[str, int]] = {}
    for row in ledger_rows:
        counts = per_rule.setdefault(row["rule_id"], {"stable": 0, "canary": 0})
        counts[row["variant"]] += 1
    result: Dict[str, Dict[str, FixedRatio]] = {}
    for rule_id in sorted(per_rule):
        counts = per_rule[rule_id]
        total = counts["stable"] + counts["canary"]
        result[rule_id] = {
            "stable": ratio_bp(counts["stable"], total),
            "canary": ratio_bp(counts["canary"], total),
        }
    return result


def batch_hit_stats(ledger_rows: List[dict]) -> Dict[str, Dict[str, object]]:
    """Per-batch hit counts and average integer durations with bounds."""
    per_batch: Dict[str, List[dict]] = {}
    for row in ledger_rows:
        per_batch.setdefault(row["batch_id"], []).append(row)
    result: Dict[str, Dict[str, object]] = {}
    for batch_id in sorted(per_batch):
        rows = per_batch[batch_id]
        stable = sum(1 for r in rows if r["variant"] == "stable")
        canary = len(rows) - stable
        total_ms = sum(r["duration_ms"] for r in rows)
        result[batch_id] = {
            "stable_hits": stable,
            "canary_hits": canary,
            "canary_ratio": ratio_bp(canary, len(rows)),
            "avg_duration": average_ms(total_ms, len(rows)),
        }
    return result
