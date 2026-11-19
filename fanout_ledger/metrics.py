"""Fixed-point metric aggregation and deterministic SLO judgment.

Fixed-point conventions (declared explicitly and embedded in every report):

* ratios (success rate) use SCALE_RATIO = 10000 (1 basis point, 0.01%).
  Quantization: truncation toward zero.  Error bound for any true ratio r:
      0 <= r - floor(r * 10000) / 10000 < 1 / 10000  (< 0.01 percentage point)
* throughput (KB/s) uses SCALE_KBPS = 100 (0.01 KB/s), truncation toward zero,
  error bound < 0.01 KB/s.
* P95 latency keeps the original integer milliseconds (nearest-rank method on
  the sorted integer samples), scale 1, error bound 0 ms.

Judgments are serialized as canonical JSON (sorted keys, fixed separators),
so judging the same ledger twice yields byte-identical conclusions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Scene
from .ledger import AttemptRecord, canonical_json

SCALE_RATIO = 10_000
SCALE_KBPS = 100
SCALE_LATENCY = 1
RATIO_ERROR_BOUND = " < 1/10000 (0.01 percentage point), truncation toward zero"
KBPS_ERROR_BOUND = " < 1/100 KB/s, truncation toward zero"
P95_ERROR_BOUND = "0 ms (integer nearest-rank)"

SUCCESS = "success"


@dataclass(frozen=True)
class MetricBlock:
    scope: str
    node_id: str | None
    samples: int
    success_count: int
    success_rate_scaled: int
    success_rate_scale: int
    p95_latency_ms: int
    throughput_kbps_scaled: int
    throughput_scale: int
    outcome_counts: dict[str, int]
    quantization: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "node_id": self.node_id,
            "samples": self.samples,
            "success_count": self.success_count,
            "success_rate_scaled": self.success_rate_scaled,
            "success_rate_scale": self.success_rate_scale,
            "p95_latency_ms": self.p95_latency_ms,
            "throughput_kbps_scaled": self.throughput_kbps_scaled,
            "throughput_scale": self.throughput_scale,
            "outcome_counts": dict(sorted(self.outcome_counts.items())),
            "quantization": self.quantization,
        }


def _quantization_block() -> dict[str, str]:
    return {
        "success_rate": f"scale={SCALE_RATIO}, truncate toward zero, error bound{RATIO_ERROR_BOUND}",
        "throughput_kbps": f"scale={SCALE_KBPS}, truncate toward zero, error bound{KBPS_ERROR_BOUND}",
        "p95_latency_ms": f"scale={SCALE_LATENCY}, nearest-rank integer, error bound {P95_ERROR_BOUND}",
    }


def nearest_rank_p95(latencies: list[int]) -> int:
    """Nearest-rank P95: rank = ceil(0.95 * N), 1-based index into sorted list."""
    if not latencies:
        return 0
    ordered = sorted(latencies)
    rank = int(np.ceil(0.95 * len(ordered)))
    rank = max(1, min(rank, len(ordered)))
    return ordered[rank - 1]


def _aggregate(scope: str, node_id: str | None, records: list[AttemptRecord],
               payload_bytes: int) -> MetricBlock:
    samples = len(records)
    outcome_counts = {name: 0 for name in
                      ("success", "failure", "timeout", "unreachable", "unknown")}
    latencies: list[int] = []
    for record in records:
        outcome_counts[record.outcome] = outcome_counts.get(record.outcome, 0) + 1
        latencies.append(record.elapsed_ms)
    success_count = outcome_counts[SUCCESS]
    if samples:
        success_rate_scaled = (success_count * SCALE_RATIO) // samples
    else:
        success_rate_scaled = 0
    p95 = nearest_rank_p95(latencies)
    if records:
        elapsed_span_ms = max(r.ended_ms for r in records) - min(r.started_ms for r in records)
        elapsed_span_ms = max(elapsed_span_ms, 1)
        total_kb = (samples * payload_bytes) / 1024.0
        kbps = total_kb / (elapsed_span_ms / 1000.0)
        throughput_scaled = int(kbps * SCALE_KBPS)  # truncate toward zero
    else:
        throughput_scaled = 0
    return MetricBlock(
        scope=scope, node_id=node_id, samples=samples, success_count=success_count,
        success_rate_scaled=success_rate_scaled, success_rate_scale=SCALE_RATIO,
        p95_latency_ms=p95, throughput_kbps_scaled=throughput_scaled,
        throughput_scale=SCALE_KBPS, outcome_counts=outcome_counts,
        quantization=_quantization_block(),
    )


def terminal_attempts(records: list[AttemptRecord]) -> list[AttemptRecord]:
    """Keep only the last attempt of each (node, delivery): the delivery verdict.

    SLO success rate is defined over delivery terminal outcomes.  Retries are
    still fully recorded in the ledger and appear in the attempt-level outcome
    counts, but they do not inflate the SLO denominator.
    """
    last: dict[tuple[str, int], AttemptRecord] = {}
    for record in records:
        last[(record.node_id, record.delivery_seq)] = record
    return [last[key] for key in sorted(last)]


def aggregate(records: list[AttemptRecord], scene: Scene) -> dict[str, Any]:
    """Aggregate per-node and global blocks, preserving scene node order."""
    by_node: dict[str, list[AttemptRecord]] = {}
    for record in records:
        by_node.setdefault(record.node_id, []).append(record)
    terminals = terminal_attempts(records)
    terminal_by_node: dict[str, list[AttemptRecord]] = {}
    for record in terminals:
        terminal_by_node.setdefault(record.node_id, []).append(record)

    node_blocks: list[MetricBlock] = []
    for node in scene.nodes:  # stable scene order
        node_attempts = by_node.get(node.node_id, [])
        node_terminals = terminal_by_node.get(node.node_id, [])
        block = _aggregate("node", node.node_id, node_terminals,
                           scene.task.payload_bytes)
        # expose the raw attempt-level distribution (including retries) too
        attempt_block = _aggregate("attempts", node.node_id, node_attempts,
                                   scene.task.payload_bytes)
        payload = block.to_dict()
        payload["attempt_outcome_counts"] = attempt_block.outcome_counts
        payload["attempt_samples"] = attempt_block.samples
        node_blocks.append(payload)
    global_block = _aggregate("global", None, terminals, scene.task.payload_bytes)
    global_payload = global_block.to_dict()
    attempt_global = _aggregate("attempts", None, records, scene.task.payload_bytes)
    global_payload["attempt_outcome_counts"] = attempt_global.outcome_counts
    global_payload["attempt_samples"] = attempt_global.samples
    latency_values = np.array([r.elapsed_ms for r in records], dtype=np.int64)
    numpy_stats = {
        "mean_elapsed_ms": float(np.mean(latency_values)) if len(latency_values) else 0.0,
        "stdev_elapsed_ms": float(np.std(latency_values)) if len(latency_values) else 0.0,
    }
    return {
        "scene_version": scene.scene_version,
        "payload_digest": scene.payload_digest,
        "global": global_payload,
        "nodes": node_blocks,
        "numpy_stats": numpy_stats,
    }


def judge_slo(metrics: dict[str, Any], scene: Scene) -> dict[str, Any]:
    """Deterministic SLO judgment; repeats over the same input are byte-identical."""
    global_block = metrics["global"]
    # SLO threshold is an integer percent; convert to the same fixed-point scale.
    threshold_scaled = scene.slo.success_rate_percent * (SCALE_RATIO // 100)
    actual_scaled = global_block["success_rate_scaled"]
    p95_actual = global_block["p95_latency_ms"]
    p95_limit = scene.slo.p95_latency_ms

    reasons: list[dict[str, Any]] = []
    if actual_scaled < threshold_scaled:
        reasons.append({
            "type": "success_rate",
            "trigger_value_scaled": actual_scaled,
            "threshold_value_scaled": threshold_scaled,
            "scale": SCALE_RATIO,
            "quantization": _quantization_block()["success_rate"],
        })
    if p95_actual > p95_limit:
        reasons.append({
            "type": "p95_latency",
            "trigger_value_ms": p95_actual,
            "threshold_value_ms": p95_limit,
            "scale": SCALE_LATENCY,
            "quantization": _quantization_block()["p95_latency_ms"],
        })
    judgment = {
        "scene_version": scene.scene_version,
        "payload_digest": scene.payload_digest,
        "passed": not reasons,
        "reasons": reasons,
        "fixed_point": {
            "success_rate_scale": SCALE_RATIO,
            "throughput_scale": SCALE_KBPS,
            "latency_scale": SCALE_LATENCY,
            "truncation": "toward zero",
            "ratio_error_bound": RATIO_ERROR_BOUND,
            "kbps_error_bound": KBPS_ERROR_BOUND,
            "p95_error_bound": P95_ERROR_BOUND,
        },
    }
    return judgment


def canonical_judgment(judgment: dict[str, Any]) -> str:
    return canonical_json(judgment)
