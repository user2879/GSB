"""Profile-session orchestration and snapshot document construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aggregate import AggregationResult, aggregate_file, aggregate_samples
from .canonical import sha256_bytes
from .models import (
    CURRENT_METRIC_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    Interval,
    ProfilePolicy,
    Sample,
    metric_definition,
)
from .ndjson import NDJSONMetadata
from .ranking import build_ranking, canonical_core_bytes


@dataclass(frozen=True, slots=True)
class ProfileSession:
    interval: Interval
    policy: ProfilePolicy
    aggregation: AggregationResult
    ranking_report: dict[str, Any]

    @property
    def metadata(self) -> NDJSONMetadata | None:
        return self.aggregation.metadata

    @property
    def conclusion_sha256(self) -> str:
        return sha256_bytes(canonical_core_bytes(self.aggregation, self.ranking_report))

    def to_summary(self) -> dict[str, Any]:
        metadata = self.metadata
        return {
            "ok": True,
            "interval": self.interval.to_dict(),
            "aggregation_path": self.aggregation.selection.path,
            "forced_path": self.aggregation.selection.forced,
            "thresholds": {
                "unique_stack_count": self.aggregation.selection.unique_stack_count,
                "selected_total_weight": self.aggregation.selection.total_weight,
                "full_unique_stack_threshold": self.aggregation.selection.unique_threshold,
                "full_total_weight_threshold": self.aggregation.selection.weight_threshold,
                "rule": self.aggregation.selection.rule,
            },
            "peak_memory_estimate_bytes": self.aggregation.memory_estimate[
                "estimated_peak_bytes"
            ],
            "line_count": metadata.line_count if metadata else None,
            "selected_row_count": metadata.selected_row_count if metadata else None,
            "total_weight": self.aggregation.total_weight,
            "total_time_ms": self.aggregation.total_time_ms,
            "conclusion_sha256": self.conclusion_sha256,
            "top_hotspots": [
                {
                    "rank": item["rank"],
                    "frame": item["frame"],
                    "self_time_ms": item["self_time_ms"],
                    "cumulative_time_ms": item["cumulative_time_ms"],
                    "self_ratio_fixed": item["self_ratio_fixed"],
                    "error_bound_dominated": item[
                        "quantization_order_error_bound_dominated"
                    ],
                }
                for item in self.ranking_report["ranking"]
            ],
            "error_annotation": self.ranking_report["error_annotation"],
        }

    def to_snapshot_document(self) -> dict[str, Any]:
        metadata = self.metadata
        if metadata is None:
            raise ValueError("a persisted profile session requires file metadata")
        core = canonical_core_bytes(self.aggregation, self.ranking_report)
        folded_stacks = self.aggregation.folded_stacks
        document: dict[str, Any] = {
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "metric_version": CURRENT_METRIC_VERSION,
            "created_by": "slowprof",
            "interval": self.interval.to_dict(),
            "policy": self.policy.to_dict(),
            "sample": metadata.to_dict(),
            "aggregation": {
                **self.aggregation.selection.to_dict(),
                "memory_estimate": self.aggregation.memory_estimate,
                "stack_fold_table_included": folded_stacks is not None,
            },
            "folded_result": {
                "metric_definition": metric_definition(),
                "total_weight": self.aggregation.total_weight,
                "total_time_ms": self.aggregation.total_time_ms,
                "frame_count": len(self.aggregation.frame_stats),
                "frames": [stat.to_dict() for stat in self.aggregation.frame_stats],
                "stack_folds": (
                    [
                        {"frames": list(frames), "weight": weight}
                        for frames, weight in folded_stacks
                    ]
                    if folded_stacks is not None
                    else None
                ),
                "stack_folds_omitted_reason": (
                    None
                    if folded_stacks is not None
                    else "streaming path retains no full stack path strings"
                ),
            },
            "hotspot_ranking": self.ranking_report["ranking"],
            "ranking_rule": self.ranking_report["ranking_rule"],
            "requested_k": self.ranking_report["requested_k"],
            "reported_k": self.ranking_report["reported_k"],
            "error_annotation": self.ranking_report["error_annotation"],
            "conclusion_sha256": sha256_bytes(core),
        }
        return document


def profile_file(
    samples_path: str | Path,
    policy: ProfilePolicy,
    interval: Interval,
    *,
    force_path: str | None = None,
) -> ProfileSession:
    aggregation = aggregate_file(
        samples_path,
        policy,
        interval,
        force_path=force_path,
    )
    ranking_report = build_ranking(aggregation, policy)
    return ProfileSession(
        interval=interval,
        policy=policy,
        aggregation=aggregation,
        ranking_report=ranking_report,
    )


def profile_samples(
    samples: Sequence[Sample],
    policy: ProfilePolicy,
    interval: Interval,
    *,
    force_path: str | None = None,
) -> ProfileSession:
    selected = tuple(sample for sample in samples if interval.includes(sample.timestamp_ms))
    aggregation = aggregate_samples(selected, policy, path=force_path)
    ranking_report = build_ranking(aggregation, policy)
    return ProfileSession(
        interval=interval,
        policy=policy,
        aggregation=aggregation,
        ranking_report=ranking_report,
    )
