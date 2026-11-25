from __future__ import annotations

import time
from typing import Any

import numpy as np

from .canonical import content_digest
from .errors import FieldError
from .spec import NodeSpec


def aggregate_samples(
    samples: dict[str, Any],
    node: NodeSpec,
    revision_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    revision_config = revision_config or {}
    multiplier = int(revision_config.get("multiplier", 1))
    group_count_override = revision_config.get("group_count")

    groups = np.asarray(samples["groups"], dtype=np.int64)
    values = np.asarray(samples["values_q"], dtype=np.int64) * np.int64(multiplier)
    if groups.shape != values.shape or groups.ndim != 1:
        raise FieldError("sample groups and values_q must be equal-length 1-D arrays", field="samples")

    unique_groups = np.unique(groups)
    if group_count_override is not None:
        max_groups = max(1, int(group_count_override))
        unique_groups = unique_groups[unique_groups < max_groups]

    grouped: list[dict[str, int]] = []
    for group in unique_groups:
        mask = groups == group
        group_values = values[mask]
        grouped.append(
            {
                "group": int(group),
                "row_count": int(group_values.size),
                "sum_q": int(group_values.sum(dtype=np.int64)),
                "min_q": int(group_values.min()),
                "max_q": int(group_values.max()),
            }
        )

    summary = {
        "algorithm": "numpy_grouped_fixed_point_v1",
        "fixed_point_scale": int(node.fixed_point_scale),
        "multiplier": multiplier,
        "total_rows": int(values.size),
        "total_sum_q": int(values.sum(dtype=np.int64)),
        "groups": grouped,
    }
    return summary, content_digest(summary)


def timed_aggregate(
    samples: dict[str, Any],
    node: NodeSpec,
    revision_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, float]:
    started_ns = time.monotonic_ns()
    summary, digest = aggregate_samples(samples, node, revision_config)
    elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
    return summary, digest, round(elapsed_ms, 3)


RECEIPT_REQUIRED_FIELDS = (
    "type",
    "node_id",
    "shard_id",
    "revision",
    "summary",
    "summary_digest",
    "elapsed_ms",
)


def validate_receipt(payload: dict[str, Any], *, expected_node: str, expected_shard: str, expected_revision: int) -> None:
    for field_name in RECEIPT_REQUIRED_FIELDS:
        if field_name not in payload:
            raise FieldError(f"receipt is missing field {field_name!r}", path="$", field=field_name)
    if payload.get("node_id") != expected_node:
        raise FieldError(
            f"receipt node_id mismatch: expected {expected_node}, got {payload.get('node_id')!r}",
            field="node_id",
        )
    if payload.get("shard_id") != expected_shard:
        raise FieldError(
            f"receipt shard_id mismatch: expected {expected_shard}, got {payload.get('shard_id')!r}",
            field="shard_id",
        )
    if payload.get("revision") != expected_revision:
        raise FieldError(
            f"receipt revision mismatch: expected {expected_revision}, got {payload.get('revision')!r}",
            field="revision",
        )
    if not isinstance(payload.get("summary"), dict):
        raise FieldError("receipt summary must be an object", field="summary")
    if not isinstance(payload.get("summary_digest"), str) or len(payload["summary_digest"]) != 64:
        raise FieldError("receipt summary_digest must be a SHA-256 hex string", field="summary_digest")
    if not isinstance(payload.get("elapsed_ms"), (int, float)) or isinstance(payload.get("elapsed_ms"), bool):
        raise FieldError("receipt elapsed_ms must be a number", field="elapsed_ms")
    recomputed = content_digest(payload["summary"])
    if recomputed != payload["summary_digest"]:
        raise FieldError("receipt summary_digest does not match summary", field="summary_digest")
