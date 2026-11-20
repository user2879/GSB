from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

from .canonical import fraction_json, sha256_json
from .spec import SamplingPolicy


def stable_bucket(client_key: str, strategy_version: int, scale_power: int) -> int:
    payload = (
        f"trace-diag-partition-v1\x1f{strategy_version}\x1f{scale_power}\x1f{client_key}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 10_000


@dataclass(frozen=True)
class SampleDecision:
    record_index: int
    bucket: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_index": self.record_index,
            "bucket": self.bucket,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SamplingResult:
    decisions: tuple[SampleDecision, ...]
    summary: dict[str, Any]

    @property
    def hits(self) -> frozenset[int]:
        return frozenset(
            decision.record_index
            for decision in self.decisions
            if decision.reason != "skip"
        )


def sample_records(
    records: Sequence[dict[str, Any]],
    policy: SamplingPolicy,
    strategy_version: int,
) -> SamplingResult:
    decisions: list[SampleDecision] = []
    head_hits = 0
    tail_hits = 0
    for index, record in enumerate(records):
        bucket = stable_bucket(record["client_key"], strategy_version, policy.scale_power)
        if bucket < policy.head_ratio_bp:
            reason = "head"
            head_hits += 1
        elif (
            record["duration_us"] >= policy.tail_duration_threshold_us
            and bucket < policy.tail_ratio_bp
        ):
            reason = "tail"
            tail_hits += 1
        else:
            reason = "skip"
        decisions.append(SampleDecision(record_index=index, bucket=bucket, reason=reason))

    hit_rows = [
        {
            "record_index": decision.record_index,
            "trace_id": records[decision.record_index]["trace_id"],
            "reason": decision.reason,
            "bucket": decision.bucket,
        }
        for decision in decisions
        if decision.reason != "skip"
    ]
    sample_hits = head_hits + tail_hits
    summary = {
        "strategy_version": strategy_version,
        "scale_power": policy.scale_power,
        "head_ratio_bp": policy.head_ratio_bp,
        "tail_ratio_bp": policy.tail_ratio_bp,
        "tail_duration_threshold_us": policy.tail_duration_threshold_us,
        "head_hits": head_hits,
        "tail_hits": tail_hits,
        "sample_hits": sample_hits,
        "skipped": len(records) - sample_hits,
        "sample_ratio": fraction_json(Fraction(sample_hits, len(records))),
        "decision_digest": sha256_json([decision.to_dict() for decision in decisions]),
        "sample_set_digest": sha256_json(hit_rows),
        "sample_trace_ids": [row["trace_id"] for row in hit_rows],
        "partition_rule": "sha256_8_bytes_mod_10000; bucket < ratio_bp",
        "tail_eligible_rule": "not head and duration_us >= threshold",
    }
    return SamplingResult(decisions=tuple(decisions), summary=summary)
