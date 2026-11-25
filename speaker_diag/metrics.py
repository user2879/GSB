"""Fixed-point diagnostics, stable decisions, and attribution error causes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from .canonical import floor_ratio
from .receipts import NormalizedReceipt
from .spec import WorkloadSpec
from .synthetic import SyntheticBundle


@dataclass(frozen=True)
class FixedPointPolicy:
    scale: int
    truncation: str = "floor_non_negative"
    configured_batch_score_bound_units: int = 1
    configured_worker_watermark_bound_units: int = 1

    @property
    def single_value_error_bound(self) -> Fraction:
        return Fraction(1, self.scale)

    @property
    def theoretical_comparison_bound(self) -> Fraction:
        return Fraction(2, self.scale)

    def batch_score_guard_units(self) -> int:
        return max(2, self.configured_batch_score_bound_units)

    def worker_watermark_guard_units(self) -> int:
        return max(2, self.configured_worker_watermark_bound_units)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "truncation": self.truncation,
            "single_value_error_bound_units": 1,
            "single_value_error_bound": f"<1/{self.scale}",
            "comparison_error_bound_units": 2,
            "comparison_error_bound": f"<2/{self.scale}",
            "configured_batch_score_guard_units": self.batch_score_guard_units(),
            "configured_worker_watermark_guard_units": self.worker_watermark_guard_units(),
            "minimum_distinguishable_difference_units": 3,
            "interval_convention": "half-open floor cell [k/s, (k+1)/s)",
        }


@dataclass(frozen=True)
class ScoredCandidate:
    key: str
    score_q: int
    kind: str = "batch"


@dataclass(frozen=True)
class StableDecision:
    selected_key: str
    best_key: str
    second_key: str | None
    best_score_q: int
    second_score_q: int | None
    score_gap_q: int | None
    error_bound_q: int
    affected_by_error_bound: bool
    rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_key": self.selected_key,
            "best_key": self.best_key,
            "second_key": self.second_key,
            "best_score_q": self.best_score_q,
            "second_score_q": self.second_score_q,
            "score_gap_q": self.score_gap_q,
            "error_bound_q": self.error_bound_q,
            "affected_by_error_bound": self.affected_by_error_bound,
            "rule": self.rule,
        }


def ratio_q(numerator: int, denominator: int, scale: int) -> int:
    return floor_ratio(numerator, denominator, scale)


def stable_choose(candidates: Sequence[ScoredCandidate], *, error_bound_q: int) -> StableDecision:
    if not candidates:
        raise ValueError("at least one candidate is required")
    ordered = sorted(candidates, key=lambda item: (-item.score_q, item.key, item.kind))
    best = ordered[0]
    if len(ordered) == 1:
        return StableDecision(
            selected_key=best.key,
            best_key=best.key,
            second_key=None,
            best_score_q=best.score_q,
            second_score_q=None,
            score_gap_q=None,
            error_bound_q=error_bound_q,
            affected_by_error_bound=False,
            rule="single_candidate",
        )
    second = ordered[1]
    gap = best.score_q - second.score_q
    near_best = [candidate for candidate in ordered if best.score_q - candidate.score_q <= error_bound_q]
    affected = len(near_best) > 1
    if affected:
        selected = sorted(near_best, key=lambda item: (item.key, item.kind))[0]
        rule = f"guard_band<={error_bound_q};lexicographic_{selected.kind}_key"
    else:
        selected = best
        rule = "strict_score_gap_exceeds_guard_band"
    return StableDecision(
        selected_key=selected.key,
        best_key=best.key,
        second_key=second.key,
        best_score_q=best.score_q,
        second_score_q=second.score_q,
        score_gap_q=gap,
        error_bound_q=error_bound_q,
        affected_by_error_bound=affected,
        rule=rule,
    )


def batch_scores(receipts: Mapping[str, NormalizedReceipt]) -> list[ScoredCandidate]:
    candidates: list[ScoredCandidate] = []
    for batch_id in sorted(receipts):
        receipt = receipts[batch_id]
        values = [
            int(window["confidence_q12"])
            for window in receipt.windows
            if type(window.get("confidence_q12")) is int
        ]
        score = sum(values) // len(values) if values else 0
        candidates.append(ScoredCandidate(key=batch_id, score_q=score, kind="batch"))
    return candidates


def batch_decision_table(receipts: Mapping[str, NormalizedReceipt], policy: FixedPointPolicy) -> dict[str, Any]:
    candidates = batch_scores(receipts)
    if not candidates:
        return {"cmp_schema": "speaker-diag-batch-decisions-v1", "candidates": [], "pairwise": []}
    bound = policy.batch_score_guard_units()
    grouped: dict[str, list[ScoredCandidate]] = {}
    for candidate in candidates:
        worker_id = receipts[candidate.key].worker_id
        grouped.setdefault(worker_id, []).append(candidate)
    decisions = []
    for worker_id in sorted(grouped):
        decision = stable_choose(grouped[worker_id], error_bound_q=bound)
        decisions.append(
            {
                "worker_id": worker_id,
                "decision": decision.to_dict(),
                "candidates": [
                    {"batch_id": candidate.key, "score_q": candidate.score_q, "elapsed_q": receipts[candidate.key].data["elapsed_ms"] * policy.scale}
                    for candidate in sorted(grouped[worker_id], key=lambda item: item.key)
                ],
            }
        )
    pairwise = []
    for index, left in enumerate(sorted(candidates, key=lambda item: item.key)):
        for right in sorted(candidates, key=lambda item: item.key)[index + 1 :]:
            decision = stable_choose([left, right], error_bound_q=bound)
            pairwise.append(
                {
                    "batch_a": left.key,
                    "batch_b": right.key,
                    **decision.to_dict(),
                }
            )
    return {
        "schema": "speaker-diag-batch-decisions-v1",
        "error_bound_q": bound,
        "candidates": [
            {
                "batch_id": candidate.key,
                "worker_id": receipts[candidate.key].worker_id,
                "score_q": candidate.score_q,
                "elapsed_q": receipts[candidate.key].data["elapsed_ms"] * policy.scale,
            }
            for candidate in sorted(candidates, key=lambda item: item.key)
        ],
        "group_decisions": decisions,
        "pairwise": pairwise,
        "stable_tie_break": "lexicographically smaller batch_id",
    }


def worker_watermark_decisions(lease_snapshot: Mapping[str, Any], policy: FixedPointPolicy) -> dict[str, Any]:
    workers = sorted(lease_snapshot.get("workers", []), key=lambda item: item["worker_id"])
    bound = policy.worker_watermark_guard_units()
    candidates = [
        ScoredCandidate(
            key=worker["worker_id"],
            score_q=int(worker["lease_watermark_q"]),
            kind="worker",
        )
        for worker in workers
    ]
    pairwise = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            decision = stable_choose([left, right], error_bound_q=bound)
            pairwise.append(
                {
                    "worker_a": left.key,
                    "worker_b": right.key,
                    "watermark_a_q": left.score_q,
                    "watermark_b_q": right.score_q,
                    **decision.to_dict(),
                }
            )
    group_decision = stable_choose(candidates, error_bound_q=bound).to_dict() if candidates else None
    return {
        "schema": "speaker-diag-worker-watermark-decisions-v1",
        "error_bound_q": bound,
        "workers": [
            {
                "worker_id": worker["worker_id"],
                "lease_watermark_q": worker["lease_watermark_q"],
                "concurrency_watermark_q": worker["concurrency_watermark_q"],
                "active_leases": worker["active_leases"],
                "active_concurrency": worker["active_concurrency"],
            }
            for worker in workers
        ],
        "group_decision": group_decision,
        "pairwise": pairwise,
        "stable_tie_break": "lexicographically smaller worker_id",
    }


def _proportions_bps(counts: Mapping[str, int], order: Sequence[str]) -> dict[str, int]:
    total = sum(counts.values())
    result = {key: 0 for key in order}
    if total <= 0:
        return result
    remainders: list[tuple[int, str]] = []
    used = 0
    for key in order:
        numerator = counts.get(key, 0) * 10_000
        base, remainder = divmod(numerator, total)
        result[key] = base
        used += base
        remainders.append((remainder, key))
    remaining = 10_000 - used
    for _remainder, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        result[key] += 1
    return result


def attribution_diagnostics(
    receipts: Mapping[str, NormalizedReceipt],
    bundle: SyntheticBundle,
    spec: WorkloadSpec,
    policy: FixedPointPolicy,
) -> dict[str, Any]:
    cause_order = ["quantization_truncation", "overlap_attribution", "batch_reorder"]
    cause_counts = {cause: 0 for cause in cause_order}
    deliveries = 0
    hits = 0
    unique_by_window: dict[str, tuple[str, bool]] = {}
    samples: list[dict[str, Any]] = []
    for batch_id in sorted(receipts, key=lambda item: spec.batch_order_key(item)):
        receipt = receipts[batch_id]
        request = spec.request_by_batch(batch_id)
        canonical_rank = {window_id: index + 1 for index, window_id in enumerate(request.window_ids)}
        for window in sorted(receipt.windows, key=lambda item: item["rank"]):
            window_id = str(window["window_id"])
            truth = bundle.truth_for(window_id)
            predicted = str(window["attribution"])
            hit = predicted == truth.truth_speaker
            deliveries += 1
            if hit:
                hits += 1
            unique_by_window.setdefault(window_id, (predicted, hit))
            if hit:
                continue
            margin = window.get("score_margin")
            if truth.overlap_bps > 0 or len(truth.active_speakers) > 1:
                cause = "overlap_attribution"
            elif window["rank"] != canonical_rank[window_id]:
                cause = "batch_reorder"
            elif type(margin) is int and margin <= policy.batch_score_guard_units():
                cause = "quantization_truncation"
            else:
                cause = "quantization_truncation"
            cause_counts[cause] += 1
            if len(samples) < 50:
                samples.append(
                    {
                        "batch_id": batch_id,
                        "window_id": window_id,
                        "predicted": predicted,
                        "truth": truth.truth_speaker,
                        "active_speakers": list(truth.active_speakers),
                        "overlap_bps": truth.overlap_bps,
                        "rank": window["rank"],
                        "canonical_rank": canonical_rank[window_id],
                        "score_margin": margin,
                        "cause": cause,
                    }
                )
    unique_hits = sum(1 for _predicted, hit in unique_by_window.values() if hit)
    errors = deliveries - hits
    return {
        "schema": "speaker-diag-attribution-metrics-v1",
        "fixed_point": policy.to_dict(),
        "delivery_counts": {
            "windows": deliveries,
            "hits": hits,
            "errors": errors,
            "hit_rate_q": ratio_q(hits, deliveries, policy.scale) if deliveries else 0,
            "error_rate_q": ratio_q(errors, deliveries, policy.scale) if deliveries else 0,
        },
        "unique_window_counts": {
            "windows": len(unique_by_window),
            "hits": unique_hits,
            "errors": len(unique_by_window) - unique_hits,
            "hit_rate_q": ratio_q(unique_hits, len(unique_by_window), policy.scale)
            if unique_by_window
            else 0,
        },
        "error_causes": {
            "denominator": "misattributed_window_deliveries",
            "total_errors": errors,
            "counts": cause_counts,
            "proportion_bps": _proportions_bps(cause_counts, cause_order),
            "truncation_direction": "largest_remainder_to_sum_10000_bps;category_lexicographic_tie_break",
        },
        "batch_elapsed_q": {
            batch_id: {
                "elapsed_ms": receipts[batch_id].data["elapsed_ms"],
                "elapsed_q": receipts[batch_id].data["elapsed_ms"] * policy.scale,
                "unit": f"millisecond*{policy.scale}",
            }
            for batch_id in sorted(receipts, key=lambda item: spec.batch_order_key(item))
        },
        "sample_errors": samples,
    }


def build_metrics_report(
    receipts: Mapping[str, NormalizedReceipt],
    bundle: SyntheticBundle,
    spec: WorkloadSpec,
    lease_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    policy = FixedPointPolicy(
        scale=spec.limits.metric_scale,
        configured_batch_score_bound_units=spec.limits.batch_score_error_bound_units,
        configured_worker_watermark_bound_units=spec.limits.worker_watermark_error_bound_units,
    )
    return {
        "schema": "speaker-diag-metrics-v1",
        "fixed_point_policy": policy.to_dict(),
        "batch_decisions": batch_decision_table(receipts, policy),
        "worker_watermark_decisions": worker_watermark_decisions(lease_snapshot, policy),
        "attribution": attribution_diagnostics(receipts, bundle, spec, policy),
    }
