from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable

from .canonical import canonical_json, sha256_hex
from .fixedpoint import (
    ceil_units,
    fixed_to_decimal_text,
    midpoint_fixed,
    safe_weight_sum,
    weighted_fixed_mean,
)
from .spec import KeyRule, ViewSpec


@dataclass(frozen=True)
class Candidate:
    env_id: str
    slice_version: int
    aggregate_units: int
    weight_units: int
    value_quantization_error_units: Decimal
    weight_quantization_error_units: Decimal
    receipt_digest: str
    sample_digest: str

    def sort_key(self) -> tuple[str, int]:
        return (self.env_id, self.slice_version)


@dataclass(frozen=True)
class ModeOutcome:
    mode: str
    selected_units: int
    adopted: tuple[str, ...]
    rejected: tuple[str, ...]
    tie_break: str
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "selected_units": self.selected_units,
            "adopted": list(self.adopted),
            "rejected": list(self.rejected),
            "tie_break": self.tie_break,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class KeyResult:
    view_id: str
    key: str
    mode: str
    scale_exponent: int
    summary_units: int | None
    summary_decimal: str | None
    summary_digest: str
    adopted: tuple[str, ...]
    rejected: tuple[str, ...]
    slice_versions: dict[str, int]
    error_bound: dict[str, Any]
    attribution: dict[str, Any]
    mode_values: dict[str, int | None]
    tie_break: str
    status: str
    error: dict[str, Any] | None
    payload: dict[str, Any]

    @property
    def adopted_json(self) -> str:
        return canonical_json(list(self.adopted))

    @property
    def rejected_json(self) -> str:
        return canonical_json(list(self.rejected))

    @property
    def bound_json(self) -> str:
        return canonical_json(self.error_bound)

    @property
    def attribution_json(self) -> str:
        return canonical_json(self.attribution)

    @property
    def slice_versions_json(self) -> str:
        return canonical_json(dict(sorted(self.slice_versions.items())))

    @property
    def result_json(self) -> str:
        return canonical_json(self.payload)

    @property
    def result_digest(self) -> str:
        return sha256_hex(self.result_json)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class ViewSummary:
    view_id: str
    version: int
    key_results: tuple[KeyResult, ...]
    error_bound: dict[str, Any]
    attribution: dict[str, Any]
    summary_digest: str

    def by_key(self) -> dict[str, KeyResult]:
        return {item.key: item for item in self.key_results}

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "version": self.version,
            "summary_digest": self.summary_digest,
            "error_bound": self.error_bound,
            "attribution": self.attribution,
            "keys": [item.to_dict() for item in self.key_results],
        }


def _ordered_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=Candidate.sort_key)


def _latest(candidates: list[Candidate], tolerance: int) -> ModeOutcome:
    ordered = sorted(candidates, key=lambda item: (-item.slice_version, item.env_id))
    winner = ordered[0]
    adopted = (winner.env_id,)
    rejected = tuple(item.env_id for item in ordered[1:])
    return ModeOutcome(
        mode="latest",
        selected_units=winner.aggregate_units,
        adopted=adopted,
        rejected=rejected,
        tie_break="highest slice_version, then lexicographically smallest environment id",
        detail={
            "winner_slice_version": winner.slice_version,
            "tolerance_units": tolerance,
        },
    )


def _weighted_clusters(candidates: list[Candidate], tolerance: int) -> list[list[Candidate]]:
    ordered = sorted(candidates, key=lambda item: (item.aggregate_units, item.env_id))
    clusters: list[list[Candidate]] = []
    for candidate in ordered:
        if not clusters:
            clusters.append([candidate])
            continue
        anchor = clusters[-1][0].aggregate_units
        if candidate.aggregate_units - anchor <= tolerance:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    return clusters


def _weighted_majority(candidates: list[Candidate], tolerance: int) -> ModeOutcome:
    ranked: list[tuple[int, int, tuple[str, ...], list[Candidate]]] = []
    for cluster in _weighted_clusters(candidates, tolerance):
        support = safe_weight_sum(item.weight_units for item in cluster)
        selected = weighted_fixed_mean(
            (item.weight_units, item.aggregate_units) for item in cluster
        )
        envs = tuple(sorted(item.env_id for item in cluster))
        ranked.append((support, selected, envs, cluster))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    support, selected, envs, winner_cluster = ranked[0]
    adopted = set(envs)
    rejected = tuple(
        sorted(item.env_id for item in candidates if item.env_id not in adopted)
    )
    return ModeOutcome(
        mode="weighted_majority",
        selected_units=selected,
        adopted=envs,
        rejected=rejected,
        tie_break="highest quantized weight support, then higher weighted value, then smallest environment tuple",
        detail={
            "winning_weight_support": support,
            "tolerance_units": tolerance,
            "cluster_span_units": max(
                item.aggregate_units for item in winner_cluster
            )
            - min(item.aggregate_units for item in winner_cluster),
        },
    )


def _interval_midpoint(candidates: list[Candidate], tolerance: int) -> ModeOutcome:
    ordered = sorted(candidates, key=lambda item: (item.aggregate_units, item.env_id))
    intervals = [
        (
            item.aggregate_units - tolerance,
            item.aggregate_units + tolerance,
            item,
        )
        for item in ordered
    ]
    endpoints = sorted({low for low, _, _ in intervals})
    ranked: list[tuple[int, int, int, tuple[str, ...], list[Candidate], int, int]] = []
    for point in endpoints:
        active = [item for low, high, item in intervals if low <= point <= high]
        if not active:
            continue
        support = safe_weight_sum(item.weight_units for item in active)
        intersection_low = max(item.aggregate_units - tolerance for item in active)
        intersection_high = min(item.aggregate_units + tolerance for item in active)
        width = intersection_high - intersection_low
        envs = tuple(sorted(item.env_id for item in active))
        ranked.append(
            (support, -width, -intersection_low, envs, active, intersection_low, intersection_high)
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    support, negative_width, _, envs, active, low, high = ranked[0]
    selected = midpoint_fixed(low, high)
    adopted_set = set(envs)
    rejected = tuple(
        sorted(item.env_id for item in candidates if item.env_id not in adopted_set)
    )
    return ModeOutcome(
        mode="interval_midpoint",
        selected_units=selected,
        adopted=envs,
        rejected=rejected,
        tie_break="highest interval overlap support, then narrower intersection, then lower bound, then smallest environment tuple",
        detail={
            "intersection_low_units": low,
            "intersection_high_units": high,
            "intersection_width_units": -negative_width,
            "winning_weight_support": support,
        },
    )


def _mode_value_map(outcomes: dict[str, ModeOutcome]) -> dict[str, int | None]:
    return {
        "latest": outcomes["latest"].selected_units,
        "weighted_majority": outcomes["weighted_majority"].selected_units,
        "interval_midpoint": outcomes["interval_midpoint"].selected_units,
    }


def _bound_breakdown(
    *,
    candidates: list[Candidate],
    tolerance: int,
    mode_values: dict[str, int | None],
) -> tuple[dict[str, Any], int, int | None, bool]:
    max_value_error = max(
        (candidate.value_quantization_error_units for candidate in candidates),
        default=Decimal(0),
    )
    max_weight_error = max(
        (candidate.weight_quantization_error_units for candidate in candidates),
        default=Decimal(0),
    )
    value_quantization_units = max(
        (ceil_units(candidate.value_quantization_error_units) for candidate in candidates),
        default=0,
    )
    weight_quantization_units = max(
        (ceil_units(candidate.weight_quantization_error_units) for candidate in candidates),
        default=0,
    )
    quantization_units = value_quantization_units + weight_quantization_units
    numeric_values = [value for value in mode_values.values() if value is not None]
    mode_difference = (max(numeric_values) - min(numeric_values)) if numeric_values else 0
    raw_bound = quantization_units + mode_difference
    absorbed = min(tolerance, raw_bound)
    decision_bound = raw_bound - absorbed
    bound = {
        "value_quantization_error_units": str(max_value_error),
        "weight_quantization_error_units": str(max_weight_error),
        "quantization_truncation_units": quantization_units,
        "arbitration_mode_difference_units": mode_difference,
        "tolerance_absorption_units": -absorbed,
        "decision_bound_units": decision_bound,
        "raw_bound_before_tolerance_units": raw_bound,
        "tolerance_units": tolerance,
        "truncation_direction": "toward_zero",
        "overflow_rule": "saturate_to_signed_int64",
    }
    return bound, decision_bound, mode_difference, False


def _primary_cause(
    *,
    error_indistinguishable: bool,
    mode_conflict: bool,
    rejected_ratio_high: bool,
) -> str:
    if error_indistinguishable:
        return "ERROR_INDISTINGUISHABLE"
    if mode_conflict:
        return "MODE_CONFLICT"
    if rejected_ratio_high:
        return "REJECTED_RATIO_HIGH"
    return "NONE"


def arbitrate_key(
    *,
    view: ViewSpec,
    key: str,
    rule: KeyRule,
    candidates: Iterable[Candidate],
    scale_exponent: int,
    source_errors: list[dict[str, Any]] | None = None,
) -> KeyResult:
    ordered = _ordered_candidates(candidates)
    if not ordered:
        errors = source_errors or [
            {
                "code": "E_FIELD",
                "field": "candidates",
                "message": "no candidate environment reported this key",
            }
        ]
        payload = {
            "schema": "config-authority/v1",
            "view_id": view.id,
            "key": key,
            "mode": rule.mode,
            "status": "ERROR",
            "error": errors,
            "summary": {"units": None, "decimal": None},
            "adopted": [],
            "rejected": [],
            "slice_versions": {},
            "error_bound": {
                "quantization_truncation_units": 0,
                "arbitration_mode_difference_units": 0,
                "tolerance_absorption_units": 0,
                "decision_bound_units": 0,
                "tolerance_units": rule.tolerance,
                "truncation_direction": "toward_zero",
                "overflow_rule": "saturate_to_signed_int64",
            },
            "attribution": {
                "primary_cause": "SERVICE_UNAVAILABLE",
                "cause_counts": {"ERROR_INDISTINGUISHABLE": 0, "MODE_CONFLICT": 0, "REJECTED_RATIO_HIGH": 0},
            },
            "mode_values": {"latest": None, "weighted_majority": None, "interval_midpoint": None},
            "tie_break": "none",
        }
        return KeyResult(
            view_id=view.id,
            key=key,
            mode=rule.mode,
            scale_exponent=scale_exponent,
            summary_units=None,
            summary_decimal=None,
            summary_digest=sha256_hex(payload),
            adopted=(),
            rejected=(),
            slice_versions={},
            error_bound=payload["error_bound"],
            attribution=payload["attribution"],
            mode_values=payload["mode_values"],
            tie_break="none",
            status="ERROR",
            error=errors[0],
            payload=payload,
        )

    outcomes = {
        "latest": _latest(ordered, rule.tolerance),
        "weighted_majority": _weighted_majority(ordered, rule.tolerance),
        "interval_midpoint": _interval_midpoint(ordered, rule.tolerance),
    }
    configured = outcomes[rule.mode]
    mode_values = _mode_value_map(outcomes)
    bound, decision_bound, mode_difference, _ = _bound_breakdown(
        candidates=ordered,
        tolerance=rule.tolerance,
        mode_values=mode_values,
    )

    rejected_candidates = [
        item for item in ordered if item.env_id not in set(configured.adopted)
    ]
    if rejected_candidates:
        runner_delta: int | None = min(
            abs(item.aggregate_units - configured.selected_units)
            for item in rejected_candidates
        )
    else:
        runner_delta = None
    error_indistinguishable = (
        runner_delta is not None and runner_delta <= decision_bound
    )
    mode_conflict = len(set(mode_values.values())) > 1
    rejected_ratio = len(configured.rejected) / len(ordered)
    rejected_ratio_high = len(configured.rejected) * 2 > len(ordered)
    primary_cause = _primary_cause(
        error_indistinguishable=error_indistinguishable,
        mode_conflict=mode_conflict,
        rejected_ratio_high=rejected_ratio_high,
    )
    attribution = {
        "primary_cause": primary_cause,
        "error_indistinguishable": error_indistinguishable,
        "mode_conflict": mode_conflict,
        "rejected_ratio_high": rejected_ratio_high,
        "rejected_ratio": f"{rejected_ratio:.6f}",
        "rejected_environment_count": len(configured.rejected),
        "candidate_environment_count": len(ordered),
        "runner_up_delta_units": runner_delta,
        "mode_value_spread_units": mode_difference,
    }
    bound["runner_up_delta_units"] = runner_delta
    bound["indistinguishable"] = error_indistinguishable
    slice_versions = {
        item.env_id: item.slice_version for item in sorted(ordered, key=lambda item: item.env_id)
    }
    sources = {
        item.env_id: {
            "aggregate_units": item.aggregate_units,
            "slice_version": item.slice_version,
            "receipt_digest": item.receipt_digest,
            "sample_digest": item.sample_digest,
        }
        for item in sorted(ordered, key=lambda item: item.env_id)
    }
    payload = {
        "schema": "config-authority/v1",
        "view_id": view.id,
        "key": key,
        "mode": rule.mode,
        "status": "OK",
        "affected_by_error": error_indistinguishable,
        "scale_exponent": scale_exponent,
        "summary": {
            "units": configured.selected_units,
            "decimal": fixed_to_decimal_text(configured.selected_units, scale_exponent),
        },
        "adopted": list(configured.adopted),
        "rejected": list(configured.rejected),
        "slice_versions": slice_versions,
        "sources": sources,
        "error_bound": bound,
        "attribution": attribution,
        "mode_values": mode_values,
        "tie_break": configured.tie_break,
        "mode_outcomes": {name: outcome.to_dict() for name, outcome in sorted(outcomes.items())},
        "source_errors": source_errors or [],
    }
    summary_digest = sha256_hex(payload)
    payload["authoritative_digest"] = summary_digest
    return KeyResult(
        view_id=view.id,
        key=key,
        mode=rule.mode,
        scale_exponent=scale_exponent,
        summary_units=configured.selected_units,
        summary_decimal=fixed_to_decimal_text(configured.selected_units, scale_exponent),
        summary_digest=summary_digest,
        adopted=configured.adopted,
        rejected=configured.rejected,
        slice_versions=dict(sorted(slice_versions.items())),
        error_bound=bound,
        attribution=attribution,
        mode_values=mode_values,
        tie_break=configured.tie_break,
        status="OK",
        error=None,
        payload=payload,
    )


def summarize_view(view_id: str, version: int, key_results: Iterable[KeyResult]) -> ViewSummary:
    ordered = tuple(sorted(key_results, key=lambda item: item.key))
    quant = sum(
        int(item.error_bound.get("quantization_truncation_units", 0)) for item in ordered
    )
    mode_difference = sum(
        int(item.error_bound.get("arbitration_mode_difference_units", 0))
        for item in ordered
    )
    tolerance_absorption = sum(
        int(item.error_bound.get("tolerance_absorption_units", 0)) for item in ordered
    )
    decision_bound = sum(
        int(item.error_bound.get("decision_bound_units", 0)) for item in ordered
    )
    causes = {
        "ERROR_INDISTINGUISHABLE": 0,
        "MODE_CONFLICT": 0,
        "REJECTED_RATIO_HIGH": 0,
        "NONE": 0,
    }
    for item in ordered:
        cause = str(item.attribution.get("primary_cause", "NONE"))
        causes[cause] = causes.get(cause, 0) + 1
    total = len(ordered)
    shares = {
        cause: f"{(count / total):.6f}" if total else "0.000000"
        for cause, count in sorted(causes.items())
    }
    percentages = {
        cause: f"{(count / total * 100):.2f}%" if total else "0.00%"
        for cause, count in sorted(causes.items())
    }
    view_bound = {
        "quantization_truncation_units": quant,
        "arbitration_mode_difference_units": mode_difference,
        "tolerance_absorption_units": tolerance_absorption,
        "decision_bound_units": decision_bound,
        "aggregation_rule": "sum_of_absolute_key_bounds",
    }
    attribution = {
        "cause_counts": dict(sorted(causes.items())),
        "cause_shares": shares,
        "cause_percentages": percentages,
        "key_count": total,
    }
    digest_payload = {
        "view_id": view_id,
        "version": version,
        "keys": {item.key: item.summary_digest for item in ordered},
        "error_bound": view_bound,
        "attribution": attribution,
    }
    return ViewSummary(
        view_id=view_id,
        version=version,
        key_results=ordered,
        error_bound=view_bound,
        attribution=attribution,
        summary_digest=sha256_hex(digest_payload),
    )
