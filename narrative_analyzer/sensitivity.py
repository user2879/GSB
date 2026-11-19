"""Deterministic coefficient perturbation scanning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .canonical import sha256_value
from .engine import ReplayOptions, replay
from .errors import E_SPEC_INVALID, BusinessError
from .model import Spec


DEFAULT_RELATIVE_DELTAS = (
    1.0e-15,
    1.0e-12,
    1.0e-9,
    1.0e-6,
    1.0e-4,
    1.0e-2,
    1.0e-1,
)


@dataclass(frozen=True)
class ScanConfig:
    relative_deltas: tuple[float, ...] = DEFAULT_RELATIVE_DELTAS
    direction: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_deltas": list(self.relative_deltas),
            "direction": self.direction,
        }


def _scan_error(message: str, **details: Any) -> BusinessError:
    return BusinessError(E_SPEC_INVALID, message, details)


def parse_scan_config(value: Mapping[str, Any] | None) -> ScanConfig:
    if value is None:
        return ScanConfig()
    if not isinstance(value, dict):
        raise _scan_error("sensitivity configuration must be a mapping")
    unknown = sorted(str(key) for key in value if key not in {"version", "relative_deltas", "direction"})
    if unknown:
        raise _scan_error("sensitivity configuration contains unknown keys", unknown=unknown)
    if value.get("version", 1) != 1:
        raise _scan_error("unsupported sensitivity configuration version", version=value.get("version"))
    direction_value = value.get("direction", "increase")
    if direction_value in ("increase", "+", 1, "+1"):
        direction = 1
    elif direction_value in ("decrease", "-", -1, "-1"):
        direction = -1
    else:
        raise _scan_error("sensitivity direction must be increase or decrease", direction=direction_value)
    raw_deltas = value.get("relative_deltas", list(DEFAULT_RELATIVE_DELTAS))
    if not isinstance(raw_deltas, list) or not raw_deltas:
        raise _scan_error("relative_deltas must be a non-empty list")
    deltas: list[float] = []
    for index, raw in enumerate(raw_deltas):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise _scan_error("relative delta must be numeric", index=index)
        delta = float(raw)
        if not math.isfinite(delta) or delta <= 0.0:
            raise _scan_error("relative delta must be finite and positive", index=index)
        deltas.append(delta)
    if any(deltas[index] >= deltas[index + 1] for index in range(len(deltas) - 1)):
        raise _scan_error("relative_deltas must be strictly increasing")
    return ScanConfig(tuple(deltas), direction)


def _decision_key(decision: Mapping[str, Any]) -> str:
    return f"step[{decision['step_index']}]/{decision['edge_id']}/{decision['condition_id']}"


def _decision_snapshot(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reachable": decision["reachable"],
        "uncertain": decision["uncertain"],
        "signed_margin": decision["signed_margin"],
        "absolute_margin": decision["absolute_margin"],
        "margin_error_upper": decision["margin_error_upper"],
    }


def scan_sensitivity(
    spec: Spec,
    selections: tuple[str, ...],
    config: ScanConfig | None = None,
) -> dict[str, Any]:
    config = config or ScanConfig()
    baseline = replay(spec, selections, ReplayOptions(enforce_reachability=False))
    baseline_by_key = {_decision_key(decision): decision for decision in baseline["decisions"]}
    decision_order = [_decision_key(decision) for decision in baseline["decisions"]]
    coefficient_order = [coefficient.key for coefficient in spec.coefficients]
    coefficient_by_key = spec.coefficient_map()

    first_flips: dict[tuple[str, str], dict[str, Any]] = {}
    coefficient_results: list[dict[str, Any]] = []

    for coefficient_key in coefficient_order:
        coefficient = coefficient_by_key[coefficient_key]
        flips_for_coefficient: list[str] = []
        for delta in config.relative_deltas:
            factor = 1.0 + float(config.direction) * delta
            perturbed_value = coefficient.value * factor
            if not math.isfinite(perturbed_value):
                raise _scan_error(
                    "coefficient perturbation produced a non-finite value",
                    coefficient=coefficient_key,
                    relative_delta=delta,
                )
            options = ReplayOptions(
                coefficient_overrides={coefficient_key: perturbed_value},
                enforce_reachability=False,
            )
            run = replay(spec, selections, options)
            run_by_key = {_decision_key(decision): decision for decision in run["decisions"]}
            for decision_key in decision_order:
                pair_key = (coefficient_key, decision_key)
                if pair_key in first_flips:
                    continue
                before = baseline_by_key[decision_key]
                after = run_by_key.get(decision_key)
                if after is None:
                    continue
                if before["reachable"] != after["reachable"]:
                    first_flips[pair_key] = {
                        "coefficient": coefficient_key,
                        "edge_id": coefficient.edge_id,
                        "effect_index": coefficient.effect_index,
                        "base_coefficient_value": coefficient.value,
                        "perturbed_coefficient_value": perturbed_value,
                        "relative_delta": delta,
                        "direction": config.direction,
                        "decision_key": decision_key,
                        "condition_id": before["condition_id"],
                        "step_index": before["step_index"],
                        "before": _decision_snapshot(before),
                        "after": _decision_snapshot(after),
                    }
                    flips_for_coefficient.append(decision_key)
        coefficient_results.append(
            {
                "coefficient": coefficient_key,
                "edge_id": coefficient.edge_id,
                "effect_index": coefficient.effect_index,
                "base_value": coefficient.value,
                "inert": coefficient.value == 0.0,
                "flipped_decision_count": len(flips_for_coefficient),
            }
        )

    ranking = sorted(
        first_flips.values(),
        key=lambda item: (
            item["relative_delta"],
            coefficient_order.index(item["coefficient"]),
            decision_order.index(item["decision_key"]),
        ),
    )
    for index, item in enumerate(ranking, start=1):
        item["rank"] = index

    return {
        "business_code": "OK",
        "spec_sha256": spec.source_sha256,
        "selections": list(selections),
        "scan_config": config.to_dict(),
        "scan_config_sha256": sha256_value(config.to_dict()),
        "baseline_content_hash": baseline["content_hash"],
        "coefficient_count": len(coefficient_order),
        "coefficient_results": coefficient_results,
        "critical_count": len(ranking),
        "ranking": ranking,
    }

