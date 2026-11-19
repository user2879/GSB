from __future__ import annotations

import math
from typing import Any

from .errors import SyncError


STRATEGIES = ("full_exact", "block_accumulation", "sparse_anchor_interpolation")


def _resource_models(
    duration_s: float,
    pulse_count: int,
    config: dict[str, Any],
    channel_count: int = 7,
) -> dict[str, dict[str, float]]:
    rate = float(config["sample_rate_hz"])
    samples = duration_s * rate
    strategy = config["strategy"]
    block_frames = min(samples, max(rate * 0.5, float(config["resample"]["block_s"]) * rate))
    pulse_bytes = 112.0
    full_memory = (samples * channel_count * 8.0 + pulse_count * channel_count * pulse_bytes) / (1024.0 * 1024.0)
    anchors = math.ceil(pulse_count / max(3, int(strategy["block_pulses"])))
    block_memory = (
        4.0 * block_frames * channel_count * 8.0
        + (pulse_count + anchors) * channel_count * pulse_bytes
        + 4.0 * 1024.0 * 1024.0
    ) / (1024.0 * 1024.0)
    stride = max(1, int(strategy["sparse_stride"]))
    sparse_memory = (
        2.0 * block_frames * channel_count * 8.0
        + math.ceil(pulse_count / stride) * channel_count * pulse_bytes
        + 2.0 * 1024.0 * 1024.0
    ) / (1024.0 * 1024.0)
    common = (samples * channel_count * 7.0e-8) + duration_s * 0.01 + 0.25
    return {
        "full_exact": {
            "memory_mb": full_memory,
            "time_s": common,
        },
        "block_accumulation": {
            "memory_mb": block_memory,
            "time_s": common * 1.12 + duration_s / 900.0,
        },
        "sparse_anchor_interpolation": {
            "memory_mb": sparse_memory,
            "time_s": common * 1.06 + duration_s / 1200.0,
        },
    }


def _max_curvature_samples_per_s2(config: dict[str, Any], duration_s: float) -> float:
    maximum = 0.0
    for channel in config["channels"][1:]:
        temp = channel["temperature_c"]
        omega = 2.0 * math.pi / max(float(temp["period_s"]), 1.0e-9)
        amplitude_ppm = abs(float(channel["temp_ppm_per_c"]) * float(temp["amplitude_c"]))
        maximum = max(maximum, amplitude_ppm * omega * 1.0e-6 * float(config["sample_rate_hz"]))
    return maximum


def _error_bounds(
    duration_s: float,
    pulse_count: int,
    config: dict[str, Any],
) -> dict[str, float]:
    detector_sigma = 0.45
    period_pulses = max(1.0, float(config["signal"]["pulse_period_s"]) * float(config["sample_rate_hz"]))
    block_pulses = max(3, int(config["strategy"]["block_pulses"]))
    stride = max(1, int(config["strategy"]["sparse_stride"]))
    curvature = _max_curvature_samples_per_s2(config, duration_s)
    block_span_s = block_pulses * period_pulses / float(config["sample_rate_hz"])
    sparse_span_s = stride * period_pulses / float(config["sample_rate_hz"])
    return {
        "full_exact": detector_sigma,
        "block_accumulation": detector_sigma + 0.125 * curvature * block_span_s * block_span_s + 0.03,
        "sparse_anchor_interpolation": detector_sigma + 0.125 * curvature * sparse_span_s * sparse_span_s + 0.05,
    }


def choose_strategy(duration_s: float, pulse_count: int, config: dict[str, Any]) -> dict[str, Any]:
    settings = config["strategy"]
    budget_mb = float(settings["memory_budget_mb"])
    budget_s = float(settings["time_budget_s"])
    minimum_mb = float(settings["min_memory_mb"])
    target = float(config["accuracy_target_samples"])
    if budget_mb < minimum_mb:
        raise SyncError(
            "E_STRATEGY_BUDGET",
            "memory budget is below the streaming minimum",
            {"memory_budget_mb": budget_mb, "min_memory_mb": minimum_mb},
        )
    resources = _resource_models(duration_s, pulse_count, config)
    bounds = _error_bounds(duration_s, pulse_count, config)
    feasible = [
        name
        for name in STRATEGIES
        if resources[name]["memory_mb"] <= budget_mb
        and resources[name]["time_s"] <= budget_s
        and bounds[name] <= target
    ]
    override = str(settings.get("override", "auto"))
    if override not in ("auto", *STRATEGIES):
        raise SyncError("E_STRATEGY_BUDGET", "unknown strategy override", {"override": override})
    if override != "auto":
        if override not in feasible:
            raise SyncError(
                "E_STRATEGY_BUDGET",
                "requested strategy violates a declared resource or accuracy budget",
                {
                    "requested": override,
                    "resource": resources[override],
                    "error_bound_samples": bounds[override],
                    "accuracy_target_samples": target,
                    "memory_budget_mb": budget_mb,
                    "time_budget_s": budget_s,
                    "feasible": feasible,
                },
            )
        selected = override
        reason = "explicit override accepted after budget and accuracy validation"
    else:
        if not feasible:
            raise SyncError(
                "E_STRATEGY_BUDGET",
                "no strategy satisfies the declared budget and accuracy target",
                {
                    "resources": resources,
                    "error_bounds_samples": bounds,
                    "accuracy_target_samples": target,
                    "memory_budget_mb": budget_mb,
                    "time_budget_s": budget_s,
                },
            )
        full_max_duration = float(settings["full_max_duration_s"])
        full_max_pulses = int(settings["full_max_pulses"])
        if (
            "full_exact" in feasible
            and duration_s <= full_max_duration
            and pulse_count <= full_max_pulses
        ):
            selected = "full_exact"
            reason = "short capture fits full exact materialization and accuracy targets"
        elif "block_accumulation" in feasible:
            selected = "block_accumulation"
            reason = "capture exceeds full-exact materialization bound; bounded block accumulation preserves target accuracy"
        else:
            selected = "sparse_anchor_interpolation"
            reason = "only sparse anchors satisfy the tighter memory budget while preserving the declared accuracy bound"
    return {
        "selected": selected,
        "reason": reason,
        "duration_s": duration_s,
        "pulse_count": pulse_count,
        "accuracy_target_samples": target,
        "error_bound_samples": bounds[selected],
        "resource_model": resources[selected],
        "all_resources": resources,
        "all_error_bounds_samples": bounds,
        "memory_budget_mb": budget_mb,
        "time_budget_s": budget_s,
        "override": override,
        "feasible": feasible,
        "deterministic": True,
    }
