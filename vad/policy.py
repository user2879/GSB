"""Scale-aware, auditable strategy selection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .features import frame_count, frame_geometry, frame_rms_values
from .util import PRODUCT_VERSIONS, round_float


STAGE_ORDER = ("policy", "features", "state_machine", "endpoints")


def estimate_scale_metrics(
    samples: np.ndarray,
    sample_rate: int,
    config: dict[str, Any],
    sample_count: int,
) -> dict[str, Any]:
    _frame_samples, hop_samples = frame_geometry(config, sample_rate)
    total_frames = frame_count(int(samples.size), hop_samples)
    energies = frame_rms_values(samples, config, sample_rate)
    high = float(config["high_threshold"])
    active = energies > high
    active_frames = int(np.count_nonzero(active))
    density = float(active_frames) / float(total_frames) if total_frames else 0.0
    runs = 0
    previous = False
    for value in active.tolist():
        if value and not previous:
            runs += 1
        previous = value
    return {
        "sample_count": int(sample_count),
        "total_frames": int(total_frames),
        "active_frames": active_frames,
        "estimated_active_frame_ratio": round_float(density, 10),
        "estimated_energy_active_runs": runs,
        "density_estimator": "raw_rms_above_configured_high_threshold",
        "density_threshold": high,
    }


def select_policy(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    profiles = config["scale_profiles"]
    frames = max(float(metrics["total_frames"]), 0.0)
    density = max(float(metrics["estimated_active_frame_ratio"]), 0.0)
    sample_count = max(float(metrics["sample_count"]), 0.0)
    candidates: list[dict[str, Any]] = []
    for order, profile in enumerate(profiles):
        frame_distance = abs(
            math.log2((frames + 1.0) / (float(profile["target_frames"]) + 1.0))
        )
        density_distance = abs(
            math.log2(
                (density + 0.02) / (max(float(profile["target_density"]), 0.0) + 0.02)
            )
        )
        load_distance = abs(
            math.log2(
                (sample_count + 1.0) / (float(profile["target_samples"]) + 1.0)
            )
        )
        weighted = (
            0.60 * frame_distance + 0.25 * density_distance + 0.15 * load_distance
        )
        score = 1.0 / (1.0 + weighted)
        candidates.append(
            {
                "profile": profile["id"],
                "order": order,
                "frame_distance": round_float(frame_distance, 10),
                "density_distance": round_float(density_distance, 10),
                "load_distance": round_float(load_distance, 10),
                "weighted_distance": round_float(weighted, 10),
                "score": round_float(score, 10),
            }
        )
    override = config.get("scale_profile_override")
    if override is not None:
        selected_index = next(
            index for index, profile in enumerate(profiles) if profile["id"] == override
        )
        reason = "configured_scale_profile_override"
    else:
        selected_index = max(
            range(len(profiles)),
            key=lambda index: (candidates[index]["score"], -candidates[index]["order"]),
        )
        reason = "highest_weighted_similarity_score; tie_break=profile_order"
    selected = profiles[selected_index]
    base_window = int(config["noise_window_frames"])
    effective_window = int(
        round(base_window * float(selected["noise_window_multiplier"]))
    )
    effective_window = max(1, min(int(metrics["total_frames"]), effective_window))
    effective = {
        "feature_precision": int(selected["feature_precision"]),
        "noise_window_frames": effective_window,
        "noise_window_multiplier": float(selected["noise_window_multiplier"]),
        "threshold_estimation": selected["threshold_estimation"],
        "low_quantile": float(selected.get("low_quantile", 0.2)),
    }
    return {
        "schema_version": "1.0",
        "product_version": PRODUCT_VERSIONS["policy"],
        "metrics": metrics,
        "candidate_scores": candidates,
        "selected_profile": selected["id"],
        "selected_profile_order": selected_index,
        "selection_reason": reason,
        "effective_strategy": effective,
    }
