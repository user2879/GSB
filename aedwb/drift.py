"""Calibration residual and confidence drift diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .util import sha256_json

EPSILON = 1e-12


def build_reference(
    baseline_calibrations: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a float64 canonical posterior for every baseline frame."""

    grouped: dict[tuple[str, int], list[tuple[int, str, list[float]]]] = defaultdict(list)
    for packet in baseline_calibrations:
        for frame in packet["frames"]:
            if frame.get("unmapped"):
                continue
            probabilities = frame["calibrated_probability"]
            if sum(probabilities) <= 0.0:
                continue
            grouped[(packet["audio_file"], int(frame["global_frame_index"]))].append(
                (0, packet["packet_id"], probabilities)
            )

    entries: list[dict[str, Any]] = []
    lookup: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for (audio_file, frame_index), contributions in sorted(grouped.items()):
        contributions.sort(key=lambda item: item[1])
        stacked = np.asarray([item[2] for item in contributions], dtype=np.float64)
        probabilities = np.mean(stacked, axis=0)
        total = float(np.sum(probabilities))
        probabilities = probabilities / total if total > 0.0 else np.zeros_like(probabilities)
        probabilities_list = [float(value) for value in probabilities]
        item = {
            "audio_file": audio_file,
            "global_frame_index": frame_index,
            "probabilities": probabilities_list,
            "source_packet_ids": sorted(item[1] for item in contributions),
        }
        entries.append(item)
        lookup[audio_file][frame_index] = item

    fingerprint_payload = [
        {
            "audio_file": item["audio_file"],
            "global_frame_index": item["global_frame_index"],
            "probabilities": item["probabilities"],
            "source_packet_ids": item["source_packet_ids"],
        }
        for item in entries
    ]
    fingerprint = sha256_json({"kind": "drift-reference", "frames": fingerprint_payload})
    metadata = {
        "fingerprint": fingerprint,
        "frame_count": len(entries),
        "baseline_packet_ids": sorted({item[1] for contributions in grouped.values() for item in contributions}),
        "precision": "float64",
        "aggregation": "equal_weight_mean_of_baseline_posteriors",
    }
    return metadata, lookup


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
    }


def _direction(signed_value: float) -> str:
    if signed_value > 1e-12:
        return "packet_above_reference"
    if signed_value < -1e-12:
        return "packet_below_reference"
    return "equal_to_reference"


def _attribution(packet: dict[str, Any], confidence_shift: float) -> dict[str, Any]:
    frames = packet["frames"]
    if not frames:
        return {
            "source": "confidence_shift",
            "scores": {
                "logit_conversion": 0.0,
                "frame_resampling": 0.0,
                "codebook_mapping": 0.0,
                "confidence_shift": confidence_shift,
            },
            "basis": "no calibrated frames",
        }
    conversion = (
        float(np.mean([frame["conversion_residual"] for frame in frames]))
        if packet["source_score_type"] == "logit"
        else 0.0
    )
    resampling = float(np.mean([frame["alignment_residual"] for frame in frames]))
    mapping = float(np.mean([frame["mapping_residual"] for frame in frames]))
    scores = {
        "logit_conversion": conversion,
        "frame_resampling": resampling,
        "codebook_mapping": mapping,
        "confidence_shift": confidence_shift,
    }
    mechanical = {
        "logit_conversion": conversion,
        "frame_resampling": resampling,
        "codebook_mapping": mapping,
    }
    # A stable order is used only for exact ties; selection itself is numeric.
    ordered = sorted(
        mechanical.items(),
        key=lambda item: (
            -item[1],
            {"codebook_mapping": 0, "frame_resampling": 1, "logit_conversion": 2}[item[0]],
        ),
    )
    source, value = ordered[0]
    if value <= 1e-15:
        source = "confidence_shift"
        basis = "all measured drift residuals are numerically zero"
    else:
        basis = (
            f"largest mechanical residual component: {source}={value:.12g}; "
            f"confidence_shift={confidence_shift:.12g}"
        )
    return {
        "source": source,
        "scores": scores,
        "basis": basis,
    }


def diagnose_packet(
    packet: dict[str, Any],
    reference_lookup: dict[str, dict[int, dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    classes = list(config["classes"])
    threshold = float(config["drift_alert_threshold"])
    per_class_abs: list[list[float]] = [[] for _ in classes]
    per_class_kl: list[list[float]] = [[] for _ in classes]
    per_class_signed: list[list[float]] = [[] for _ in classes]
    per_class_max: list[dict[str, Any] | None] = [None for _ in classes]
    reference_ids: set[str] = set()
    compared_frames = 0
    overall_abs: list[float] = []

    for frame in packet["frames"]:
        if frame.get("unmapped"):
            continue
        reference = reference_lookup.get(packet["audio_file"], {}).get(int(frame["global_frame_index"]))
        if reference is None:
            continue
        compared_frames += 1
        reference_ids.update(reference["source_packet_ids"])
        packet_probability = np.asarray(frame["calibrated_probability"], dtype=np.float64)
        reference_probability = np.asarray(reference["probabilities"], dtype=np.float64)
        signed = packet_probability - reference_probability
        absolute = np.abs(signed)
        overall_abs.extend(float(value) for value in absolute)
        for class_index in range(len(classes)):
            p = float(packet_probability[class_index])
            q = float(reference_probability[class_index])
            abs_error = abs(p - q)
            kl_contribution = (
                p * math.log(max(p, EPSILON) / max(q, EPSILON)) if p > 0.0 else 0.0
            )
            per_class_abs[class_index].append(abs_error)
            per_class_signed[class_index].append(p - q)
            per_class_kl[class_index].append(kl_contribution)
            current_max = per_class_max[class_index]
            if current_max is None or abs_error > current_max["absolute_error"]:
                per_class_max[class_index] = {
                    "global_frame_index": int(frame["global_frame_index"]),
                    "global_time_ms": int(frame["global_time_ms"]),
                    "packet_probability": p,
                    "reference_probability": q,
                    "signed_error": p - q,
                    "absolute_error": abs_error,
                    "kl_contribution": kl_contribution,
                    "direction": _direction(p - q),
                }

    class_results: dict[str, Any] = {}
    alerts: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(classes):
        absolute_distribution = _distribution(per_class_abs[class_index])
        kl_distribution = _distribution([abs(value) for value in per_class_kl[class_index]])
        mean_signed = (
            float(np.mean(per_class_signed[class_index]))
            if per_class_signed[class_index]
            else 0.0
        )
        max_abs = float(absolute_distribution["max"])
        max_abs_kl = float(kl_distribution["max"])
        exceeded = max_abs > threshold or max_abs_kl > threshold
        metric = "max_absolute_error" if max_abs >= max_abs_kl else "max_absolute_kl_contribution"
        alert_value = max(max_abs, max_abs_kl)
        class_results[class_name] = {
            "absolute_error_distribution": absolute_distribution,
            "kl_contribution_distribution": kl_distribution,
            "direction": _direction(mean_signed),
            "mean_signed_error": mean_signed,
            "max_absolute_error_frame": per_class_max[class_index],
            "threshold_exceeded": exceeded,
        }
        if exceeded:
            alerts.append(
                {
                    "audio_file": packet["audio_file"],
                    "packet_id": packet["packet_id"],
                    "class": class_name,
                    "metric": metric,
                    "value": alert_value,
                    "max_absolute_error": max_abs,
                    "max_absolute_kl_contribution": max_abs_kl,
                    "threshold": threshold,
                    "direction": _direction(mean_signed),
                    "max_frame": per_class_max[class_index],
                }
            )

    mean_confidence_shift = float(np.mean(overall_abs)) if overall_abs else 0.0
    attribution = _attribution(packet, mean_confidence_shift)
    for class_result in class_results.values():
        class_result["attribution"] = attribution
    for alert in alerts:
        alert["source"] = attribution["source"]
        alert["source_scores"] = attribution["scores"]
        alert["source_basis"] = attribution["basis"]

    return {
        "packet_id": packet["packet_id"],
        "audio_file": packet["audio_file"],
        "source_score_type": packet["source_score_type"],
        "calibration_fingerprint": packet["calibration_fingerprint"],
        "reference_packet_ids": sorted(reference_ids),
        "compared_frame_count": compared_frames,
        "overall_absolute_error_distribution": _distribution(overall_abs),
        "per_class": class_results,
        "alerts": sorted(alerts, key=lambda item: item["class"]),
        "attribution": attribution,
    }
