"""Quantized threshold decisions, deterministic tie breaking, and fusion."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


def quantize_scores(probabilities: list[float], quantum: float) -> list[float]:
    values = np.asarray(probabilities, dtype=np.float64)
    scaled = values / quantum
    # Half-up rather than banker's rounding keeps class-order precedence stable.
    rounded = np.floor(scaled + 0.5) * quantum
    return [float(value) for value in np.clip(rounded, 0.0, 1.0)]


def decide_probabilities(
    probabilities: list[float],
    config: dict[str, Any],
) -> dict[str, Any]:
    classes = list(config["classes"])
    priority = list(config["class_priority"])
    priority_rank = {name: index for index, name in enumerate(priority)}
    quantum = float(config["probability_quantum"])
    threshold = float(config["decision_threshold"])
    epsilon = max(1e-12, quantum * 1e-9)

    scores = quantize_scores(probabilities, quantum)
    ranked = sorted(
        ((score, priority_rank[classes[index]], index, classes[index]) for index, score in enumerate(scores)),
        key=lambda item: (-item[0], item[1]),
    )
    top_score, _, top_index, top_class = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    tie_candidates = [
        (index, classes[index], score)
        for index, score in enumerate(scores)
        if score >= top_score - quantum / 2.0 - epsilon
    ]
    tie_candidates.sort(key=lambda item: priority_rank[item[1]])
    tie_break = len(tie_candidates) > 1 or margin <= quantum / 2.0 + epsilon
    threshold_equality = abs(top_score - threshold) <= epsilon
    boundary = bool(tie_break or threshold_equality)

    chosen_index, chosen_class, chosen_score = tie_candidates[0] if tie_break else (
        top_index,
        top_class,
        top_score,
    )
    reasons: list[str] = []
    if tie_break:
        reasons.append("quantum_tie_priority")
    if threshold_equality:
        reasons.append("threshold_equality")

    if chosen_score + epsilon < threshold:
        decision = "none"
        reasons.append("below_threshold")
    else:
        decision = chosen_class
    if decision == "none" and threshold_equality:
        # Equality is still a valid boundary diagnosis even if another guard
        # later treats the frame as below threshold.
        boundary = True

    return {
        "decision": decision,
        "selected_class": chosen_class,
        "selected_probability": chosen_score,
        "quantized_scores": scores,
        "top1_class": top_class,
        "top1_score": top_score,
        "top2_score": second_score,
        "top1_top2_margin": margin,
        "threshold": threshold,
        "probability_quantum": quantum,
        "boundary_sample": boundary,
        "boundary_reasons": reasons,
        "tie_break_priority": tie_break,
        "threshold_margin": chosen_score - threshold,
    }


def decide_packet(
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for frame in calibration["frames"]:
        decision = decide_probabilities(frame["calibrated_probability"], config)
        frames.append(
            {
                "global_frame_index": frame["global_frame_index"],
                "global_time_ms": frame["global_time_ms"],
                "calibrated_probability": frame["calibrated_probability"],
                **decision,
            }
        )
    return {
        **{key: calibration[key] for key in (
            "packet_id",
            "source_file",
            "source_score_type",
            "content_sha256",
            "audio_file",
            "start_ms",
            "end_ms",
            "frame_shift_ms",
            "class_codebook_version",
            "frame_grid_version",
            "detector_version",
            "calibration_fingerprint",
            "injection_ids",
        )},
        "frames": frames,
        "decision_counts": _count_decisions(frame["decision"] for frame in frames),
        "boundary_sample_count": sum(1 for frame in frames if frame["boundary_sample"]),
    }


def _count_decisions(decisions: Any, classes: list[str] | None = None) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for decision in decisions:
        counts[decision] += 1
    if classes is not None:
        for name in classes:
            counts.setdefault(name, 0)
    return {key: int(counts[key]) for key in sorted(counts)}


def fuse_and_decide_global_frames(
    calibration_results: list[dict[str, Any]],
    packet_indexes: dict[str, int],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Confidence-stream fusion across overlapping shards, then one decision per frame."""

    grouped: dict[tuple[str, int], list[tuple[int, str, list[float], dict[str, Any]]]] = defaultdict(list)
    for packet_index, packet in enumerate(calibration_results):
        for frame in packet["frames"]:
            if frame.get("unmapped"):
                continue
            probabilities = frame["calibrated_probability"]
            if sum(probabilities) <= 0.0:
                continue
            grouped[(packet["audio_file"], int(frame["global_frame_index"]))].append(
                (
                    packet_indexes[packet["packet_id"]],
                    packet["packet_id"],
                    probabilities,
                    frame,
                )
            )

    global_frames: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    decisions: list[str] = []
    for (audio_file, frame_index), contributions in sorted(grouped.items()):
        contributions.sort(key=lambda item: (item[0], item[1]))
        stacked = np.asarray([item[2] for item in contributions], dtype=np.float64)
        fused = np.mean(stacked, axis=0)
        total = float(np.sum(fused))
        fused = fused / total if total > 0.0 else np.zeros_like(fused)
        fused_list = [float(value) for value in fused]
        decision = decide_probabilities(fused_list, config)
        sample = {
            "audio_file": audio_file,
            "global_frame_index": frame_index,
            "global_time_ms": frame_index * int(config["frame_shift_ms"]),
            "source_packet_ids": [item[1] for item in contributions],
            "source_packet_indexes": [item[0] for item in contributions],
            "fused_probability": fused_list,
            **decision,
        }
        global_frames.append(sample)
        decisions.append(decision["decision"])
        if decision["boundary_sample"]:
            boundaries.append(
                {
                    "audio_file": audio_file,
                    "global_frame_index": frame_index,
                    "global_time_ms": sample["global_time_ms"],
                    "source_packet_ids": sample["source_packet_ids"],
                    "fused_probability": fused_list,
                    "quantized_scores": decision["quantized_scores"],
                    "decision": decision["decision"],
                    "top1_class": decision["top1_class"],
                    "top1_score": decision["top1_score"],
                    "top2_score": decision["top2_score"],
                    "top1_top2_margin": decision["top1_top2_margin"],
                    "threshold": decision["threshold"],
                    "probability_quantum": decision["probability_quantum"],
                    "boundary_reasons": decision["boundary_reasons"],
                }
            )

    counts = _count_decisions(decisions, list(config["classes"]) + ["none"])
    for name in list(config["classes"]) + ["none"]:
        counts.setdefault(name, 0)
    return global_frames, boundaries, {key: int(counts[key]) for key in sorted(counts)}
