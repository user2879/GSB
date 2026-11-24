"""Two-tier error propagation and boundary flip attribution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .config import DiarConfig, QUANT_TIERS
from .quantize import q16_to_float

STAGE_QUANT = "quantization"
STAGE_SIM = "similarity"
STAGE_CLUSTER = "clustering"
STAGE_SEGMENT = "segmentation"
STAGES = (STAGE_QUANT, STAGE_SIM, STAGE_CLUSTER, STAGE_SEGMENT)


@dataclass
class WindowStageDiff:
    window: int
    quantization_max_abs: float
    similarity_abs_q16: int
    cluster_changed: bool
    boundary_delta_ms: float
    first_exceed_stage: str


def higher_tier_config(cfg: DiarConfig) -> DiarConfig:
    idx = QUANT_TIERS.index(cfg.word_bits)
    if idx == len(QUANT_TIERS) - 1:
        # Already the highest tier: widen the fractional width within the same
        # word size when possible, otherwise compare against itself.
        frac = min(cfg.frac_bits + 2, cfg.word_bits - 2)
        return cfg.with_quant(cfg.word_bits, frac)
    next_word = QUANT_TIERS[idx + 1]
    next_frac = min(cfg.frac_bits + 2, next_word - 2)
    return cfg.with_quant(next_word, next_frac)


def _segment_map(segments) -> Dict[str, Tuple[float, float]]:
    return {s.speaker: (s.start_ms, s.end_ms) for s in segments}


def attribute_flip(base_result: Dict[str, Any], high_result: Dict[str, Any],
                   cfg: DiarConfig) -> Dict[str, Any]:
    base_clusters = base_result["clustering"]
    high_clusters = high_result["clustering"]
    speaker_count_changed = len(base_clusters.clusters) != len(high_clusters.clusters)

    base_sims = {c.right_window: c.similarity_q16 for c in base_clusters.comparisons}
    high_sims = {c.right_window: c.similarity_q16 for c in high_clusters.comparisons}
    threshold = base_clusters.threshold_q16

    near_threshold: List[Dict[str, Any]] = []
    for wid in sorted(set(base_sims).intersection(high_sims)):
        base_q = base_sims[wid]
        high_q = high_sims[wid]
        base_gap = abs(base_q - threshold)
        high_gap = abs(high_q - threshold)
        decision_flipped = (base_q >= threshold) != (high_q >= threshold)
        near_threshold.append({
            "window": wid,
            "base_similarity": q16_to_float(base_q),
            "high_similarity": q16_to_float(high_q),
            "threshold": q16_to_float(threshold),
            "base_gap_q16": base_gap,
            "high_gap_q16": high_gap,
            "decision_flipped": decision_flipped,
        })

    base_segments = base_result["segmentation"].segments
    high_segments = high_result["segmentation"].segments
    boundary_moves: List[Dict[str, Any]] = []
    base_map = _segment_map(base_segments)
    high_map = _segment_map(high_segments)
    for speaker in sorted(set(base_map).intersection(high_map)):
        b_start, b_end = base_map[speaker]
        h_start, h_end = high_map[speaker]
        if abs(b_start - h_start) > 1e-6 or abs(b_end - h_end) > 1e-6:
            boundary_moves.append({
                "speaker": speaker,
                "base_start_ms": b_start, "high_start_ms": h_start,
                "base_end_ms": b_end, "high_end_ms": h_end,
                "start_delta_ms": round(h_start - b_start, 3),
                "end_delta_ms": round(h_end - b_end, 3),
            })

    # Quantification evidence: compare integer embeddings and residuals.
    quant_evidence = []
    for base_q, high_q in zip(base_result["quantized"], high_result["quantized"]):
        int_delta = max(abs(a - b) for a, b in zip(base_q.integers, high_q.integers))
        quant_evidence.append({
            "window": base_q.index,
            "integer_l1_max": int_delta,
            "base_max_residual": base_q.max_residual_abs,
            "high_max_residual": high_q.max_residual_abs,
        })

    reasons: List[str] = []
    flipped_windows = [item["window"] for item in near_threshold
                       if item["decision_flipped"]]
    quant_only = [item for item in quant_evidence if item["integer_l1_max"] > 0]
    if speaker_count_changed or flipped_windows:
        if flipped_windows:
            reasons.append("similarity_threshold_neighborhood")
        if quant_only and not flipped_windows:
            reasons.append("quantization_truncation")
    if boundary_moves and not flipped_windows and not speaker_count_changed:
        reasons.append("boundary_arbitration_rule")
    if not reasons:
        reasons.append("no_flip")

    return {
        "speaker_count_base": len(base_clusters.clusters),
        "speaker_count_high": len(high_clusters.clusters),
        "speaker_count_changed": speaker_count_changed,
        "threshold_evidence": near_threshold,
        "boundary_moves": boundary_moves,
        "quantization_evidence": quant_evidence,
        "attributed_reasons": reasons,
        "criteria": {
            "threshold_gap_q16_neutral_band": max(8, threshold // 128),
            "quantization_l1_threshold": 0,
            "tolerance": cfg.tolerance,
        },
    }


def propagate(base_result: Dict[str, Any], high_result: Dict[str, Any],
              cfg: DiarConfig) -> Tuple[List[WindowStageDiff], Dict[str, Any]]:
    diffs: List[WindowStageDiff] = []
    base_quant = {q.index: q for q in base_result["quantized"]}
    high_quant = {q.index: q for q in high_result["quantized"]}
    base_sim = {c.right_window: c.similarity_q16
                for c in base_result["clustering"].comparisons}
    high_sim = {c.right_window: c.similarity_q16
                for c in high_result["clustering"].comparisons}
    base_assign = base_result["clustering"].assignments
    high_assign = high_result["clustering"].assignments

    # Precompute per-window boundary movement from segment extents.
    boundary_delta_by_window: Dict[int, float] = {wid: 0.0 for wid in base_quant}
    for index, base_seg in enumerate(base_result["segmentation"].segments):
        high_segments = high_result["segmentation"].segments
        if index >= len(high_segments):
            continue
        high_seg = high_segments[index]
        if base_seg.speaker != high_seg.speaker:
            continue
        delta = max(abs(base_seg.start_ms - high_seg.start_ms),
                    abs(base_seg.end_ms - high_seg.end_ms))
        for wid in base_seg.arbitration_windows:
            if wid in boundary_delta_by_window:
                boundary_delta_by_window[wid] = max(
                    boundary_delta_by_window[wid], delta)

    for wid in sorted(base_quant):
        bq, hq = base_quant[wid], high_quant[wid]
        quant_diff = max(abs(a - b)
                         for a, b in zip(bq.dequantized, hq.dequantized))
        sim_diff = abs(base_sim.get(wid, 0) - high_sim.get(wid, 0))
        cluster_changed = base_assign.get(wid) != high_assign.get(wid)
        boundary_delta = boundary_delta_by_window.get(wid, 0.0)

        # Each stage is judged independently; the first stage (in processing
        # order) whose difference exceeds the configured tolerance is flagged.
        quant_exceed = quant_diff > cfg.tolerance
        sim_exceed = q16_to_float(sim_diff) > cfg.tolerance
        cluster_exceed = cluster_changed
        segment_exceed = boundary_delta > cfg.tolerance
        if quant_exceed:
            first = STAGE_QUANT
        elif sim_exceed:
            first = STAGE_SIM
        elif cluster_exceed:
            first = STAGE_CLUSTER
        elif segment_exceed:
            first = STAGE_SEGMENT
        else:
            first = "within_tolerance"
        diffs.append(WindowStageDiff(
            window=wid, quantization_max_abs=quant_diff,
            similarity_abs_q16=sim_diff, cluster_changed=cluster_changed,
            boundary_delta_ms=boundary_delta, first_exceed_stage=first,
        ))
    attribution = attribute_flip(base_result, high_result, cfg)
    return diffs, attribution
