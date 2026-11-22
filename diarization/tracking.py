"""Dual bit-width error propagation tracking and flip attribution."""


def _boundaries(segments):
    out = []
    for seg in segments:
        out.append(seg["start_frame"])
        out.append(seg["end_frame"])
    return out


def _stage(name, entries, tol, key):
    max_diff = 0.0
    first_exceed = None
    for entry in entries:
        if entry["diff"] > max_diff:
            max_diff = entry["diff"]
        if first_exceed is None and entry["diff"] > tol:
            first_exceed = entry[key]
    return {
        "stage": name,
        "per_window": entries,
        "max_diff": max_diff,
        "first_exceed": first_exceed,
    }


def compare_widths(low, high, config):
    """Compare the low and high bit-width pipelines stage by stage.

    Stages: embedding quantization, similarity comparison, clustering
    assignment, segmentation merge. Reports per-window diffs, the first
    stage exceeding the configured tolerance, and - when speaker counts
    or segment boundaries diverge - attributes the cause to embedding
    quantization residual, similarity rounding, threshold-neighborhood
    decision, or merge-gate hysteresis boundary.
    """
    tol = float(config["tolerance"])
    threshold = float(config["similarity_threshold"])
    hysteresis = float(config["merge_hysteresis"])
    stages = []

    per_window = []
    for idx in sorted(low["dequantized"]):
        vec_low = low["dequantized"][idx]
        vec_high = high["dequantized"][idx]
        diff = max((abs(float(x) - float(y)) for x, y in zip(vec_low, vec_high)),
                   default=0.0)
        per_window.append({"window_index": idx, "diff": diff})
    stages.append(_stage("embedding_quantization", per_window, tol, "window_index"))

    per_pair = []
    for i in sorted(low["similarities"]):
        diff = abs(low["similarities"][i] - high["similarities"][i])
        per_pair.append({"pair": [i, i + 1], "diff": diff})
    stages.append(_stage("similarity_comparison", per_pair, tol, "pair"))

    per_assign = []
    assign_equal = True
    all_idx = sorted(set(low["assignments"]) | set(high["assignments"]))
    for idx in all_idx:
        mismatch = 0 if low["assignments"].get(idx) == high["assignments"].get(idx) else 1
        if mismatch:
            assign_equal = False
        per_assign.append({"window_index": idx, "diff": mismatch})
    stages.append(_stage("clustering_assignment", per_assign, tol, "window_index"))

    bound_low = _boundaries(low["segments"])
    bound_high = _boundaries(high["segments"])
    per_seg = []
    for k in range(max(len(bound_low), len(bound_high))):
        a = bound_low[k] if k < len(bound_low) else None
        b = bound_high[k] if k < len(bound_high) else None
        if a == b:
            diff = 0
        elif a is None or b is None:
            diff = 1
        else:
            diff = abs(a - b)
        per_seg.append({"segment_index": k, "diff": diff})
    stages.append(_stage("segmentation_merge", per_seg, tol, "segment_index"))

    first_exceed = None
    for stage in stages:
        if stage["max_diff"] > tol:
            first_exceed = stage["stage"]
            break

    speakers_low = sorted({v for v in low["assignments"].values()})
    speakers_high = sorted({v for v in high["assignments"].values()})
    diverged = len(speakers_low) != len(speakers_high) or bound_low != bound_high

    cause = None
    if diverged:
        emb_exceed = stages[0]["max_diff"] > tol
        sim_exceed = stages[1]["max_diff"] > tol
        merge_differs = (low["merge_events"] != high["merge_events"]
                         or low["merge_boundary_events"] != high["merge_boundary_events"])
        if emb_exceed and not sim_exceed and assign_equal:
            cause = "embedding_quantization_residual"
        elif assign_equal and sim_exceed:
            cause = "similarity_rounding"
        elif not assign_equal:
            neighborhood = False
            for idx in all_idx:
                if low["assignments"].get(idx) == high["assignments"].get(idx):
                    continue
                for margins in (low["decision_margins"], high["decision_margins"]):
                    margin = margins.get(idx)
                    if margin is not None and abs(margin - threshold) <= hysteresis:
                        neighborhood = True
            if neighborhood:
                cause = "threshold_neighborhood_decision"
            elif merge_differs:
                cause = "merge_gate_hysteresis_boundary"
            else:
                cause = "threshold_neighborhood_decision"
        elif merge_differs:
            cause = "merge_gate_hysteresis_boundary"
        else:
            cause = "embedding_quantization_residual"

    return {
        "stages": stages,
        "first_exceed_stage": first_exceed,
        "tolerance": tol,
        "flip": {
            "diverged": diverged,
            "cause": cause,
            "cluster_count_low": len(speakers_low),
            "cluster_count_high": len(speakers_high),
            "boundary_frames_low": bound_low,
            "boundary_frames_high": bound_high,
        },
    }
