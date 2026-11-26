"""Stage-wise error tracking against a wider no-quantization reference path."""

from . import fixedpoint as fp
from .features import FEATURE_DIMS


def _to_reference(raw, src_tier):
    converted, _ = fp.requantize(raw, src_tier, fp.REFERENCE)
    return converted


def compare_vectors(quantized, reference, q_tier):
    diffs = []
    for q_value, r_value in zip(quantized, reference):
        q_ref = _to_reference(q_value, q_tier)
        diffs.append(fp.signed_distance(q_ref, r_value))
    return diffs


def track_window(quant_windows, ref_windows, q_tier, tolerance):
    per_segment = {}
    first_breach = None
    for q_win, r_win in zip(quant_windows, ref_windows):
        diffs = compare_vectors(q_win["features"], r_win["features"], q_tier)
        for dim in range(FEATURE_DIMS):
            for seg_id in q_win["segment_refs"]:
                per_segment.setdefault(seg_id, []).append(diffs[dim])
            if first_breach is None and diffs[dim] > tolerance:
                first_breach = {
                    "stage": "window",
                    "window_id": q_win["window_id"],
                    "dimension": dim,
                    "diff": diffs[dim],
                    "tolerance": tolerance,
                }
    maximum = {seg_id: (max(values) if values else 0)
               for seg_id, values in per_segment.items()}
    return {"stage": "window", "per_segment_max_diff": maximum,
            "first_breach": first_breach}


def track_words(quant_words, ref_words, q_tier, tolerance):
    per_segment = {}
    first_breach = None
    ref_by_key = {item["word_key"]: item for item in ref_words}
    for q_word in quant_words:
        r_word = ref_by_key[q_word["word_key"]]
        diffs = compare_vectors(q_word["strength"], r_word["strength"], q_tier)
        per_segment[q_word["segment_id"]] = max(
            per_segment.get(q_word["segment_id"], 0), max(diffs))
        for dim, diff in enumerate(diffs):
            if first_breach is None and diff > tolerance:
                first_breach = {
                    "stage": "word",
                    "word_key": q_word["word_key"],
                    "dimension": dim,
                    "diff": diff,
                    "tolerance": tolerance,
                }
    return {"stage": "word", "per_segment_max_diff": per_segment,
            "first_breach": first_breach}


def track_calibration(quant_cals, ref_cals, tolerance):
    per_segment = {}
    first_breach = None
    ref_by_key = {item["word_key"]: item for item in ref_cals}
    for q_item in quant_cals:
        r_item = ref_by_key[q_item["word_key"]]
        diff = fp.signed_distance(q_item["calibration"],
                                  _to_reference(r_item["calibration"],
                                                fp.REFERENCE))
        per_segment[q_item["segment_id"]] = max(
            per_segment.get(q_item["segment_id"], 0), diff)
        if first_breach is None and diff > tolerance:
            first_breach = {
                "stage": "calibration",
                "word_key": q_item["word_key"],
                "diff": diff,
                "tolerance": tolerance,
            }
    return {"stage": "calibration",
            "per_segment_max_diff": per_segment,
            "first_breach": first_breach}


def track_segments(quant_summaries, ref_summaries, tolerance):
    per_segment = {}
    first_breach = None
    ref_by_id = {item["segment_id"]: item for item in ref_summaries}
    for q_item in quant_summaries:
        r_item = ref_by_id[q_item["segment_id"]]
        diff = fp.signed_distance(
            q_item["confidence"],
            _to_reference(r_item["confidence"], fp.REFERENCE))
        per_segment[q_item["segment_id"]] = diff
        if first_breach is None and diff > tolerance:
            first_breach = {
                "stage": "segment",
                "segment_id": q_item["segment_id"],
                "diff": diff,
                "tolerance": tolerance,
            }
    return {"stage": "segment",
            "per_segment_max_diff": per_segment,
            "first_breach": first_breach}


def first_overall_breach(reports):
    for stage in ("window", "word", "calibration", "segment"):
        report = next(item for item in reports if item["stage"] == stage)
        if report["first_breach"] is not None:
            return report["first_breach"]
    return None


def attribute_flip(coarse_summary, fine_summary, coarse_cals, fine_cals,
                   thresholds, tier_pair):
    """Deterministically attribute a label flip between two quant tiers."""
    coarse_tier, fine_tier = tier_pair
    c_conf = coarse_summary["confidence"]
    f_conf = fine_summary["confidence"]
    near_threshold = False
    for name in ("negative", "neutral", "positive"):
        threshold = coarse_tier.to_raw(thresholds[name])
        if fp.signed_distance(c_conf, threshold) <= coarse_tier.half_step:
            near_threshold = True
    c_by_key = {item["word_key"]: item for item in coarse_cals}
    f_by_key = {item["word_key"]: item for item in fine_cals}
    decay_dominant = False
    for key, c_item in c_by_key.items():
        f_item = f_by_key[key]
        c_decay = c_item["calibration"] - c_item["evidence_value"]
        f_decay = _to_reference(f_item["calibration"], fine_tier) - \
            _to_reference(f_item["evidence_value"], fine_tier)
        if fp.signed_distance(
                _to_reference(c_decay, coarse_tier), f_decay) > \
                fp.signed_distance(
                    _to_reference(c_item["evidence_value"], coarse_tier),
                    _to_reference(f_item["evidence_value"], fine_tier)):
            decay_dominant = True
    if near_threshold:
        reason = "threshold_comparison"
    elif decay_dominant:
        reason = "decay_multiplication"
    else:
        reason = "quantization_rounding"
    return {
        "segment_id": coarse_summary["segment_id"],
        "coarse_label": coarse_summary["label"],
        "fine_label": fine_summary["label"],
        "flip_reason": reason,
    }


def flip_report(coarse_summaries, fine_summaries, coarse_cals, fine_cals,
                config, tiers):
    flips = []
    fine_by_id = {item["segment_id"]: item for item in fine_summaries}
    for coarse in coarse_summaries:
        fine = fine_by_id[coarse["segment_id"]]
        if coarse["label"] != fine["label"]:
            flips.append(attribute_flip(
                coarse, fine, coarse_cals, fine_cals,
                config["emotion_thresholds"],
                (tiers["coarse"], tiers["fine"])))
    return flips
