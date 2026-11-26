"""Cross-segment emotion aggregation, banding and overlap arbitration."""

from . import fixedpoint as fp


def classify(confidence, thresholds, tier):
    """Deterministic four-band classification with boundary marking.

    A value exactly on a threshold takes the higher band.  A value within half
    a quantization step of a threshold is flagged as a boundary interval.
    """
    neg = tier.to_raw(thresholds["negative"])
    neu = tier.to_raw(thresholds["neutral"])
    pos = tier.to_raw(thresholds["positive"])
    half = tier.half_step
    boundary = False
    matched_threshold = None
    # Exact equality always resolves to the higher band (round half up).
    if confidence == neg:
        label, boundary, matched_threshold = "neutral", True, neg
    elif confidence == neu:
        label, boundary, matched_threshold = "neutral", True, neu
    elif confidence == pos:
        label, boundary, matched_threshold = "positive", True, pos
    elif confidence < neg:
        label = "negative"
    elif confidence < neu:
        label = "neutral"
    elif confidence < pos:
        label = "positive" if confidence > neu else "neutral"
    else:
        label = "positive"
    if not boundary:
        for threshold in (neg, neu, pos):
            if fp.signed_distance(confidence, threshold) <= half:
                boundary = True
                matched_threshold = threshold
                break
    label = "positive" if label == "positive_low" else label
    label = "neutral" if label == "neutral_low" else label
    return {
        "label": label,
        "boundary": boundary,
        "threshold_distance": (
            fp.signed_distance(confidence, matched_threshold)
            if matched_threshold is not None else None
        ),
    }


def aggregate_segments(calibrations, word_records, transcript, config, tier):
    by_segment = {}
    for item in calibrations:
        by_segment.setdefault(item["segment_id"], []).append(item)
    words_by_segment = {}
    window_usage = {}
    for record in word_records:
        words_by_segment.setdefault(record["segment_id"], []).append(
            record["word_key"])
        for window_id in record["window_ids"]:
            window_usage.setdefault(record["segment_id"], set()).add(window_id)

    summaries = []
    segment_index = {seg["segment_id"]: seg
                     for seg in transcript["segments"]}
    for seg_id, items in by_segment.items():
        ordered = sorted(items, key=lambda entry: entry["round_index"])
        total = 0
        for entry in ordered:  # explicit ascending accumulation
            total = fp.add(total, entry["calibration"])
        confidence = fp.div_by_int(total, len(ordered), tier)
        decision = classify(
            confidence, config["emotion_thresholds"], tier)
        seg = segment_index[seg_id]
        summaries.append({
            "segment_id": seg_id,
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "confidence": confidence,
            "label": decision["label"],
            "boundary": decision["boundary"],
            "threshold_distance": decision["threshold_distance"],
            "word_keys": list(words_by_segment.get(seg_id, [])),
            "window_ids": sorted(window_usage.get(seg_id, set())),
            "suppressed": False,
            "suppression_reason": None,
        })
    return arbitrate(summaries)


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def _priority_key(summary):
    return (-summary["confidence"], summary["start"],
            summary["speaker"], summary["segment_id"])


def arbitrate(summaries):
    """Resolve every overlapping pair by confidence/start/speaker priority."""
    ordered = sorted(summaries, key=_priority_key)
    for index, candidate in enumerate(ordered):
        if candidate["suppressed"]:
            continue
        for other in ordered[index + 1:]:
            if overlaps(candidate, other):
                other["suppressed"] = True
                other["covered_by"] = candidate["segment_id"]
                other["suppression_reason"] = (
                    f"overlaps {candidate['segment_id']} with higher "
                    f"priority (confidence {candidate['confidence']} >= "
                    f"{other['confidence']}, start {candidate['start']}, "
                    f"speaker {candidate['speaker']})"
                )
    return sorted(summaries,
                  key=lambda s: (s["start"], s["speaker"], s["segment_id"]))
