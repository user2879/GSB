"""Word-level evidence construction in fixed-point integers."""

from . import fixedpoint as fp
from .features import FEATURE_DIMS


def windows_for_word(word, windows, frame_length, frame_hop):
    """Return [(window, coverage_frames)] covering a word's sample span."""
    hits = []
    first_frame = word["start"] // frame_hop
    last_frame = (word["end"] - 1) // frame_hop
    for window in windows:
        coverage = 0
        for frame_idx in window["frame_indices"]:
            frame_start = frame_idx * frame_hop
            frame_end = frame_start + frame_length
            overlap = min(word["end"], frame_end) - max(
                word["start"], frame_start)
            if overlap > 0:
                coverage += 1
        if coverage:
            hits.append((window, coverage, first_frame, last_frame))
    return hits


def global_word_sequence(transcript):
    """Flatten words in segment order; co-occurrence spans segment edges."""
    sequence = []
    for seg in transcript["segments"]:
        for index, word in enumerate(seg["words"]):
            sequence.append({
                "segment_id": seg["segment_id"],
                "word_index": index,
                "word": word["word"],
                "emotion": word["emotion"],
                "start": word["start"],
                "end": word["end"],
            })
    return sequence


def semantic_counts(sequence, position, config):
    """Integer co-occurrence counts within the configured window.

    Counts are polarity buckets (-1/0/1) weighted by distance so nearer
    neighbors dominate; the result is fully order deterministic.
    """
    k = config["cooccur_window"]
    polarity_map = config["emotion_polarity"]
    counts = {-1: 0, 0: 0, 1: 0}
    total = 0
    target = sequence[position]
    for distance in range(1, k + 1):
        weight = k + 1 - distance
        for neighbor_pos in (position - distance, position + distance):
            if 0 <= neighbor_pos < len(sequence):
                neighbor = sequence[neighbor_pos]
                polarity = polarity_map[neighbor["emotion"]]
                counts[polarity] += weight
                total += weight
    return {"counts": counts, "total": total, "token": target["word"]}


def semantic_vector(sem, tier):
    total = max(1, sem["total"])
    vector = [0] * 3
    for dim, polarity in enumerate((-1, 0, 1)):
        raw_count = fp.from_int(sem["counts"][polarity], tier)
        vector[dim] = fp.div_by_int(raw_count, total, tier)
    return vector


def acoustic_vector(hits, tier):
    """Coverage-frame weighted average of the covering window features."""
    total_weight = sum(coverage for _, coverage, _, _ in hits)
    total_weight = max(1, total_weight)
    accum = [0] * FEATURE_DIMS
    for window, coverage, _, _ in hits:
        for dim in range(FEATURE_DIMS):  # dimension index ascending
            accum[dim] += fp.mul_by_int(window["features"][dim], coverage)
    return [fp.div_by_int(accum[dim], total_weight, tier)
            for dim in range(FEATURE_DIMS)]


def build_word_evidence(word_ref, word, windows, sequence, position,
                        config, tier):
    hits = windows_for_word(
        word, windows, config["frame_length"], config["frame_hop"])
    acoustic = acoustic_vector(hits, tier)
    sem = semantic_counts(sequence, position, config)
    semantic = semantic_vector(sem, tier)

    wa = tier.to_raw(config["acoustic_weight"])
    ws = tier.to_raw(config["semantic_weight"])
    semantic_map = (semantic[0], semantic[1], semantic[2],
                    semantic[0], semantic[1], semantic[2])
    strength = [0] * FEATURE_DIMS
    for dim in range(FEATURE_DIMS):  # fixed dimension-ascending fusion
        strength[dim] = fp.add(
            fp.mul(acoustic[dim], wa, tier),
            fp.mul(semantic_map[dim], ws, tier),
        )

    return {
        "word_key": f"{word_ref['segment_id']}:{word_ref['word_index']}",
        "segment_id": word_ref["segment_id"],
        "word_index": word_ref["word_index"],
        "token": word["word"],
        "emotion": word["emotion"],
        "window_ids": [window["window_id"] for window, _, _, _ in hits],
        "coverage_frames": {
            window["window_id"]: coverage
            for window, coverage, _, _ in hits
        },
        "acoustic": acoustic,
        "semantic": list(semantic),
        "semantic_total": sem["total"],
        "strength": strength,
        "quant_step": fp.quant_step(tier),
        "residual": [0] * FEATURE_DIMS,
        "tier": tier.name,
    }


def quantize_word_evidence(record, compute_tier, target_tier):
    """Requantize fused strength and record exact per-dimension residuals."""
    if compute_tier.name == target_tier.name:
        record["residual"] = [0] * FEATURE_DIMS
        record["strength"] = [fp.clamp_to_width(v, target_tier)
                              for v in record["strength"]]
        return record
    new_strength = []
    residuals = []
    for dim in range(FEATURE_DIMS):
        raw, residual = fp.requantize(
            record["strength"][dim], compute_tier, target_tier)
        new_strength.append(fp.clamp_to_width(raw, target_tier))
        residuals.append(residual)
    record["strength"] = new_strength
    record["residual"] = residuals
    record["tier"] = target_tier.name
    return record


def build_all_word_evidence(transcript, windows, config, tier):
    sequence = global_word_sequence(transcript)
    by_token = {(item["segment_id"], item["word_index"]): index
                for index, item in enumerate(sequence)}
    records = []
    for seg in transcript["segments"]:
        for index, word in enumerate(seg["words"]):
            word_ref = {"segment_id": seg["segment_id"],
                        "word_index": index}
            position = by_token[(seg["segment_id"], index)]
            intermediate = tier if tier.name == fp.REFERENCE.name else fp.FINE
            record = build_word_evidence(
                word_ref, word, windows, sequence, position,
                config, intermediate)
            quantize_word_evidence(record, intermediate, tier)
            records.append(record)
    return records
