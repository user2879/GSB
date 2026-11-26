"""Round-by-round fixed-point emotion calibration."""

from . import fixedpoint as fp
from .features import FEATURE_DIMS

POLARITY_SIGN = {1: 1, 0: 0, -1: -1}


def ordered_rounds(transcript):
    """Rounds ordered by (start sample, speaker) with stable segment order."""
    rounds = []
    for seg in sorted(
        transcript["segments"],
        key=lambda s: (s["start"], s["speaker"], s["segment_id"]),
    ):
        for index, word in enumerate(seg["words"]):
            rounds.append({
                "round_index": len(rounds),
                "segment_id": seg["segment_id"],
                "word_index": index,
                "word_key": f"{seg['segment_id']}:{index}",
                "start": word["start"],
            })
    return rounds


def round_evidence(rounds, word_records, polarity_map):
    by_key = {record["word_key"]: record for record in word_records}
    result = []
    for item in rounds:
        record = by_key.get(item["word_key"])
        if record is None:
            result.append(None)
            continue
        polarity = polarity_map[record["emotion"]]
        sign = POLARITY_SIGN[polarity]
        value = record["strength"][0]
        result.append({
            "word_key": item["word_key"],
            "segment_id": item["segment_id"],
            "polarity": polarity,
            "value": value if sign >= 0 else -value,
        })
    return result


def decay_once(value, num, den, tier):
    """One integer half-life step using numerator/denominator rounding."""
    return fp.div_by_int(fp.mul_by_int(value, num), den, tier)


def calibrate(rounds, word_records, config, tier):
    evidence = round_evidence(
        rounds, word_records, config["emotion_polarity"])
    gain = tier.to_raw(config["calibration_gain"])
    num = config["decay_num"]
    den = config["decay_den"]
    previous = 0
    gap = 0
    outputs = []
    for round_index, item in enumerate(rounds):
        ev = evidence[round_index]
        if ev is None or ev["value"] == 0:
            gap += 1
            decayed = previous
            step = 0
            while step < gap:  # integer accumulation only, no float power
                decayed = decay_once(decayed, num, den, tier)
                step += 1
            outputs.append({
                "round_index": round_index,
                "word_key": item["word_key"],
                "segment_id": item["segment_id"],
                "calibration": decayed,
                "inherited": True,
                "gap": gap,
                "evidence_value": 0,
            })
            previous = decayed
            continue
        gap = 0
        decayed = decay_once(previous, num, den, tier)
        contribution = fp.mul(ev["value"], gain, tier)
        calibrated = fp.clamp_to_width(
            fp.add(decayed, contribution), tier)
        outputs.append({
            "round_index": round_index,
            "word_key": item["word_key"],
            "segment_id": item["segment_id"],
            "calibration": calibrated,
            "inherited": False,
            "gap": 0,
            "evidence_value": ev["value"],
        })
        previous = calibrated
    return outputs


def calibrate_by_segment(calibrations):
    grouped = {}
    for item in calibrations:
        grouped.setdefault(item["segment_id"], []).append(item)
    return grouped
