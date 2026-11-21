"""Stage-by-stage error propagation tracking (requirement 4).

The full chain (normalize -> quantize -> weighted_sum -> clip) is executed
twice per turn: once at the configured precision tier and once at the next
higher tier.  Per stage, per turn, the difference between the two tiers is
reported, and the first stage whose error exceeds the configured tolerance
is flagged.  When the sentiment classification flips between tiers, the
flip is attributed to either quantisation truncation or weighted-sum
accumulation based on evidence derived from the actual computation path.
"""

from __future__ import annotations

from .fusion import CHANNELS, classify


def higher_tier(config):
    """Return (frac_bits, quantum) one tier above the configured tier.

    Falls back to the highest allowed tier when already at the top.
    """
    precision = config["precision"]
    allowed = sorted(precision.get("allowed_frac_bits", [precision["frac_bits"]]))
    current = precision["frac_bits"]
    higher = [b for b in allowed if b > current]
    return (higher[0] if higher else allowed[-1], precision["quantum"])


def _exceeds(diff, ref, abs_tol, rel_tol):
    return abs(diff) > abs_tol + rel_tol * abs(ref)


def compare_tiers(tier_lo, tier_hi, config):
    """Compare the per-stage products of two precision tiers for one turn.

    tier_lo / tier_hi: dicts with keys per stage:
        normalize: {channel: float|None}
        quantize:  {channel: int}
        weighted_sum: {"accumulator": int, "pre_clip_float": float} | None
        clip: {"confidence_int": int|None, "confidence": float|None}
    """
    abs_tol = config["tolerances"]["abs"]
    rel_tol = config["tolerances"]["rel"]
    bits_lo = config["precision"]["frac_bits"]
    bits_hi, _ = higher_tier(config)
    scale_lo = 1 << bits_lo
    scale_hi = 1 << bits_hi

    stage_diffs = {}
    first_exceed = None

    norm_diffs = {}
    for ch in CHANNELS:
        a = tier_lo["normalize"].get(ch)
        b = tier_hi["normalize"].get(ch)
        diff = None if (a is None or b is None) else abs(a - b)
        norm_diffs[ch] = diff
        if diff is not None and first_exceed is None and _exceeds(diff, b, abs_tol, rel_tol):
            first_exceed = "normalize"
    stage_diffs["normalize"] = norm_diffs

    quant_diffs = {}
    for ch in CHANNELS:
        a = tier_lo["quantize"].get(ch)
        b = tier_hi["quantize"].get(ch)
        if a is None or b is None:
            quant_diffs[ch] = None
            continue
        # integers live on different fixed-point scales; compare as floats
        diff = abs(a / scale_lo - b / scale_hi)
        quant_diffs[ch] = diff
        if first_exceed is None and _exceeds(diff, abs(b) / scale_hi, abs_tol, rel_tol):
            first_exceed = "quantize"
    stage_diffs["quantize"] = quant_diffs

    ws_diff = None
    if tier_lo["weighted_sum"] is not None and tier_hi["weighted_sum"] is not None:
        a = tier_lo["weighted_sum"]["pre_clip_float"]
        b = tier_hi["weighted_sum"]["pre_clip_float"]
        ws_diff = abs(a - b)
        if first_exceed is None and _exceeds(ws_diff, b, abs_tol, rel_tol):
            first_exceed = "weighted_sum"
    stage_diffs["weighted_sum"] = ws_diff

    clip_diff = None
    if tier_lo["clip"]["confidence"] is not None and tier_hi["clip"]["confidence"] is not None:
        a = tier_lo["clip"]["confidence"]
        b = tier_hi["clip"]["confidence"]
        clip_diff = abs(a - b)
        if first_exceed is None and _exceeds(clip_diff, b, abs_tol, rel_tol):
            first_exceed = "clip"
    stage_diffs["clip"] = clip_diff

    threshold = config.get("sentiment_threshold", 0.5)
    cls_lo = classify(tier_lo["clip"]["confidence"], threshold)
    cls_hi = classify(tier_hi["clip"]["confidence"], threshold)
    flip = None
    if cls_lo is not None and cls_hi is not None and cls_lo != cls_hi:
        # compare quantized integers on their respective fixed-point scales
        def _quant_equal(ch):
            a = tier_lo["quantize"].get(ch)
            b = tier_hi["quantize"].get(ch)
            if a is None or b is None:
                return a is b
            return a / scale_lo == b / scale_hi

        quantized_equal = all(_quant_equal(ch) for ch in CHANNELS)
        if quantized_equal:
            flip = {
                "from": cls_lo,
                "to": cls_hi,
                "attribution": "weighted_sum_accumulation",
                "evidence": {
                    "quantized_inputs_equal": True,
                    "weighted_sum_diff": ws_diff,
                    "reason": "quantized inputs identical across tiers; "
                              "divergence first appears in the weighted sum",
                },
            }
        else:
            diff_channels = [ch for ch in CHANNELS
                             if not _quant_equal(ch)]
            flip = {
                "from": cls_lo,
                "to": cls_hi,
                "attribution": "quantization_truncation",
                "evidence": {
                    "quantized_inputs_equal": False,
                    "diverging_channels": diff_channels,
                    "quantize_diffs": {ch: quant_diffs[ch] for ch in diff_channels},
                    "reason": "quantized inputs differ across tiers; "
                              "truncation at the quantize stage propagates downstream",
                },
            }

    return {
        "stage_diffs": stage_diffs,
        "first_exceed_stage": first_exceed,
        "classification": {"tier_config": cls_lo, "tier_higher": cls_hi},
        "flip": flip,
    }
