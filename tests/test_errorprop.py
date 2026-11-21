from workbench.errorprop import compare_tiers, higher_tier


def cfg():
    return {
        "precision": {"frac_bits": 12, "quantum": 2,
                      "allowed_frac_bits": [8, 12, 16, 20]},
        "tolerances": {"abs": 1e-6, "rel": 1e-6},
        "sentiment_threshold": 0.5,
    }


def make_tier(quantize_vals, confidence):
    return {
        "normalize": {"text": 0.5, "acoustic": 0.5, "tool": 0.5},
        "quantize": quantize_vals,
        "weighted_sum": {"accumulator": 1, "pre_clip_float": confidence},
        "clip": {"confidence_int": int(confidence * 4096), "confidence": confidence},
    }


def test_higher_tier_selection():
    assert higher_tier(cfg()) == (16, 2)
    top = cfg()
    top["precision"]["frac_bits"] = 20
    assert higher_tier(top) == (20, 2)


def test_first_exceed_stage_flagged():
    lo = make_tier({"text": 100, "acoustic": 100, "tool": 100}, 0.6)
    hi = make_tier({"text": 100, "acoustic": 100, "tool": 100}, 0.6)
    hi["normalize"]["text"] = 0.5 + 1e-3  # big normalize diff
    result = compare_tiers(lo, hi, cfg())
    assert result["first_exceed_stage"] == "normalize"


def test_flip_attributed_to_quantization_truncation():
    # tier-16 integers are on a 2**16 scale; 32785/65536 differs from 2048/4096
    lo = make_tier({"text": 2048, "acoustic": 2048, "tool": 2048}, 0.4999)
    hi = make_tier({"text": 32785, "acoustic": 32768, "tool": 32768}, 0.5001)
    result = compare_tiers(lo, hi, cfg())
    flip = result["flip"]
    assert flip is not None
    assert flip["attribution"] == "quantization_truncation"
    assert flip["evidence"]["quantized_inputs_equal"] is False
    assert flip["evidence"]["diverging_channels"] == ["text"]


def test_flip_attributed_to_weighted_sum_accumulation():
    # identical quantized inputs (2048/4096 == 32768/65536), sums differ
    lo = make_tier({"text": 2048, "acoustic": 2048, "tool": 2048}, 0.4999)
    hi = make_tier({"text": 32768, "acoustic": 32768, "tool": 32768}, 0.5001)
    result = compare_tiers(lo, hi, cfg())
    flip = result["flip"]
    assert flip is not None
    assert flip["attribution"] == "weighted_sum_accumulation"
    assert flip["evidence"]["quantized_inputs_equal"] is True
    assert flip["evidence"]["weighted_sum_diff"] > 0


def test_no_flip_when_classifications_match():
    lo = make_tier({"text": 1, "acoustic": 1, "tool": 1}, 0.7)
    hi = make_tier({"text": 1, "acoustic": 1, "tool": 1}, 0.7001)
    result = compare_tiers(lo, hi, cfg())
    assert result["flip"] is None


def test_flip_attribution_through_real_fusion_path():
    """End-to-end flip attribution derived from the actual fusion chain."""
    from workbench import fusion
    from workbench.errorprop import compare_tiers

    cfg = {
        "weights": {"text": 0.3, "acoustic": 0.2, "tool": 0.5},
        "temperature": 1.0, "bias": 0.0, "confidence_interval": [0.0, 1.0],
        "precision": {"frac_bits": 12, "quantum": 2,
                      "allowed_frac_bits": [8, 12, 16, 20]},
        "tolerances": {"abs": 1e-9, "rel": 1e-9},
        "sentiment_threshold": 0.5,
    }

    def tier(normalized, bits):
        fused = fusion.fuse_turn(normalized, cfg, bits, 2)
        return {
            "normalize": normalized,
            "quantize": fused["quantized"],
            "weighted_sum": fused["weighted_sum"],
            "clip": {"confidence_int": fused["confidence_int"],
                     "confidence": fused["confidence"]},
        }

    # Case 1: quantized inputs identical across tiers, weight rounding in
    # the weighted sum flips the verdict -> weighted_sum_accumulation.
    norm = {"text": 2080 / 4096, "acoustic": 1998 / 4096, "tool": 2048 / 4096}
    result = compare_tiers(tier(norm, 12), tier(norm, 16), cfg)
    assert result["flip"] is not None
    assert result["flip"]["attribution"] == "weighted_sum_accumulation"
    assert result["flip"]["evidence"]["quantized_inputs_equal"] is True

    # Case 2: a value just below a quantization boundary rounds up at tier
    # 12 but stays below at tier 16 -> quantization_truncation.
    cfg2 = dict(cfg, weights={"text": 1.0, "acoustic": 0.0, "tool": 0.0})
    norm2 = {"text": 0.499755859375, "acoustic": 0.0, "tool": 0.0}
    result2 = compare_tiers(tier(norm2, 12), tier(norm2, 16), cfg2)
    # note: tier() closes over cfg; rebuild tiers with cfg2 weights
    def tier2(normalized, bits):
        fused = fusion.fuse_turn(normalized, cfg2, bits, 2)
        return {
            "normalize": normalized,
            "quantize": fused["quantized"],
            "weighted_sum": fused["weighted_sum"],
            "clip": {"confidence_int": fused["confidence_int"],
                     "confidence": fused["confidence"]},
        }
    result2 = compare_tiers(tier2(norm2, 12), tier2(norm2, 16), cfg2)
    assert result2["flip"] is not None
    assert result2["flip"]["attribution"] == "quantization_truncation"
    assert result2["flip"]["evidence"]["diverging_channels"] == ["text"]
