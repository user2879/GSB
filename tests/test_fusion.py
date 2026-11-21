from fractions import Fraction

from workbench import codes, fusion


def base_config():
    return {
        "weights": {"text": 0.5, "acoustic": 0.3, "tool": 0.2},
        "temperature": 1.0,
        "bias": 0.0,
        "confidence_interval": [0.0, 1.0],
    }


def test_quantize_deterministic_and_exact():
    q1 = fusion.quantize(0.5, 12, 2)
    q2 = fusion.quantize(0.5, 12, 2)
    assert q1 == q2 == 2048
    assert fusion.quantize(1.0, 12, 2) == 4096
    # quantum rounding: nearest multiple of 2
    assert fusion.quantize(0.3001, 12, 2) % 2 == 0


def test_round_half_up():
    assert fusion.round_half_up(Fraction(1, 2)) == 1
    assert fusion.round_half_up(Fraction(3, 2)) == 2
    assert fusion.round_half_up(Fraction(-1, 2)) == -1
    assert fusion.round_half_up(Fraction(5, 4)) == 1


def test_weight_sum_zero_emits_code_and_no_default():
    cfg = base_config()
    cfg["weights"] = {"text": 0.0, "acoustic": 0.0, "tool": 0.0}
    result = fusion.fuse_turn({"text": 0.5, "acoustic": 0.5, "tool": 0.5}, cfg, 12, 2)
    assert codes.E_FUSE_WEIGHT_SUM_ZERO in result["codes"]
    assert result["confidence"] is None


def test_missing_evidence_emits_code():
    result = fusion.fuse_turn({"text": 0.5, "acoustic": None, "tool": 0.5},
                              base_config(), 12, 2)
    assert codes.E_FUSE_EVIDENCE_MISSING in result["codes"]
    assert result["missing"] == ["acoustic"]
    assert result["confidence"] is None


def test_clipping_emits_code_and_endpoint():
    cfg = base_config()
    cfg["bias"] = 5.0
    result = fusion.fuse_turn({"text": 0.9, "acoustic": 0.9, "tool": 0.9}, cfg, 12, 2)
    assert codes.E_FUSE_CLIPPED in result["codes"]
    assert result["clipped"] is True
    assert result["confidence"] == 1.0
    assert result["confidence_int"] == 4096


def test_fixed_point_integer_output():
    result = fusion.fuse_turn({"text": 0.6, "acoustic": 0.4, "tool": 0.5},
                              base_config(), 12, 2)
    assert isinstance(result["confidence_int"], int)
    scale = 1 << 12
    assert result["confidence"] == result["confidence_int"] / scale


def test_temperature_scales_sum():
    cfg = base_config()
    cfg["temperature"] = 2.0
    hot = fusion.fuse_turn({"text": 0.8, "acoustic": 0.8, "tool": 0.8}, cfg, 12, 2)
    cfg["temperature"] = 1.0
    cold = fusion.fuse_turn({"text": 0.8, "acoustic": 0.8, "tool": 0.8}, cfg, 12, 2)
    assert hot["confidence"] < cold["confidence"]
