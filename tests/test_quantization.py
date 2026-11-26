from fractions import Fraction

from arg_review.quantization import boundary_information, quantize_score


def test_width_and_level_change_fixed_intermediates():
    score = Fraction(6137, 10000)
    coarse_d3 = quantize_score(
        score, {"frac_digits": 3, "level": "standard",
                "level_multipliers": {"standard": 5}}
    )
    fine_d4 = quantize_score(
        score, {"frac_digits": 4, "level": "fine",
                "level_multipliers": {"fine": 1}}
    )
    coarse_q25 = quantize_score(
        score, {"frac_digits": 3, "level": "coarse",
                "level_multipliers": {"coarse": 25}}
    )
    assert coarse_d3["unquantized_fixed_int"] != fine_d4["unquantized_fixed_int"]
    assert coarse_d3["scale"] == 1000
    assert fine_d4["scale"] == 10000
    assert coarse_d3["quantum"] == 5
    assert coarse_q25["quantum"] == 25
    assert coarse_d3["quantized_fixed_int"] != coarse_q25["quantized_fixed_int"]
    assert coarse_d3["rounding_direction"] in {"up", "down", "exact", "up_tie"}
    assert fine_d4["high_width_digits"] == 5


def test_boundary_uses_half_quantum_and_high_width_reference():
    boundary = boundary_information(
        Fraction(6001, 10000),
        600,
        {"quantum": 5, "scale": 1000,
         "high_width_digits": 4,
         "high_width_reference_value": "0.6001"},
    )
    assert boundary["boundary"] is True
    assert boundary["high_width_reference_value"] == "0.6001"

    far = boundary_information(
        Fraction(900, 1000),
        600,
        {"quantum": 5, "scale": 1000,
         "high_width_digits": 4,
         "high_width_reference_value": "0.9000"},
    )
    assert far["boundary"] is False
