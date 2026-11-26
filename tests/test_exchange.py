from chaindiag.exchange import e1_to_e2, e2_to_e1, format_e1_line, parse_e1_text
from chaindiag.samplegen import default_bundle


def test_e1_fixed_width_line_contract():
    line = format_e1_line({"seq": 7, "scale_permille": 1234, "int_bits": 24})
    assert line == "E10007001234024"
    assert len(line) == 15
    parsed = parse_e1_text(line)
    assert parsed[0]["seq"] == 7
    assert parsed[0]["scale_permille"] == 1234
    assert parsed[0]["int_bits"] == 24


def test_e1_to_e2_exact_permille_and_scheme_mapping():
    e2 = e1_to_e2("E10001001125016\n")
    op = e2["operators"][0]
    assert op["scale_permille"] == 1125
    assert op["scale_float"] == 1.125
    assert op["scale_rational"] == "9/8"
    assert op["quant_scheme"] == "qint16"
    assert op["exchange_meta"]["precision_tag"] == "e1-exact:1125/1000"


def test_e2_e1_e2_bypass_restores_covered_information():
    original = default_bundle()["chain_e2"]
    e1_text, sidecar = e2_to_e1(original)
    assert all(len(line) == 15 for line in e1_text.strip().splitlines())
    assert sidecar["schema"] == "chaindiag-e1-bypass-v1"

    restored = e1_to_e2(e1_text, sidecar)
    attention = next(op for op in restored["operators"] if op["seq"] == 2)
    original_attention = original["operators"][2]
    assert attention["inputs"] == [0, 1]
    assert attention["attention_snapshot"] == original_attention["attention_snapshot"]
    assert attention["params"] == original_attention["params"]
    assert attention["restored_from_bypass"] is True
    assert attention["exchange_meta"]["restored_precision_tag"] == "sample-lossy-1.0004"

    entry = sidecar["entries"]["2"]
    assert entry["scale_round_residual"]["exact"] == "1/2500"
    assert entry["overwritten_references"]["inputs"] == [0, 1]

    # Parsed e1 facts remain visible even after bypass restoration.
    assert attention["scale_permille"] == 1000
    assert attention["quant_scheme"] == "qint24"
