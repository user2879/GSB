import pytest

from chaindiag.errors import E_CHAIN_FORMAT, ChainFormatError
from chaindiag.exchange import parse_e1_text
from chaindiag.model import normalize_boundaries, validate_chain, validate_config, normalize_tokens
from chaindiag.samplegen import default_bundle


def test_default_sample_validates():
    bundle = default_bundle()
    cfg = validate_config(bundle["config"])
    ops, _ = validate_chain(bundle["chain_e2"])
    tokens = normalize_tokens(bundle["tokens"], cfg["vocab_size"])
    boundaries = normalize_boundaries(bundle["segments"], len(tokens))
    assert [op["seq"] for op in ops] == [0, 1, 2, 3]
    assert boundaries == [0, 3, 6, 8]


def test_config_enums_and_attention_dimension():
    bad = default_bundle()["config"]
    bad["frac_bits"] = 7
    with pytest.raises(ChainFormatError) as exc:
        validate_config(bad)
    assert exc.value.issues[0].code == E_CHAIN_FORMAT
    assert "frac_bits" in exc.value.issues[0].field

    bad = default_bundle()["config"]
    bad["heads"] = 3
    with pytest.raises(ChainFormatError):
        validate_config(bad)


def test_boundaries_must_be_strict_and_cover_all_tokens():
    with pytest.raises(ChainFormatError) as exc:
        normalize_boundaries([0, 3, 3, 8], 8)
    assert "严格递增" in exc.value.issues[0].message

    with pytest.raises(ChainFormatError) as exc:
        normalize_boundaries([0, 3, 6, 7], 8)
    assert "末边界" in exc.value.issues[0].message


def test_operator_duplicate_gap_reference_and_cycle():
    def chain(ops):
        return {"version": "e2-1.0", "operators": ops}

    with pytest.raises(ChainFormatError) as exc:
        validate_chain(chain([
            {"seq": 0, "kind": "embedding", "inputs": []},
            {"seq": 0, "kind": "layer_norm", "inputs": [2]},
        ]))
    assert any("重复" in i.message for i in exc.value.issues)

    with pytest.raises(ChainFormatError) as exc:
        validate_chain(chain([
            {"seq": 0, "kind": "embedding", "inputs": []},
            {"seq": 2, "kind": "layer_norm", "inputs": [0]},
        ]))
    assert any("连续" in i.message for i in exc.value.issues)

    with pytest.raises(ChainFormatError) as exc:
        validate_chain(chain([
            {"seq": 0, "kind": "embedding", "inputs": []},
            {"seq": 1, "kind": "segment_pool", "inputs": [9]},
        ]))
    assert any("引用 9 不存在" in i.message for i in exc.value.issues)

    with pytest.raises(ChainFormatError) as exc:
        validate_chain(chain([
            {"seq": 0, "kind": "segment_pool", "inputs": [1]},
            {"seq": 1, "kind": "layer_norm", "inputs": [0]},
        ]))
    assert any("环" in i.message for i in exc.value.issues)


def test_unknown_operator_tree_and_bad_version_are_chain_format():
    with pytest.raises(ChainFormatError) as exc:
        validate_chain({"version": "e2-1.0", "operators": [{"seq": 0, "kind": "future_op", "inputs": []}]})
    assert exc.value.issues[0].code == E_CHAIN_FORMAT
    assert exc.value.issues[0].field.endswith("kind")

    with pytest.raises(ChainFormatError) as exc:
        validate_chain({"version": "eX-9", "operators": [{"seq": 0, "kind": "embedding", "inputs": []}]})
    assert "版本串解析失败" in exc.value.issues[0].message


def test_chain_sequence_must_be_unique():
    chain = {
        "chains": [
            {"seq": 1, "operators": [{"seq": 0, "kind": "embedding", "inputs": []}]},
            {"seq": 1, "operators": [{"seq": 0, "kind": "embedding", "inputs": []}]},
        ]
    }
    with pytest.raises(ChainFormatError) as exc:
        validate_chain(chain)
    assert any("链序号" in issue.message for issue in exc.value.issues)


def test_e1_short_and_bad_version_report_line_and_field():
    with pytest.raises(ChainFormatError) as exc:
        parse_e1_text("E1000000100\nXX001001000016\n", location="sample.e1")
    fields = {(i.line, i.field) for i in exc.value.issues}
    assert (1, "scale_permille") in fields or (1, "int_bits") in fields
    assert (2, "version") in fields
    assert all(i.code == E_CHAIN_FORMAT for i in exc.value.issues)
