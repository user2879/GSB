import pytest

from gsb.errors import GSBError
from gsb.hotupdate import compute_param_digest
from gsb.receipts import (
    NormContext,
    compute_output_digest,
    decode_and_normalize,
    decode_frame,
    encode_frame,
)

PARAM_DIGEST = compute_param_digest(202, 8, 16, "fp16")
CTX = NormContext(
    sample_ids=("q-001", "q-002"),
    param_digest=PARAM_DIGEST,
    param_version=1,
    quant_profile="fp16",
)


def _data(**extra):
    base = {
        "batch_id": "batch-1",
        "route": "gray",
        "output_digest": compute_output_digest(
            "batch-1", "gray", PARAM_DIGEST, ["q-001", "q-002"]
        ),
    }
    base.update(extra)
    return base


def test_roundtrip_v1_fill_basis():
    frame = encode_frame("v1", _data())
    receipt = decode_and_normalize(frame, CTX)
    assert receipt.source_format == "v1"
    assert receipt.param_digest == PARAM_DIGEST
    assert receipt.param_version == 1
    assert receipt.elapsed_ms == 0
    assert receipt.sample_ids == ("q-001", "q-002")
    assert set(receipt.filled_fields) == {
        "param_digest",
        "quant_profile",
        "elapsed_ms",
        "sample_ids",
        "param_version",
    }
    assert receipt.fill_basis["param_digest"] == "dispatch:param_digest"
    assert receipt.fill_basis["quant_profile"] == "backend_decl:quant_profile"
    assert receipt.fill_basis["elapsed_ms"] == "default:elapsed_ms=0"
    assert receipt.fill_basis["sample_ids"] == "dispatch:sample_ids"
    assert receipt.fill_basis["param_version"] == "dispatch:param_version"


def test_roundtrip_v2_partial_fill():
    frame = encode_frame(
        "v2",
        _data(param_digest=PARAM_DIGEST, quant_profile="fp16", elapsed_ms=5),
    )
    receipt = decode_and_normalize(frame, CTX)
    assert receipt.source_format == "v2"
    assert receipt.elapsed_ms == 5
    assert set(receipt.filled_fields) == {"sample_ids", "param_version"}


def test_roundtrip_v3_no_fill():
    frame = encode_frame(
        "v3",
        _data(
            param_digest=PARAM_DIGEST,
            quant_profile="fp16",
            elapsed_ms=2,
            sample_ids=["q-002", "q-001"],
            param_version=1,
        ),
    )
    receipt = decode_and_normalize(frame, CTX)
    assert receipt.source_format == "v3"
    assert receipt.filled_fields == []
    assert receipt.sample_ids == ("q-001", "q-002")  # 归一后排序


def test_length_mismatch_e_format():
    frame = encode_frame("v1", _data()).replace("LEN ", "LEN 999", 1)
    with pytest.raises(GSBError) as exc:
        decode_frame(frame)
    assert exc.value.code == "E_FORMAT"
    assert exc.value.context["field"] == "length"


def test_unknown_version_e_format():
    frame = encode_frame("v1", _data()).replace("GSB-RECEIPT v1", "GSB-RECEIPT v9", 1)
    with pytest.raises(GSBError) as exc:
        decode_frame(frame)
    assert exc.value.code == "E_FORMAT"
    assert exc.value.context["field"] == "format_version"


def test_missing_required_field_locates_batch_and_field():
    data = _data(param_digest=PARAM_DIGEST, quant_profile="fp16")  # 缺 elapsed_ms
    frame = encode_frame("v2", data)
    with pytest.raises(GSBError) as exc:
        decode_frame(frame)
    assert exc.value.code == "E_FORMAT"
    assert exc.value.context["batch_id"] == "batch-1"
    assert exc.value.context["field"] == "elapsed_ms"


def test_digest_binding_violation():
    data = _data()
    data["output_digest"] = "0" * 64
    frame = encode_frame("v1", data)
    with pytest.raises(GSBError) as exc:
        decode_and_normalize(frame, CTX)
    assert exc.value.code == "E_FORMAT"
    assert exc.value.context["batch_id"] == "batch-1"
    assert exc.value.context["field"] == "output_digest"


def test_sample_files_decode():
    for name, fmt in [("receipt_v1.txt", "v1"), ("receipt_v2.txt", "v2"), ("receipt_v3.txt", "v3")]:
        frame = open(f"samples/receipts/{name}").read()
        version, _ = decode_frame(frame)
        assert version == fmt
        receipt = decode_and_normalize(frame, CTX)
        assert receipt.batch_id == "batch-1"


@pytest.mark.parametrize(
    "name, field",
    [
        ("receipt_bad_len.txt", "length"),
        ("receipt_unknown_version.txt", "format_version"),
        ("receipt_missing_field.txt", "elapsed_ms"),
    ],
)
def test_bad_sample_files_e_format(name, field):
    frame = open(f"samples/receipts/{name}").read()
    with pytest.raises(GSBError) as exc:
        decode_frame(frame)
    assert exc.value.code == "E_FORMAT"
    assert exc.value.context["field"] == field
