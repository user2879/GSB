import pytest

from receipt_ledger import codec
from receipt_ledger.errors import CodecError

DIGEST = "a" * 64


@pytest.fixture()
def fields():
    return {
        "batch_id": "r-b0", "worker_id": "w1", "destination": "w2",
        "output_digest": DIGEST, "param_digest": "b" * 64,
        "quant": "fp32", "elapsed_ms": 5,
        "request_seqs": [3, 1, 2], "param_version": 2,
    }


@pytest.mark.parametrize("fmt,expected_keys", [
    ("r1", ("batch_id", "worker_id", "destination", "output_digest")),
    ("r2", ("param_digest", "quant", "elapsed_ms")),
    ("r3", ("request_seqs", "param_version")),
])
def test_roundtrip_and_version_fields(fields, fmt, expected_keys):
    raw = codec.encode_receipt(fmt, fields)
    decoded_fmt, obj, raw_len = codec.decode_receipt(raw)
    assert decoded_fmt == fmt and raw_len == len(raw)
    for key in expected_keys:
        assert key in obj
    if fmt == "r3":
        assert obj["request_seqs"] == [1, 2, 3]


def test_length_field_mismatch(fields):
    raw = codec.encode_receipt("r2", fields)
    tampered = raw[:5] + (raw[5] + 1).to_bytes(4, "big") + raw[9:]
    with pytest.raises(CodecError) as exc:
        codec.decode_receipt(tampered)
    assert exc.value.location["field"] == "payload_len"


def test_bad_magic_and_trailing_bytes(fields):
    raw = codec.encode_receipt("r1", fields)
    with pytest.raises(CodecError):
        codec.decode_receipt(b"XXXX" + raw[4:])
    with pytest.raises(CodecError):
        codec.decode_receipt(raw + b"\x00")


def test_illegal_field_type(fields):
    raw = codec.encode_receipt("r2", fields)
    import json
    obj = json.loads(raw[9:])
    obj["elapsed_ms"] = -3
    bad = raw[:9] + codec.canonical_json(obj)
    bad = bad[:5] + (len(bad) - 9).to_bytes(4, "big") + bad[9:]
    with pytest.raises(CodecError) as exc:
        codec.decode_receipt(bad)
    assert exc.value.location["field"] == "elapsed_ms"


def test_normalize_carries_source_filled_and_basis(fields):
    context = {
        "batch_id": "r-b0", "worker_id": "w1", "destination": "w2",
        "quant": "int8", "param_version": 9,
        "request_seqs": [1, 2, 3], "param_digest": "c" * 64,
    }
    raw = codec.encode_receipt("r1", fields)
    fmt, obj, raw_len = codec.decode_receipt(raw)
    view = codec.normalize_receipt(fmt, obj, context, raw_len)
    assert view["source_format"] == "r1"
    filled = {item["field"] for item in view["filled_fields"]}
    assert filled == {"param_digest", "quant", "elapsed_ms",
                      "request_seqs", "param_version"}
    basis = {item["field"]: item["basis"] for item in view["filled_fields"]}
    assert basis["elapsed_ms"].startswith("default:")
    assert basis["param_version"].startswith("context:")
    assert view["quant"] == "int8" and view["param_version"] == 9

    raw3 = codec.encode_receipt("r3", fields)
    fmt3, obj3, len3 = codec.decode_receipt(raw3)
    view3 = codec.normalize_receipt(fmt3, obj3, context, len3)
    assert view3["filled_fields"] == []

