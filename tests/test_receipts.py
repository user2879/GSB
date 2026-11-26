import pytest

from edge_cache.errors import EdgeCacheError
from edge_cache.receipts import (
    C1,
    C2,
    C3,
    MISSING_MARKER,
    C1_LINE_WIDTH,
    Receipt,
    ReceiptFormatCodec,
    integrity_label_for,
    parse_c1,
    parse_c3,
    precision_label_for,
    serialize_c1,
    serialize_c3,
)


def _receipt(**overrides):
    base = Receipt(
        shard_id="s7",
        index=7,
        node_id="node-a",
        check_rate_permille=875,
        residual=42,
        check_rate_float=0.875042,
        quota_version="q3",
        quota_code=300,
        digest="abc123digest",
        duplicate_ref="dup-001",
        fixed_point=4,
        precision_label=precision_label_for(4, 42),
        clock_ms=1234,
    )
    base.integrity_label = integrity_label_for(base)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_c1_fixed_width_roundtrip_uses_sidecar():
    codec = ReceiptFormatCodec({"q1": 100, "q2": 200, "q3": 300})
    receipt = _receipt()
    line = codec.encode(receipt, C1)
    assert len(line) == C1_LINE_WIDTH
    parsed = parse_c1(line)
    assert parsed == {"index": 7, "check_rate_permille": 875, "quota_code": 300}

    restored = codec.decode(line, C1)
    assert restored.shard_id == receipt.shard_id
    assert restored.residual == receipt.residual
    assert restored.digest == receipt.digest
    assert restored.duplicate_ref == receipt.duplicate_ref
    assert restored.precision_label == receipt.precision_label
    assert restored.quota_version == "q3"
    assert restored.integrity_label == receipt.integrity_label


def test_c2_json_roundtrip():
    codec = ReceiptFormatCodec({"q3": 300})
    receipt = _receipt()
    line = codec.encode(receipt, C2)
    restored = codec.decode(line, C2)
    assert restored.check_rate_float == receipt.check_rate_float
    assert restored.quota_version == receipt.quota_version
    # c2 has no digest field, so promotion marks it explicitly missing.
    assert restored.digest is None


def test_c3_keyvalue_roundtrip_preserves_all_fields():
    codec = ReceiptFormatCodec({"q3": 300})
    receipt = _receipt()
    line = codec.encode(receipt, C3)
    raw = parse_c3(line)
    assert raw["digest"] == "abc123digest"
    assert raw["duplicate_ref"] == "dup-001"
    restored = codec.decode(line, C3)
    assert restored.to_dict() == receipt.to_dict()


def test_promotion_to_c3_uses_missing_marker_and_label():
    codec = ReceiptFormatCodec({"q3": 300})
    receipt = _receipt(digest=None, duplicate_ref=None)
    line = serialize_c3(receipt)
    raw = parse_c3(line)
    assert raw["digest"] == MISSING_MARKER
    assert raw["duplicate_ref"] == MISSING_MARKER
    assert raw["integrity_label"] == receipt.integrity_label


def test_tampered_c3_integrity_label_rejected():
    codec = ReceiptFormatCodec({"q3": 300})
    receipt = _receipt()
    line = codec.encode(receipt, C3)
    tampered = line.replace("875", "876", 1)
    with pytest.raises(EdgeCacheError):
        codec.decode(tampered, C3)


def test_c3_to_c1_and_back_is_lossless():
    codec = ReceiptFormatCodec({"q1": 100, "q2": 200, "q3": 300})
    receipt = _receipt()
    c1_line = codec.encode(receipt, C1)
    back = codec.decode(c1_line, C1)
    c3_again = codec.encode(back, C3)
    restored = codec.decode(c3_again, C3)
    assert restored.digest == receipt.digest
    assert restored.duplicate_ref == receipt.duplicate_ref
    assert restored.residual == receipt.residual
    assert restored.precision_label == receipt.precision_label
    assert restored.integrity_label == receipt.integrity_label


def test_oversized_c1_field_rejected():
    receipt = _receipt(index=123456789)
    with pytest.raises(EdgeCacheError):
        serialize_c1(receipt)
