import pytest

from ckpt_diag.errors import FieldError, FormatError
from ckpt_diag.receipts import (
    decode_receipt, encode_receipt, normalize_mixed_batch)


def test_c1_backfills_legacy_fields():
    line = encode_receipt(1, batch_id="batch-001", shard_id="shard-alpha",
                          loss_summary="0123456789abcdef")
    receipt = decode_receipt(line)
    assert receipt.source_format == 1
    assert set(receipt.backfilled_fields) == {
        "generation", "param_version", "elapsed_ms", "sample_ids",
        "param_summary", "lease_id"}
    assert receipt.backfill_basis["lease_id"] == "no_lease_legacy"
    assert receipt.param_version == 0 and receipt.elapsed_ms == 0
    assert receipt.lease_id == "NO_LEASE"


def test_c2_roundtrip_and_partial_backfill():
    line = encode_receipt(2, batch_id="batch-002", shard_id="shard-alpha",
                          loss_summary="0123456789abcdef", param_version=3,
                          generation=2, elapsed_ms=12)
    receipt = decode_receipt(line)
    assert receipt.source_format == 2
    assert receipt.param_version == 3
    assert set(receipt.backfilled_fields) == {
        "sample_ids", "param_summary", "lease_id"}


def test_c3_full_fields():
    line = encode_receipt(3, batch_id="batch-003", shard_id="shard-alpha",
                          loss_summary="0123456789abcdef", param_version=1,
                          generation=1, elapsed_ms=4, sample_ids=["s-1", "s-0"],
                          param_summary="abcdef0123456789", lease_id="lease-1")
    receipt = decode_receipt(line)
    assert receipt.sample_ids == ("s-0", "s-1")
    assert receipt.backfilled_fields == ()


def test_unknown_version_is_format_error():
    with pytest.raises(FormatError) as exc:
        decode_receipt('{"format_version": 7, "batch_id": "b", "shard_id": "s"}')
    assert exc.value.code == "E_FORMAT"


def test_missing_required_is_field_error():
    line = '{"format_version": 2, "batch_id": "batch-009", "shard_id": "shard-alpha"}'
    with pytest.raises(FieldError) as exc:
        decode_receipt(line)
    assert exc.value.code == "E_FIELD"


def test_negative_elapsed_is_field_error():
    line = encode_receipt(2, batch_id="batch-010", shard_id="shard-alpha",
                          loss_summary="0123456789abcdef", elapsed_ms=-1)
    with pytest.raises(FieldError) as exc:
        decode_receipt(line)
    assert exc.value.field == "elapsed_ms"


def test_width_underflow_is_field_error():
    line = encode_receipt(3, batch_id="tiny", shard_id="shard-alpha",
                          loss_summary="0123456789abcdef", lease_id="lease-9")
    with pytest.raises(FieldError) as exc:
        decode_receipt(line)
    assert exc.value.field == "batch_id"


def test_mixed_batch_arbitration_prefers_higher_param_then_format():
    c1 = decode_receipt(encode_receipt(
        1, batch_id="batch-mix", shard_id="shard-alpha",
        loss_summary="0123456789abcdef"))
    c2 = decode_receipt(encode_receipt(
        2, batch_id="batch-mix", shard_id="shard-alpha",
        loss_summary="0123456789abcdef", param_version=1, elapsed_ms=5))
    c3 = decode_receipt(encode_receipt(
        3, batch_id="batch-mix", shard_id="shard-alpha",
        loss_summary="0123456789abcdef", param_version=1, elapsed_ms=5,
        sample_ids=["s-0"], param_summary="abcdef0123456789",
        lease_id="lease-001"))
    accepted, rejected = normalize_mixed_batch([c1, c2, c3])
    assert accepted.source_format == 3
    assert {r.source_format for r in rejected} == {1, 2}
    assert all("superseded" in r.rejected_reason for r in rejected)


def test_mixed_batch_higher_param_version_wins_over_format():
    low_fmt_high_pv = decode_receipt(encode_receipt(
        2, batch_id="batch-pv", shard_id="shard-alpha",
        loss_summary="0123456789abcdef", param_version=9, elapsed_ms=1))
    high_fmt_low_pv = decode_receipt(encode_receipt(
        3, batch_id="batch-pv", shard_id="shard-alpha",
        loss_summary="0123456789abcdef", param_version=1, elapsed_ms=1,
        sample_ids=[], param_summary="abcdef0123456789", lease_id="lease-002"))
    accepted, rejected = normalize_mixed_batch([high_fmt_low_pv, low_fmt_high_pv])
    assert accepted.source_format == 2 and accepted.param_version == 9
    assert len(rejected) == 1
