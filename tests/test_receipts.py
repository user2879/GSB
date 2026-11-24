from __future__ import annotations

import pytest

from batch_diagnostic.errors import ChainError, FieldError
from batch_diagnostic.receipt import (
    DEFAULT_PARAM_VERSION,
    DEFAULT_WORKER_ID,
    batch_summary_digest,
    decode_receipt,
    encode_receipt,
    validate_chain,
)

KNOWN = {"req-1", "req-2"}


def test_r1_defaults_are_explicit_and_traceable():
    receipt = decode_receipt(
        {
            "format_version": "r1",
            "batch_id": "b",
            "request_id": "req-1",
            "output_digest": "d",
        }
    )
    assert receipt.source_version == "r1"
    assert receipt.param_version == DEFAULT_PARAM_VERSION
    assert receipt.worker_id == DEFAULT_WORKER_ID
    names = receipt.filled_field_names()
    assert set(names) == {
        "param_version",
        "elapsed_ms",
        "worker_id",
        "rank",
        "duplicate_ref",
        "param_digest",
    }
    assert all(record.rationale for record in receipt.filled_fields)


def test_r2_fills_only_r3_fields():
    receipt = decode_receipt(
        {
            "format_version": "r2",
            "batch_id": "b",
            "request_id": "req-1",
            "output_digest": "d",
            "param_version": 3,
            "elapsed_ms": 5,
            "worker_id": "w-a",
        }
    )
    assert receipt.param_version == 3
    assert receipt.filled_field_names() == ("rank", "duplicate_ref", "param_digest")


def test_unknown_version_and_missing_fields_are_e_field():
    with pytest.raises(FieldError) as exc:
        decode_receipt({"format_version": "r9", "batch_id": "b"})
    assert exc.value.code == "E_FIELD"
    with pytest.raises(FieldError) as exc:
        decode_receipt({"format_version": "r1", "batch_id": "b"})
    assert exc.value.context["field"] == "request_id"


def test_negative_elapsed_rejected():
    with pytest.raises(FieldError) as exc:
        decode_receipt(
            {
                "format_version": "r2",
                "batch_id": "b",
                "request_id": "req-1",
                "output_digest": "d",
                "param_version": 1,
                "elapsed_ms": -1,
                "worker_id": "w",
            }
        )
    assert exc.value.context["field"] == "elapsed_ms"


def test_rank_bounds_and_dangling_ref_at_decode():
    base = {
        "format_version": "r3",
        "batch_id": "b",
        "request_id": "req-1",
        "output_digest": "d",
        "param_version": 1,
        "elapsed_ms": 1,
        "worker_id": "w",
        "rank": -1,
        "duplicate_ref": None,
        "param_digest": "p",
    }
    with pytest.raises(FieldError) as exc:
        decode_receipt(dict(base), KNOWN)
    assert exc.value.context["field"] == "rank"
    base["rank"] = 0
    base["duplicate_ref"] = "ghost"
    with pytest.raises(FieldError) as exc:
        decode_receipt(dict(base), KNOWN)
    assert exc.value.context["field"] == "duplicate_ref"


def test_validate_chain_e_chain_after_tamper():
    receipt = decode_receipt(
        {
            "format_version": "r3",
            "batch_id": "b",
            "request_id": "req-1",
            "output_digest": "d",
            "param_version": 1,
            "elapsed_ms": 1,
            "worker_id": "w",
            "rank": 0,
            "duplicate_ref": "ghost",
            "param_digest": "p",
        }
    )
    with pytest.raises(ChainError) as exc:
        validate_chain([receipt])
    assert exc.value.code == "E_CHAIN"


def test_round_trip_codecs_preserve_historical_shapes():
    normalized = decode_receipt(
        {
            "format_version": "r1",
            "batch_id": "b",
            "request_id": "req-1",
            "output_digest": "d",
        }
    )
    r1 = encode_receipt(normalized, "r1")
    assert set(r1) == {"format_version", "batch_id", "request_id", "output_digest"}
    r3 = encode_receipt(normalized, "r3")
    assert {"rank", "duplicate_ref", "param_digest"}.issubset(r3)


def test_batch_summary_order_independent():
    def make(rid):
        return decode_receipt(
            {
                "format_version": "r1",
                "batch_id": "b",
                "request_id": rid,
                "output_digest": "d-" + rid,
            }
        )

    left = [make("req-1"), make("req-2")]
    right = [make("req-2"), make("req-1")]
    assert batch_summary_digest(left) == batch_summary_digest(right)
