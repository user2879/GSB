from __future__ import annotations

import hashlib
import struct

import pytest

from tunnel_interop.errors import E_FIELD, E_FORMAT, TunnelError
from tunnel_interop.fragmenter import build_bundle, context_for_plan, migrate_all_variants
from tunnel_interop.util import sha256_json
from tunnel_interop.wire import (
    FORMAT_TO_VERSION,
    MAGIC,
    WireContext,
    compute_checksum,
    compute_header_digest,
    decode_fragment,
    encode_fragment,
    migrate_fragment,
)


def assert_error(code: str, path: str, function) -> None:
    with pytest.raises(TunnelError) as captured:
        function()
    assert captured.value.code == code
    assert captured.value.path == path


def test_three_formats_migrate_to_equal_normalized_fields(spec):
    bundle = build_bundle(spec)
    for message in bundle.messages:
        for fragment in message.fragments:
            variants = migrate_all_variants(fragment, message=message.plan)
            scores = {name: sha256_json(value.semantic_key()) for name, value in variants.items()}
            assert len(set(scores.values())) == 1
            assert variants["f1"].padded_fields
            assert "total_fragments" in variants["f1"].padding_basis
            assert set(variants["f2"].padded_fields) == {
                "encapsulation_version",
                "fragment_header_digest",
                "retry_count",
            }
            assert variants["f3"].padded_fields == ()


def test_f1_f2_f3_each_decode_with_same_semantics(spec):
    bundle = build_bundle(spec)
    record = bundle.messages[0].fragments[1]
    context = context_for_plan(bundle.messages[0].plan, record.total_fragments)
    normalized = []
    for name, wire in record.format_variants.items():
        assert wire[4] == FORMAT_TO_VERSION[name]
        normalized.append(decode_fragment(wire, context))
    assert len({sha256_json(item.semantic_key()) for item in normalized}) == 1


def test_unknown_format_version_is_e_format(spec):
    bundle = build_bundle(spec)
    record = bundle.messages[0].fragments[0]
    wire = bytearray(record.wire)
    wire[4] = 99
    context = context_for_plan(bundle.messages[0].plan, record.total_fragments)
    assert_error(E_FORMAT, "fragment.format", lambda: decode_fragment(bytes(wire), context))


def test_bad_magic_is_e_format(spec):
    bundle = build_bundle(spec)
    record = bundle.messages[0].fragments[0]
    context = context_for_plan(bundle.messages[0].plan, record.total_fragments)
    wire = b"BAD!" + record.wire[4:]
    assert_error(E_FORMAT, "fragment.magic", lambda: decode_fragment(wire, context))


def test_missing_required_field_is_e_field(spec):
    bundle = build_bundle(spec)
    record = bundle.messages[0].fragments[0]
    context = context_for_plan(bundle.messages[0].plan, record.total_fragments)
    assert_error(E_FIELD, "fragment.payload", lambda: decode_fragment(record.wire[:-10], context))


def _offsets(spec, record):
    tunnel_length = len(record.tunnel_id.encode("utf-8"))
    common_end = 5 + 2 + tunnel_length + 12
    return {
        "index": 5 + 2 + tunnel_length,
        "offset": 5 + 2 + tunnel_length + 4,
        "payload_length": 5 + 2 + tunnel_length + 8,
        "total": common_end,
        "message_id": common_end + 4,
        "generated": common_end + 36,
        "checksum": common_end + 44,
        "encapsulation_version": common_end + 76,
        "retry": common_end + 78,
        "digest": common_end + 82,
        "payload": common_end + 114,
    }


def test_total_fragments_zero_is_e_field(spec):
    bundle = build_bundle(spec)
    message = bundle.messages[0]
    record = message.fragments[0]
    wire = bytearray(record.format_variants["f2"])
    offsets = _offsets(spec, record)
    struct.pack_into(">I", wire, offsets["total"], 0)
    context = context_for_plan(message.plan, record.total_fragments)
    assert_error(E_FIELD, "fragment.total_fragments", lambda: decode_fragment(bytes(wire), context))


def test_index_must_be_less_than_total(spec):
    bundle = build_bundle(spec)
    message = bundle.messages[0]
    record = message.fragments[0]
    wire = bytearray(record.format_variants["f1"])
    offsets = _offsets(spec, record)
    struct.pack_into(">I", wire, offsets["index"], 1)
    context = WireContext(
        fragment_size=message.plan.fragment_size,
        effective_payload_limit=message.plan.endpoint.effective_payload_limit,
        max_message_bytes=8 * 1024 * 1024,
        total_fragments=1,
        message_id=message.plan.message_id,
        generated_at_ms=message.plan.generated_at_ms,
    )
    assert_error(E_FIELD, "fragment.fragment_index", lambda: decode_fragment(bytes(wire), context))


def test_offset_alignment_is_e_field(spec):
    bundle = build_bundle(spec)
    message = bundle.messages[0]
    record = message.fragments[0]
    wire = bytearray(record.format_variants["f1"])
    offsets = _offsets(spec, record)
    struct.pack_into(">I", wire, offsets["offset"], 3)
    context = context_for_plan(message.plan, 1)
    assert_error(E_FIELD, "fragment.fragment_offset", lambda: decode_fragment(bytes(wire), context))


def test_offset_plus_length_limit_is_e_field(spec):
    bundle = build_bundle(spec)
    message = bundle.messages[0]
    record = message.fragments[0]
    context = WireContext(
        fragment_size=message.plan.fragment_size,
        effective_payload_limit=message.plan.endpoint.effective_payload_limit,
        max_message_bytes=10,
        total_fragments=1,
        message_id=message.plan.message_id,
        generated_at_ms=message.plan.generated_at_ms,
    )
    assert_error(
        E_FIELD,
        "fragment.fragment_offset",
        lambda: decode_fragment(record.format_variants["f1"], context),
    )


def test_fragment_length_must_not_exceed_fragment_size(spec):
    bundle = build_bundle(spec)
    message = bundle.messages[0]
    record = message.fragments[0]
    context = WireContext(
        fragment_size=message.plan.fragment_size - 1,
        effective_payload_limit=message.plan.endpoint.effective_payload_limit,
        max_message_bytes=8 * 1024 * 1024,
        total_fragments=record.total_fragments,
        message_id=message.plan.message_id,
        generated_at_ms=message.plan.generated_at_ms,
    )
    assert_error(
        E_FIELD,
        "fragment.payload_length",
        lambda: decode_fragment(record.format_variants["f1"], context),
    )


def test_negative_retry_is_e_field(spec):
    bundle = build_bundle(spec)
    message = bundle.messages[0]
    record = message.fragments[0]
    wire = bytearray(record.format_variants["f3"])
    offsets = _offsets(spec, record)
    struct.pack_into(">i", wire, offsets["retry"], -1)
    context = context_for_plan(message.plan, record.total_fragments)
    assert_error(E_FIELD, "fragment.retry_count", lambda: decode_fragment(bytes(wire), context))


def test_zero_encapsulation_version_is_e_field(spec):
    bundle = build_bundle(spec)
    message = bundle.messages[0]
    record = message.fragments[0]
    wire = bytearray(record.format_variants["f3"])
    offsets = _offsets(spec, record)
    struct.pack_into(">H", wire, offsets["encapsulation_version"], 0)
    context = context_for_plan(message.plan, record.total_fragments)
    assert_error(
        E_FIELD,
        "fragment.encapsulation_version",
        lambda: decode_fragment(bytes(wire), context),
    )


def test_checksum_mismatch_is_e_field(spec):
    bundle = build_bundle(spec)
    message = bundle.messages[0]
    record = message.fragments[0]
    wire = bytearray(record.format_variants["f2"])
    offsets = _offsets(spec, record)
    wire[offsets["checksum"]] ^= 0xFF
    context = context_for_plan(message.plan, record.total_fragments)
    assert_error(E_FIELD, "fragment.checksum", lambda: decode_fragment(bytes(wire), context))


def test_header_digest_mismatch_is_e_field(spec):
    bundle = build_bundle(spec)
    message = bundle.messages[0]
    record = message.fragments[0]
    wire = bytearray(record.format_variants["f3"])
    offsets = _offsets(spec, record)
    wire[offsets["digest"]] ^= 0xFF
    context = context_for_plan(message.plan, record.total_fragments)
    assert_error(
        E_FIELD,
        "fragment.fragment_header_digest",
        lambda: decode_fragment(bytes(wire), context),
    )
